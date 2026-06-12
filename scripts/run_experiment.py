r"""Deterministic experiment harness for the shielded boundary policies.

Runs one rollout of HORIZON epochs per (config, policy, seed) and writes the
paper-reported safety, response, load-dispersion, and throughput summaries as
JSONL, one line per run. Policy modes:

  - hold                     : a_ij = 0 for all edges (static-partition baseline).
  - fixed_band / safe_greedy : the deterministic shielded policies.
  - masked_ppo               : the trained CPAC policy (needs torch and a
                               checkpoint).

Usage
-----
    python scripts/run_experiment.py --smoke
    python scripts/run_experiment.py --configs all
    python scripts/run_experiment.py --configs a5_4corner_a50 \
        --policies hold fixed_band safe_greedy --seeds 0 1 2

Output
------
One JSONL file under --output-dir (default: runs/), one line per
(config, policy, seed). Results are reproducible under --seed (default 0).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Optional

import numpy as np

# Set up Python path so `certified_marl` imports work regardless of where
# the script is run from.
HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(REPO_ROOT))

from certified_marl.env.arrivals import make_arrival
from certified_marl.env.csgrag import CSGRAGEnv
from certified_marl.env.geometry import Obstacle, Rect
from certified_marl.metrics.panel import ExperimentPanel
from certified_marl.utils.registry import load_config


# ---------------------------------------------------------------------------
# Policy definitions
# ---------------------------------------------------------------------------


def _hold_policy(state, rng):
    """a_ij = 0 for every edge. Static-partition baseline."""
    return {edge: 0 for edge in state.interfaces.keys()}


# ---------------------------------------------------------------------------
# Paper Sec. 7.3 baselines: Fixed-Band + Slack-Gradient.
# These DO NOT need training -- they are deterministic policies operating
# through the same shield as CPAC, so comparison isolates the value of
# LEARNING the boundary adaptation over a HAND-DESIGNED one.
# ---------------------------------------------------------------------------


def make_fixed_band_policy(env, band_fraction: float = 0.5):
    """Fixed-band policy.

    Sets all interface bands delta_ij to band_fraction * delta*_ij ONCE at t=0 and
    holds forever. No adaptation. This isolates the value of having a
    non-zero initial buffer vs. a fully static partition (hold).

    Because we cannot rewrite the state from inside policy_fn (which only
    returns actions), we achieve the same effect by applying a_ij = -1
    (increase delta_ij) for the first few steps until every interface reaches
    the target, then holding.
    """
    target_fraction = float(band_fraction)
    initialized = {"done": False}

    def policy_fn(state, rng):
        actions = {edge: 0 for edge in state.interfaces.keys()}
        if initialized["done"]:
            return actions
        # Ramp up each interface's delta_ij toward target_fraction * delta*_ij
        # using a_ij = -1 (which INCREASES delta_ij per the directed-slack update).
        all_done = True
        for edge, ifs in state.interfaces.items():
            target = target_fraction * ifs.delta_star
            if ifs.band_ij.delta < target - env.delta_step * 0.5:
                # Need to expand i's side -> a_ij = -1 (per csgrag.py:606).
                actions[edge] = -1
                all_done = False
        if all_done:
            initialized["done"] = True
        return actions

    return policy_fn


def make_safe_greedy_policy(env, threshold: float = 0.01):
    """Slack-gradient boundary policy.

    At each epoch, for every ACTIVE interface (i, j) from the shield's
    matching, if |l_i - l_j| > threshold, push the boundary toward the
    heavier endpoint by one delta_step. Else hold. Action is filtered
    through the SAME A^safe mask CPAC uses, so safety is preserved.


    The default threshold is 0.01: a coarser 0.05 is too insensitive under
    small-gradient asymmetric loads, while 0.01 catches the slower-forming
    imbalances where SG should act to redistribute load.

    this policy is PURE adaptation. Any baseline buffer width
    is controlled at the ENVIRONMENT level via `env.initial_delta_fraction`
    (beta in [0, 1], setting delta_ij(0) = beta * delta*_ij). All policies (pi_hold, pi_FB,
    pi_SG, CPAC) inherit the same beta, so the comparison isolates per-policy
    ADAPTATION over a shared starting geometry.

    In the CANONICAL frame (canonical orientation):
      a_ij = +1 => "i expands into j" (delta_ij decreases, delta_ji increases)
      a_ij = -1 => "j expands into i" (delta_ij increases, delta_ji decreases)
    When i is heavier and should shed load, widen i's buffer (a_ij = -1)
    so more i-side events become buffer-eligible and can be dispatched to
    the lighter endpoint j via interface_dispatch.
    """
    from certified_marl.shield.feasibility_kernel import safe_action_set
    from certified_marl.shield.matching import active_interfaces

    # Accept either an env directly OR a zero-arg callable returning the
    # env, resolved at every call.
    if callable(env) and not hasattr(env, "agents"):
        _env_getter = env
    else:
        _env = env
        _env_getter = lambda: _env  # noqa: E731

    def policy_fn(state, rng):
        env = _env_getter()
        neighbors: dict[int, list[int]] = {i: [] for i in range(env.n)}
        for (i, j) in state.interfaces.keys():
            neighbors[i].append(j)
            neighbors[j].append(i)
        base_loads = [a.load_pressure_ewma for a in state.agents]
        active = active_interfaces(base_loads, neighbors)
        actions = {edge: 0 for edge in state.interfaces.keys()}
        for (i, j) in active:
            key = (min(i, j), max(i, j))
            ifs = state.interfaces[key]
            _t_now_re = float(state.t) * env.dt
            _obs_re = (env._current_obstacles(_t_now_re) if hasattr(env, '_current_obstacles') else env.obstacles)
            safe = safe_action_set(state, ifs, env.delta_step, _obs_re,
                                   speed=env.speed)
            diff = base_loads[i] - base_loads[j]
            if abs(diff) <= threshold:
                cand = 0
            elif diff > 0:
                cand = -1 if -1 in safe else 0   # widen i's buffer
            else:
                cand = +1 if +1 in safe else 0   # widen j's buffer
            actions[key] = cand
        return actions

    return policy_fn


def make_env_aware_masked_ppo_policy(bundle, env):
    """Given the checkpoint bundle + the live env, build a proper policy_fn.

    The env is needed because:
      - safe_action_set() requires env.delta_step and env.obstacles.
      - active_interfaces() requires the full neighbor graph.
    """
    import torch
    actor = bundle["actor"]
    actor_state_vec = bundle["actor_state_vec"]
    mask_from_safe_actions = bundle["mask_from_safe_actions"]
    orient_safe_actions_for_controller = bundle["orient_safe_actions_for_controller"]
    safe_action_set = bundle["safe_action_set"]
    active_interfaces = bundle["active_interfaces"]
    controller_of = bundle["controller_of"]
    use_extended_state = bool(bundle.get("use_extended_state", False))

    @torch.no_grad()
    def policy_fn(state, rng):
        # Default: every interface holds (a = 0).
        actions = {edge: 0 for edge in state.interfaces.keys()}

        # Matching step: select endpoint-disjoint active interfaces.
        neighbors = {i: [] for i in range(env.n)}
        for (i, j) in state.interfaces.keys():
            neighbors[i].append(j)
            neighbors[j].append(i)
        loads = [a.load_pressure_ewma for a in state.agents]
        active = active_interfaces(loads, neighbors)

        # For each active interface, sample masked action from the policy.
        for (i, j) in active:
            edge = (min(i, j), max(i, j))
            ifs = state.interfaces[edge]
            # Pass env.speed to match the env's unit system.
            _t_now_re = float(state.t) * env.dt
            _obs_re = (env._current_obstacles(_t_now_re) if hasattr(env, '_current_obstacles') else env.obstacles)
            safe = safe_action_set(state, ifs, env.delta_step, _obs_re,
                                    speed=env.speed)
            c = controller_of(i, j, loads)
            safe_controller = orient_safe_actions_for_controller(
                safe, controller=c, canonical_i=i,
            )
            mask = mask_from_safe_actions(safe_controller)
            s_vec = actor_state_vec(state, i, j,
                                    include_m_i=use_extended_state,
                                    speed=env.speed)
            s_t = torch.as_tensor(s_vec, dtype=torch.float32).unsqueeze(0)
            logits = actor(s_t).squeeze(0)
            mask_t = torch.as_tensor(mask, dtype=torch.float32)
            neg_inf = torch.finfo(logits.dtype).min
            masked_logits = torch.where(mask_t > 0, logits,
                                         torch.full_like(logits, neg_inf))
            # STOCHASTIC evaluation: sample from the masked softmax
            # The executed policy is the masked softmax, not argmax: greedy
            # eval would collapse to hold because the argmax can differ from
            # the sampled mode of operation under training.
            probs = torch.softmax(masked_logits, dim=-1)
            action_idx = int(torch.multinomial(probs, 1).item())
            action_signed = action_idx - 1  # {0,1,2} -> {-1,0,+1}
            # Controller-frame -> canonical-frame canonicalization.
            canonical_action = action_signed if c == i else -action_signed
            actions[edge] = canonical_action

        return actions

    return policy_fn


def _try_load_torch_policy(checkpoint_path: str):
    """Load a CPAC checkpoint into a bundle for
    make_env_aware_masked_ppo_policy."""
    import torch
    from certified_marl.models.flat_actor import FlatEdgeActor
    from certified_marl.shield.feasibility_kernel import safe_action_set
    from certified_marl.shield.matching import active_interfaces, controller_of
    from certified_marl.trainers.masked_ppo import (
        actor_state_vec, mask_from_safe_actions,
        orient_safe_actions_for_controller,
    )
    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu",
                          weights_only=False)
    except TypeError:
        ckpt = torch.load(checkpoint_path, map_location="cpu")
    mcfg = ckpt.get("model_cfg", {})
    tcfg = ckpt.get("trainer_cfg", {})
    use_extended_state = bool(tcfg.get("use_extended_state", False))
    state_dim = int(mcfg.get("state_dim", 7 if use_extended_state else 6))
    actor = FlatEdgeActor(state_dim=state_dim,
                          hidden=int(mcfg.get("hidden", 64)))
    actor.load_state_dict(ckpt["actor_state_dict"])
    actor.eval()
    bundle = {
        "actor": actor,
        "use_extended_state": use_extended_state,
        "actor_state_vec": actor_state_vec,
        "mask_from_safe_actions": mask_from_safe_actions,
        "orient_safe_actions_for_controller":
            orient_safe_actions_for_controller,
        "safe_action_set": safe_action_set,
        "active_interfaces": active_interfaces,
        "controller_of": controller_of,
    }
    return bundle, f"loaded masked_ppo from {checkpoint_path}"


def get_policy(name: str, checkpoint_path: Optional[str] = None):
    """Return (policy_fn_or_bundle, info_msg) for the named policy.

    For hold: returns a ready policy_fn(state, rng).
    For masked_ppo: returns a bundle (dict) that must be wrapped via
        make_env_aware_masked_ppo_policy(bundle, env)
    to produce a policy_fn. This is because masked_ppo requires access to
    env.delta_step and env.obstacles (for safe_action_set) and the full
    neighbor graph (for active_interfaces).
    """
    if name == "hold":
        return _hold_policy, "deterministic hold (all a_ij = 0)"
    if name == "masked_ppo":
        if checkpoint_path is None:
            return None, "masked_ppo requested but no --checkpoint given"
        return _try_load_torch_policy(checkpoint_path)
    if name == "fixed_band":
        # Marker; real policy built with env access in run_one.
        return "fixed_band", "Fixed-band policy"
    if name == "safe_greedy":
        return "safe_greedy", "Slack-gradient boundary policy"
    raise ValueError("unknown policy: " + name)


# ---------------------------------------------------------------------------
# Env construction from YAML config
# ---------------------------------------------------------------------------


def build_env_from_config(cfg: dict, seed: int, max_horizon: Optional[int] = None) -> CSGRAGEnv:
    """Construct a CSGRAGEnv directly from the config dictionary.

    We don't use the full `registry.build_env` path here because we want to
    be explicit about what's loaded (and avoid any torch imports). This
    mirrors the env-construction pattern used in the test suite.
    """
    env_cfg = cfg["env"]
    regions = [Rect(*r) for r in env_cfg["regions"]]
    kernels = [Rect(*k) for k in env_cfg["kernels"]]
    U_bar = list(env_cfg["U_bar"])
    bounds = tuple(env_cfg["bounds"])

    # Build the arrival process for this regime via make_arrival().
    arrivals = make_arrival(
        cfg["arrivals"],
        bounds=bounds,
        rng=np.random.default_rng(seed),
    )

    # Optional static obstacles.
    obstacles_cfg = env_cfg.get("obstacles", []) or []
    obstacles = [Obstacle(rect=Rect(*o)) for o in obstacles_cfg]

    horizon = int(env_cfg["horizon"])
    if max_horizon is not None:
        horizon = min(horizon, max_horizon)

    return CSGRAGEnv(
        regions=regions,
        kernels=kernels,
        U_bar=U_bar,
        arrivals=arrivals,
        obstacles=obstacles,
        bounds=bounds,
        dt=float(env_cfg.get("dt", 1.0)),
        speed=float(env_cfg.get("speed", 1.0)),
        delta_star_default=float(env_cfg.get("delta_star_default", 1.5)),
        delta_step=float(env_cfg.get("delta_step", 0.15)),
        rng=np.random.default_rng(seed + 1),
        horizon=horizon,
        # Head-start equalization. Default 0.0 sets delta_ij(0)=0.
        # Set in YAML: env.initial_delta_fraction: 0.5  for apples-to-apples.
        initial_delta_fraction=float(env_cfg.get("initial_delta_fraction", 0.0)),
        # Spatial-aware load: gamma_w >= 0 weights W_i/U_bar in the
        # reduced Markov load. Default 0.0 = pure load-pressure numerics.
        gamma_w_load=float(env_cfg.get("gamma_w_load", 0.0)),
        # Dynamic service-surface certification. Default False preserves
        # the static-owner certification path unless a config opts into
        # the executable eligibility-region model.
        dynamic_service_regions=bool(env_cfg.get("dynamic_service_regions", False)),
    )


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------


def run_one(
    config_name: str,
    policy_name: str,
    seed: int = 0,
    max_horizon: Optional[int] = None,
    checkpoint: Optional[str] = None,
    sg_threshold: Optional[float] = None,
    gamma_w: Optional[float] = None,
    horizon_override: Optional[int] = None,
) -> dict:
    """Run one (config, policy, seed) triple; return the serialized
    results dict ready for JSONL writing.
    """
    cfg_path = REPO_ROOT / "configs" / "experiment" / (config_name + ".yaml")
    cfg = load_config(str(cfg_path))
    # CLI override for the spatial-aware load weight.
    if gamma_w is not None:
        cfg.setdefault("env", {})["gamma_w_load"] = float(gamma_w)
    # Horizon-sweep override: replace env.horizon entirely.
    if horizon_override is not None:
        cfg.setdefault("env", {})["horizon"] = int(horizon_override)
    env = build_env_from_config(cfg, seed=seed, max_horizon=max_horizon)

    policy_obj, policy_msg = get_policy(policy_name, checkpoint_path=checkpoint)
    if policy_obj is None:
        return dict(
            config=config_name,
            policy=policy_name,
            seed=seed,
            status="SKIPPED",
            reason=policy_msg,
        )

    # Env-aware policies (need env for shield / matching / band targets).
    if policy_name == "masked_ppo":
        policy_fn = make_env_aware_masked_ppo_policy(policy_obj, env)
    elif policy_name == "fixed_band":
        policy_fn = make_fixed_band_policy(env, band_fraction=0.5)
    elif policy_name == "safe_greedy":
        thr = sg_threshold if sg_threshold is not None else 0.01
        policy_fn = make_safe_greedy_policy(env, threshold=thr)
    else:
        policy_fn = policy_obj

    rng = np.random.default_rng(seed + 100)
    panel = ExperimentPanel()
    panel.start()
    state = env.reset(seed=seed)
    for _ in range(env.horizon):
        actions = policy_fn(state, rng)
        state, info = env.step(actions)
        panel.record(state, info)
        if info["done"]:
            break
    results = panel.finalize(state)

    # Per-region response and rejection summaries used by reported tables.
    per_region_stats = {}
    rt_by_agent = getattr(state, "response_times_by_agent", None)
    rej_by_agent = getattr(state, "rejected_by_agent", None)
    if rt_by_agent or rej_by_agent:
        import statistics as _stats
        per_region_max = {i: (max(v) if v else 0.0)
                          for i, v in (rt_by_agent or {}).items()}
        per_region_count = {i: len(v) for i, v in (rt_by_agent or {}).items()}
        all_rts = list(state.response_times)

        rej_vals = list((rej_by_agent or {}).values())
        total_rejected = sum(rej_vals)

        per_region_stats = {
            "per_region_max": per_region_max,
            "per_region_count": per_region_count,
            "global_max_T": max(all_rts) if all_rts else 0.0,
            "mean_of_per_region_max": (
                _stats.mean(per_region_max.values())
                if per_region_max else 0.0
            ),
            "rejection_count": total_rejected,
            "per_region_rejection": dict(rej_by_agent or {}),
        }

    # Stamp git hash for reproducibility.
    git_hash = _current_git_hash()
    return dict(
        config=config_name,
        policy=policy_name,
        seed=seed,
        status="OK",
        policy_info=policy_msg,
        horizon=env.horizon,
        git_hash=git_hash,
        results=results.as_dict(),
        per_region_stats=per_region_stats,
    )


def _current_git_hash() -> str:
    """Return short git hash of the current HEAD, or '(none)' if unavailable."""
    import subprocess
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(REPO_ROOT), stderr=subprocess.DEVNULL,
        ).decode().strip()
        return out if out else "(none)"
    except Exception:
        return "(none)"


def print_summary(records: list[dict]) -> None:
    """Print a compact ASCII table of key metrics across runs."""
    ok = [r for r in records if r["status"] == "OK"]
    skipped = [r for r in records if r["status"] != "OK"]

    print()
    print("=" * 100)
    print("EXPERIMENT SUMMARY")
    print("=" * 100)
    if not ok:
        print("No successful runs.")
    else:
        header = "{:24s} {:12s} {:>5s} {:>8s} {:>8s} {:>8s} {:>9s} {:>9s} {:>9s} {:>8s}".format(
            "CONFIG", "POLICY", "SEED", "VIOLS", "ADMIT", "REJECT", "MEAN", "P95", "MAX", "LOADCV"
        )
        print(header)
        print("-" * len(header))
        for r in ok:
            res = r["results"]
            tail = res["tail"] or {}
            print("{:24s} {:12s} {:5d} {:8d} {:8d} {:8d} {:9.3f} {:9.3f} {:9.3f} {:8.3f}".format(
                r["config"][:24],
                r["policy"][:12],
                r["seed"],
                int(res["safety"]["total"]),
                int(res["throughput"]["admitted"]),
                int(res["throughput"]["rejected"]),
                float(tail.get("mean", float("nan"))) if tail else float("nan"),
                float(tail.get("p95", float("nan"))) if tail else float("nan"),
                float(tail.get("max", float("nan"))) if tail else float("nan"),
                float(res["imbalance"].get("mean_coef_var", float("nan"))),
            ))
    if skipped:
        print("\nSkipped runs:")
        for r in skipped:
            print("  {:24s} {:12s}  -> {}".format(r["config"], r["policy"], r["reason"]))
    print("=" * 100)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


# The six reported arrival regimes (U, S1, SB, N, A, R). The scaling and
# sweep configs are run explicitly via --configs (see REPRODUCE.md).
ALL_CONFIGS = [
    "exp00_g4_c8",                      # U  (uniform)
    "f4i_g4_c16_beta0_true_baselines",  # S1 (single hotspot)
    "a5_4corner_a50",                   # SB (corner hotspots)
    "a4_shifting_slow",                 # N  (shifting hotspot, period 1000)
    "adv_b_boundary_stress",            # A  (adaptive stress)
    "chicago_real_replay",              # R  (Chicago real trace)
]
ALL_POLICIES = ["hold", "fixed_band", "safe_greedy", "masked_ppo"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=None,
                    help="Configs to run. Default: all. Use 'all' for all.")
    ap.add_argument("--policies", nargs="+",
                    default=["hold", "fixed_band", "safe_greedy"],
                    help="Policies to evaluate.")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0],
                    help="Seeds to run (one line per seed). Default: [0].")
    ap.add_argument("--checkpoint", default=None,
                    help="Path to trained PPO checkpoint (for masked_ppo policy).")
    ap.add_argument("--max-horizon", type=int, default=None,
                    help="Cap horizon for each rollout (useful for quick iteration).")
    ap.add_argument("--output-dir", default="runs",
                    help="Directory to write JSONL results. Default: runs/")
    ap.add_argument("--smoke", action="store_true",
                    help="Smoke mode: SB regime + hold/random, 200 epochs.")
    ap.add_argument("--sg-threshold", type=float, default=None,
                    help="Override safe_greedy threshold (default 0.01). "
                         "Used for the threshold sensitivity sweep.")
    ap.add_argument("--gamma-w", type=float, default=None,
                    help="Override env gamma_w_load (spatial term in "
                         "load pressure; default 0.0).")
    ap.add_argument("--horizon-override", type=int, default=None,
                    help="Replace env.horizon with this value (for the "
                         "rollout-window sweep). Unlike --max-horizon which "
                         "caps, this fully replaces.")
    args = ap.parse_args()

    if args.smoke:
        configs = ["a5_4corner_a50"]
        policies = ["hold", "fixed_band", "safe_greedy"]
        max_horizon = args.max_horizon or 200
    else:
        configs = args.configs or ALL_CONFIGS
        if len(configs) == 1 and configs[0] == "all":
            configs = ALL_CONFIGS
        policies = args.policies
        max_horizon = args.max_horizon

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_path = output_dir / ("run_" + timestamp + ".jsonl")

    records: list[dict] = []
    print("Running {} configs x {} policies x {} seeds = {} runs".format(
        len(configs), len(policies), len(args.seeds), len(configs) * len(policies) * len(args.seeds)
    ))
    print("Output: " + str(output_path))
    print("-" * 70)

    with open(output_path, "w") as f:
        for cfg in configs:
            for pol in policies:
                for seed in args.seeds:
                    label = "[{}/{}/seed={}]".format(cfg, pol, seed)
                    try:
                        print(label + " running...", end=" ", flush=True)
                        record = run_one(
                            config_name=cfg, policy_name=pol, seed=seed,
                            max_horizon=max_horizon, checkpoint=args.checkpoint,
                            sg_threshold=args.sg_threshold,
                            gamma_w=args.gamma_w,
                            horizon_override=args.horizon_override,
                        )
                        records.append(record)
                        f.write(json.dumps(record) + "\n")
                        f.flush()
                        if record["status"] == "OK":
                            v = int(record["results"]["safety"]["total"])
                            cv = float(record["results"]["imbalance"].get("mean_coef_var", float("nan")))
                            print("OK (viols={}, load_cv={:.3f})".format(v, cv))
                        else:
                            print("SKIPPED: " + record["reason"])
                    except Exception as e:
                        err = dict(
                            config=cfg, policy=pol, seed=seed,
                            status="ERROR",
                            reason=str(e),
                            traceback=traceback.format_exc(),
                        )
                        records.append(err)
                        f.write(json.dumps(err) + "\n")
                        f.flush()
                        print("ERROR: " + str(e))

    print_summary(records)
    print()
    print("JSONL written to: " + str(output_path))
    print("  -> Paste the file contents (or its last N lines) back to share results.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
