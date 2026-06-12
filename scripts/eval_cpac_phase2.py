r"""Phase 2: CPAC evaluation - load checkpoint, run held-out rollouts.

Usage
-----
    python scripts/eval_cpac_phase2.py \\
        --checkpoint runs/p2_cpac/S1/seed0/ckpt.pt \\
        --config f4i_g4_c16_beta0_true_baselines \\
        --seeds 0 1 2 3 4 5 6 7 8 9 \\
        --out runs/p2_cpac/S1/eval

Output
------
    <out>/run_seed<i>_<timestamp>.jsonl  one line per seed.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(REPO_ROOT))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True,
                   help="path to ckpt.pt from train_cpac_phase2.py")
    p.add_argument("--config", required=True,
                   help="config base name under configs/experiment/")
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(10)),
                   help="evaluation seeds (matched-seed protocol)")
    p.add_argument("--out", required=True,
                   help="output directory for evaluation rollout JSONLs")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--policy-name", default="cpac_full",
                   help="label written into rollout JSONL (e.g. cpac_full, "
                        "cpac_no_latent_feature, etc.)")
    p.add_argument("--deterministic", action="store_true",
                   help="argmax actions instead of sampling (eval-time "
                        "default: stochastic to match training distribution)")
    args = p.parse_args()

    import torch
    from certified_marl.models.flat_actor import FlatEdgeActor
    from certified_marl.shield.feasibility_kernel import safe_action_set
    from certified_marl.shield.matching import active_interfaces, controller_of
    from certified_marl.trainers.masked_ppo import (
        actor_state_vec, mask_from_safe_actions,
        orient_safe_actions_for_controller,
    )
    from scripts.run_experiment import (
        build_env_from_config, load_config, _current_git_hash,
    )

    cfg_path = REPO_ROOT / "configs" / "experiment" / (args.config + ".yaml")
    if not cfg_path.exists():
        print(f"ERROR: config not found: {cfg_path}", file=sys.stderr)
        return 2
    cfg = load_config(str(cfg_path))

    # ---- Load checkpoint and detect format ----
    try:
        ckpt = torch.load(args.checkpoint, map_location=args.device,
                          weights_only=False)
    except TypeError:
        ckpt = torch.load(args.checkpoint, map_location=args.device)
    mcfg = ckpt.get("model_cfg", {})
    tcfg = ckpt.get("trainer_cfg", {})
    device = torch.device(args.device)

    # Detect whether this checkpoint uses the 7-D extended state.
    use_extended_state = bool(tcfg.get("use_extended_state", False))
    state_dim = int(mcfg.get("state_dim", 7 if use_extended_state else 6))
    hidden = int(mcfg.get("hidden", 64))
    actor = FlatEdgeActor(state_dim=state_dim, hidden=hidden).to(device).eval()
    actor.load_state_dict(ckpt["actor_state_dict"])

    print(f"[eval] checkpoint = {args.checkpoint}")
    print(f"[eval]   trainer = {ckpt.get('experiment_cfg', {}).get('trainer', '?')}")
    print(f"[eval]   state_dim = {state_dim} ({'7-D w/m_c' if use_extended_state else '6-D'})")
    print(f"[eval]   augmented_reward = "
          f"{ckpt.get('experiment_cfg', {}).get('augmented_reward', False)}")
    print(f"[eval]   policy-name = {args.policy_name}  config = {args.config}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    git_hash = _current_git_hash()

    for seed in args.seeds:
        env = build_env_from_config(cfg, seed=seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        @torch.no_grad()
        def flat_policy_fn(state):
            """Masked-softmax actions for all active interfaces, using the
            actor on the controller-centered state."""
            actions = {edge: 0 for edge in state.interfaces.keys()}
            neighbors = {i: [] for i in range(env.n)}
            for (i, j) in state.interfaces.keys():
                neighbors[i].append(j)
                neighbors[j].append(i)
            loads = [a.load_pressure_ewma for a in state.agents]
            active = active_interfaces(loads, neighbors)
            if not active:
                return actions
            for (i, j) in active:
                edge = (min(i, j), max(i, j))
                ifs = state.interfaces[edge]
                c = controller_of(i, j, loads)
                # Match the env's apply-time obstacle set under
                # dynamic_service_regions.
                _t_now_eval = float(state.t) * env.dt
                _active_obs_eval = (
                    env._current_obstacles(_t_now_eval)
                    if hasattr(env, "_current_obstacles")
                    else env.obstacles
                )
                safe = safe_action_set(state, ifs, env.delta_step,
                                       _active_obs_eval, speed=env.speed)
                safe_controller = orient_safe_actions_for_controller(
                    safe, controller=c, canonical_i=i,
                )
                mask = mask_from_safe_actions(safe_controller)

                s_actor_np = actor_state_vec(
                    state, i, j,
                    include_m_i=use_extended_state,
                    speed=env.speed,
                )
                s_actor = torch.as_tensor(s_actor_np, dtype=torch.float32,
                                          device=device).unsqueeze(0)
                logits = actor(s_actor)
                mask_t = torch.as_tensor(mask, dtype=torch.float32,
                                         device=device).unsqueeze(0)
                neg_inf = torch.finfo(logits.dtype).min
                masked_logits = torch.where(
                    mask_t > 0, logits, torch.full_like(logits, neg_inf),
                )
                probs = torch.softmax(masked_logits, dim=-1).squeeze(0)
                # Defensive (mirror trainer collector fix): if numerical
                # issues yield NaN / negative / all-zero probs, fall back
                # to uniform-over-safe. The shield guarantees hold is in
                # A^safe (Section 3.4), so the fallback always has an action.
                if (not torch.isfinite(probs).all()) or \
                        float(probs.sum().item()) < 1e-9:
                    safe_mask_vec = mask_t.squeeze(0)
                    if float(safe_mask_vec.sum().item()) > 0:
                        probs = safe_mask_vec / safe_mask_vec.sum()
                    else:
                        probs = torch.zeros_like(probs)
                        probs[1] = 1.0  # force hold
                if args.deterministic:
                    action_idx = int(torch.argmax(probs).item())
                else:
                    action_idx = int(torch.multinomial(probs, 1).item())
                action_signed = action_idx - 1
                # (i, j) orientation. See trainers/masked_ppo.py docstring.
                canonical = action_signed if c == i else -action_signed
                actions[edge] = canonical
            return actions

        # Rollout assembly: mirror run_one() metric block from
        # scripts/run_experiment.py so the JSONL is shape-compatible
        # with phase1_master_aggregate.py.
        record = _run_one_eval_rollout(
            env=env, policy_fn=flat_policy_fn,
            config_name=args.config, policy_name=args.policy_name,
            seed=seed,
            policy_msg=(f"CPAC flat (ckpt={args.checkpoint}; "
                        f"augmented_reward="
                        f"{ckpt.get('experiment_cfg', {}).get('augmented_reward', False)})"),
            git_hash=git_hash,
        )
        ts = time.strftime("%Y%m%d-%H%M%S")

        fname = f"run_seed{seed}_{ts}.jsonl"
        path = out_dir / fname
        with open(path, "w") as f:
            f.write(json.dumps(record) + "\n")
        print(f"  seed {seed} -> {path}")

    return 0


def _run_one_eval_rollout(env, policy_fn, config_name: str,
                          policy_name: str, seed: int, policy_msg: str,
                          git_hash: str) -> dict:
    """Run one CPAC evaluation rollout and assemble the release JSONL record."""
    import statistics as stats
    from certified_marl.metrics.panel import ExperimentPanel

    state = env.reset()

    panel = ExperimentPanel()

    while True:
        actions = policy_fn(state)
        state, info = env.step(actions)
        panel.record(state, info)
        if info["done"]:
            break

    results = panel.finalize(state)
    rt_by_agent = getattr(state, "response_times_by_agent", None)
    rej_by_agent = getattr(state, "rejected_by_agent", None)

    per_region_stats: dict = {}
    if rt_by_agent or rej_by_agent:
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
                stats.mean(per_region_max.values())
                if per_region_max else 0.0
            ),
            "rejection_count": total_rejected,
            "per_region_rejection": dict(rej_by_agent or {}),
        }

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


if __name__ == "__main__":
    sys.exit(main())
