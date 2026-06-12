r"""CPAC training: masked pairwise actor-critic trained with masked PPO.

Trains the CPAC policy: a masked actor and centralized critic optimized
inside the same feasibility kernel as the deterministic policies. The release
path follows the reported CPAC settings: 7-D controller-centered actor
state, supervised imitation of pi_SG, annealed CE anchor, entropy regularization,
reward normalization, and the base plus auxiliary CPAC reward.

The shield and CS-LSTF provide certification. CPAC only selects among actions
that pass the feasibility mask. Optional research flags below are disabled
unless supplied explicitly; defaults reproduce the reported setting.

Usage
-----
    python scripts/train_cpac_phase2.py \
        --configs <regime_config> \
        --seed 0 \
        --num-updates 100 \
        --rollout-length 256 \
        --out runs/<runroot>/seed0 \
        --device cpu

Output
------
    <out>/ckpt.pt              checkpoint (actor + critic + model_cfg
                                + trainer_cfg + shaping_cfg + experiment_cfg)
    <out>/training_log.jsonl   per-update training diagnostics (one JSON
                                line per PPO update, regime-tagged)
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
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--configs", nargs="+", required=True,
                   help="one or more config base names under "
                        "configs/experiment/ (e.g. 'f4i_g4_c16_beta0_true_baselines')")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--env-seed", type=int, default=None,
                   help="Env stochastic seed (arrivals, etc.). If None, "
                        "uses --seed. Pass a constant (e.g. 0) to decouple "
                        "the env arrival realization from the trainer RNG.")
    p.add_argument("--init-seed", type=int, default=None,
                   help="Actor/critic weight-initialization seed. "
                        "If None, uses --seed. Pass a constant (e.g. 0) to "
                        "fix model-init RNG across trainer seeds so seed "
                        "variation reflects exploration RNG only.")
    p.add_argument("--num-updates", type=int, default=100)
    p.add_argument("--rollout-length", type=int, default=256)
    p.add_argument("--epochs-per-update", type=int, default=4)
    p.add_argument("--minibatch-size", type=int, default=64)
    p.add_argument("--lr-actor", type=float, default=3e-4)
    p.add_argument("--lr-critic", type=float, default=1e-3)
    p.add_argument("--entropy-coef", type=float, default=0.03)
    p.add_argument("--admission-reward-coef", type=float, default=0.5)
    p.add_argument("--epsilon-start", type=float, default=0.30)
    p.add_argument("--epsilon-final", type=float, default=0.05)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--omega-b", type=float, default=0.5,
                   help="Psi weight on endpoint latent-overload b_bar; "
                        "use 0.0 for the no-latent-feature ablation")
    p.add_argument("--eta", type=float, default=0.01,
                   help="action-oscillation (motion) cost eta in Eq. 26. "
                        "(reported default 0.01). Increase to 0.05 if the "
                        "policy converges to always-move (frac_hold < 0.05).")
    p.add_argument("--beta", type=float, default=1.0,
                   help="Psi weight on pairwise load imbalance |l_i - l_j| "
                        "(CPAC base reward imbalance term). Default 1.0. Sensitivity "
                        "sweep uses {0.1, 0.25, 0.5, 1.0, 2.0}.")
    # Rejection-aware auxiliary reward terms.
    p.add_argument("--aux-omega-zeta", type=float, default=0.5,
                   help="(c) max-rejection-rate weight: -omega_zeta * "
                        "max{zeta_i, zeta_j}(t+1). zeta_i = R_i/(G_i+eps).")
    p.add_argument("--aux-omega-delta", type=float, default=0.2,
                   help="(d) rejection-imbalance weight: -omega_Delta * "
                        "|zeta_i - zeta_j|(t+1).")
    p.add_argument("--aux-omega-c", type=float, default=0.1,
                   help="(e) true-cross-service weight: +omega_c * "
                        "C_ij^true-cross(t+1). Distinct from same-owner buffer.")
    p.add_argument("--aux-omega-chi", type=float, default=0.0,
                   help="(f) capacity-overflow weight: -omega_chi * "
                        "max(0, chi-1) where chi_i = lambda_i/lambda_i^cert.")
    p.add_argument("--aux-chi-max", type=float, default=3.0,
                   help="Clip chi at chi_max for stable training (default 3.0).")
    p.add_argument("--aux-edge-local-zeta", action="store_true",
                   help="Use edge-local rejection rate zeta_{i|ij} instead of "
                        "per-agent zeta_i in terms (c) and (d). Tightens credit "
                        "assignment on multi-active-interface regimes. "
                        "Default off.")
    # Reward normalizer; on by default.
    p.add_argument("--no-reward-normalize", dest="reward_normalize",
                   action="store_false", default=True,
                   help="Disable per-step reward normalization (on by "
                        "default reported setting).")
    p.add_argument("--reward-norm-alpha", type=float, default=0.99,
                   help="EMA decay for running mean/std (default 0.99).")
    p.add_argument("--reward-norm-clip-std", type=float, default=5.0,
                   help="Clip normalized rewards at +- clip_std (default 5).")
    p.add_argument("--reward-norm-warmup-steps", type=int, default=100,
                   help="Pass rewards through unchanged for first N samples "
                        "while normalizer warms up.")
    # Periodic checkpointing.
    p.add_argument("--save-every", type=int, default=0,
                   help="If > 0, save an intermediate ckpt_u{N}.pt every N "
                        "PPO updates (in addition to final ckpt.pt). Default "
                        "0 = no intermediate ckpts.")
    # n-step commit fix (on by default reported setting): drop the n+1 psi_hist
    # cap and flush eligible pending transitions before episode reset.
    p.add_argument("--no-nstep-fix", dest="nstep_fix",
                   action="store_false", default=True,
                   help="Disable the n-step commit fix (on by default).")
    # BC pre-training + KL-regularized PPO with pi_SG as prior.
    p.add_argument("--bc-pretrain-steps", type=int, default=500,
                   help="number of pi_SG rollout steps for behavioral "
                        "cloning pre-training (0 disables).")
    p.add_argument("--bc-pretrain-epochs", type=int, default=2)
    p.add_argument("--bc-pretrain-lr", type=float, default=1e-3)
    p.add_argument("--bc-label-smoothing", type=float, default=0.30,
                   help="smooth BC labels over A_safe; 0.0 = hard BC.")
    p.add_argument("--entropy-floor-coef", type=float, default=0.05,
                   help="soft penalty coefficient for staying below an "
                        "entropy floor over the feasible support.")
    p.add_argument("--entropy-floor-frac", type=float, default=0.60,
                   help="target entropy as a fraction of log(|A_safe|); "
                        "ignored when entropy-floor-coef is 0.")
    p.add_argument("--kl-prior-coef-start", type=float, default=0.05,
                   help="CE-anchor-to-pi_SG weight at update 0 "
                        "(annealed to --kl-prior-coef-final).")
    p.add_argument("--kl-prior-coef-final", type=float, default=0.0,
                   help="CE-anchor weight at last update; linearly annealed. "
                        "Default 0.0 releases the actor after anchoring.")
    p.add_argument("--sg-threshold", type=float, default=0.01,
                   help="threshold for the pi_SG prior policy (paper "
                        "default 0.01)")
    # n-step shaping.
    p.add_argument("--n-step", type=int, default=4,
                   help="n-step horizon for progressive Psi shaping (n=1 "
                        "recovers single-step reward exactly)")
    # Extended 7-D state.
    p.add_argument("--no-extended-state", action="store_true",
                   help="drop m_c from the state (6-D state)")
    # I/O.
    p.add_argument("--out", required=True,
                   help="output directory for checkpoint + training log")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = p.parse_args()

    use_extended_state = not args.no_extended_state

    # Lazy imports so --help works without torch.
    import torch
    from certified_marl.models.flat_actor import FlatEdgeActor, FlatCritic
    from certified_marl.objectives.shaping import ShapingWeights
    from certified_marl.trainers.masked_ppo import MaskedPPOConfig, MaskedPPOTrainer
    from scripts.run_experiment import (
        build_env_from_config, load_config, make_safe_greedy_policy,
    )

    # ---- Load configs and build envs ----
    cfg_pairs: list[tuple[str, dict]] = []
    envs = []
    for cfg_name in args.configs:
        cfg_path = REPO_ROOT / "configs" / "experiment" / (cfg_name + ".yaml")
        if not cfg_path.exists():
            print(f"ERROR: config not found: {cfg_path}", file=sys.stderr)
            return 2
        cfg = load_config(str(cfg_path))
        cfg_pairs.append((cfg_name, cfg))
        env_seed = args.env_seed if args.env_seed is not None else args.seed
        envs.append(build_env_from_config(cfg, seed=env_seed))
    if len(envs) != 1:
        print("ERROR: pass exactly one config (this trainer handles one regime "
              "per run).", file=sys.stderr)
        return 2

    # ---- Reproducibility ----
    # Two-seed protocol: actor/critic weights are initialized with
    # `init_seed` (defaults to args.seed). If --init-seed is passed, it
    # overrides the model-init RNG only; the PPO/exploration
    # RNGs still derive from args.seed. This decouples model-initialization
    # effects from the stochastic-policy/exploration variation across runs.
    init_seed = args.init_seed if args.init_seed is not None else args.seed
    np.random.seed(args.seed)
    torch.manual_seed(init_seed)  # used for upcoming actor/critic init

    # ---- Models ----
    # Actor state: 6-D base or 7-D with the m_c margin (the paper's
    # Eq. 20 state); critic adds 3 augmentation scalars.
    state_dim = 7 if use_extended_state else 6
    critic_dim = 10 if use_extended_state else 9
    device = torch.device(args.device)
    actor = FlatEdgeActor(state_dim=state_dim, hidden=args.hidden)
    critic = FlatCritic(input_dim=critic_dim, hidden=args.hidden)
    # Two-seed protocol: now that model weights are initialized using
    # init_seed's RNG state, re-seed torch with args.seed so subsequent
    # stochastic operations (PPO action sampling, exploration noise, etc.)
    # vary across trainer seeds as intended. No-op when init_seed == args.seed.
    if init_seed != args.seed:
        torch.manual_seed(args.seed)

    # ---- Shaping ----
    # CPAC uses omega_b for latent-overload pressure; rho is reserved
    # for response-budget tolerance.
    sw = ShapingWeights(
        beta=args.beta,             # imbalance weight
        omega_b=args.omega_b,       # latent-overload weight
        kappa=0.1,
        eta=args.eta,
    )

    # ---- Trainer config ----
    tcfg = MaskedPPOConfig(
        lr_actor=args.lr_actor,
        lr_critic=args.lr_critic,
        entropy_coef=args.entropy_coef,
        epochs_per_update=args.epochs_per_update,
        minibatch_size=args.minibatch_size,
        rollout_length=args.rollout_length,
        admission_reward_coef=args.admission_reward_coef,
        epsilon_start=args.epsilon_start,
        epsilon_final=args.epsilon_final,
        # CPAC auxiliary reward terms; chi is an optional extension.
        aux_omega_zeta=args.aux_omega_zeta,
        aux_omega_delta=args.aux_omega_delta,
        aux_omega_c=args.aux_omega_c,
        aux_omega_chi=args.aux_omega_chi,
        aux_chi_max=args.aux_chi_max,
        aux_edge_local_zeta=bool(args.aux_edge_local_zeta),
        # Reward normalizer.
        reward_normalize=bool(args.reward_normalize),
        reward_norm_alpha=args.reward_norm_alpha,
        reward_norm_clip_std=args.reward_norm_clip_std,
        reward_norm_warmup_steps=args.reward_norm_warmup_steps,
        # n-step commit fix.
        nstep_fix=bool(args.nstep_fix),
        # n-step shaping.
        n_step_horizon=args.n_step,
        use_extended_state=use_extended_state,
        # BC pre-training + KL anchor.
        bc_pretrain_steps=args.bc_pretrain_steps,
        bc_pretrain_epochs=args.bc_pretrain_epochs,
        bc_pretrain_lr=args.bc_pretrain_lr,
        bc_label_smoothing=args.bc_label_smoothing,
        entropy_floor_coef=args.entropy_floor_coef,
        entropy_floor_frac=args.entropy_floor_frac,
        kl_prior_coef_start=args.kl_prior_coef_start,
        kl_prior_coef_final=args.kl_prior_coef_final,
    )

    # Build the pi_SG prior policy on the training env.
    env = envs[0]

    use_prior = (args.bc_pretrain_steps > 0
                 or args.kl_prior_coef_start > 0
                 or args.kl_prior_coef_final > 0)
    if not use_prior:
        prior_action_fn = None
    else:
        prior_action_fn = make_safe_greedy_policy(
            env, threshold=args.sg_threshold,
        )

    # ---- Build the trainer ----
    trainer = MaskedPPOTrainer(
        env, actor, critic, tcfg, device=device,
        prior_action_fn=prior_action_fn,
    )
    trainer.collector.shaping_w = sw

    # Behavioral-cloning pre-training before the PPO loop.
    if prior_action_fn is not None and args.bc_pretrain_steps > 0:
        bc_diag = trainer.pretrain_bc(
            n_steps=args.bc_pretrain_steps,
            n_epochs=args.bc_pretrain_epochs,
            lr=args.bc_pretrain_lr,
        )
        print(f"[bc] pi_SG prior: {bc_diag}")

    # ---- Training loop ----
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "training_log.jsonl"
    ckpt_path = out_dir / "ckpt.pt"

    print(f"[train] configs={[n for n, _ in cfg_pairs]}  seed={args.seed}  "
          f"updates={args.num_updates}  rollout={args.rollout_length}  "
          f"ext_state={use_extended_state}  n_step={args.n_step}")
    print(f"[train] base reward: omega_l={sw.beta}, omega_b={sw.omega_b}, "
          f"kappa={sw.kappa}, eta={sw.eta}; aux alpha_c={tcfg.admission_reward_coef}, "
          f"omega_zeta={tcfg.aux_omega_zeta}, omega_delta={tcfg.aux_omega_delta}, "
          f"omega_c={tcfg.aux_omega_c}")

    t0 = time.time()
    log_handle = open(log_path, "w")
    try:
        for u in range(args.num_updates):
            # Linear epsilon anneal over the full budget.
            if args.num_updates > 1:
                frac = u / (args.num_updates - 1)
            else:
                frac = 1.0
            eps_now = (
                tcfg.epsilon_start
                + (tcfg.epsilon_final - tcfg.epsilon_start) * frac
            )
            trainer.collector.epsilon = eps_now
            # Anneal KL-prior coefficient.
            kl_now = (tcfg.kl_prior_coef_start
                      + (tcfg.kl_prior_coef_final
                         - tcfg.kl_prior_coef_start) * frac)
            trainer._kl_prior_coef = kl_now

            regime_name = cfg_pairs[0][0]

            buf = trainer.collect()

            # Per-rollout rejection rate uses the rollout-level
            # admitted/rejected aggregates that survive mid-rollout
            # env.reset() (Chicago configs have horizon < rollout_length, so
            # resets fire every rollout). The collector exposes these as
            # `_rollout_admitted`/`_rollout_rejected` counters, reset at the
            # top of each collect() call, matching eval semantics:
            #   rejection_rate = rejected / (admitted + rejected)
            roll_rej = int(getattr(trainer.collector, "_rollout_rejected", 0))
            roll_adm = int(getattr(trainer.collector, "_rollout_admitted", 0))
            roll_done = int(getattr(trainer.collector, "_rollout_completed", 0))
            roll_resets = int(getattr(trainer.collector, "_rollout_resets", 0))
            denom = max(1, roll_rej + roll_adm)
            rejection_rate = roll_rej / denom

            # Update.
            diag = trainer.update(buf)
            diag.update(dict(
                update=u,
                epsilon=eps_now,
                kl_prior_coef=kl_now,
                regime=regime_name,
                buffer_size=len(buf),
                rejection_rate=rejection_rate,
                rollout_admitted=roll_adm,
                rollout_rejected=roll_rej,
                rollout_completed=roll_done,
                rollout_resets=roll_resets,
                wall_seconds=round(time.time() - t0, 2),
            ))
            log_handle.write(json.dumps(diag) + "\n")
            log_handle.flush()
            # Periodic checkpoint save. Saves a partial-training ckpt for
            if (args.save_every > 0 and (u + 1) % args.save_every == 0
                    and u != args.num_updates - 1):
                snap_path = out_dir / f"ckpt_u{u+1}.pt"
                _snap_dict = {
                    "actor_state_dict": actor.state_dict(),
                    "critic_state_dict": critic.state_dict(),
                    "model_cfg": dict(
                        state_dim=state_dim, critic_dim=critic_dim,
                        hidden=args.hidden,
                    ),
                    "trainer_cfg": dict(
                        use_extended_state=tcfg.use_extended_state,
                    ),
                    "snapshot_update": int(u + 1),
                }
                torch.save(_snap_dict, snap_path)
            if (u + 1) % 5 == 0 or u == args.num_updates - 1:
                print(f"  update {u+1}/{args.num_updates} regime={regime_name:<32}  "
                      f"buf={len(buf):3d}  loss_pi={diag.get('loss_pi', 0):+.3f}  "
                      f"loss_v={diag.get('loss_v', 0):.2f}  "
                      f"H={diag.get('entropy', 0):.3f}  "
                      f"Hn={diag.get('entropy_norm', 0):.3f}  "
                      f"|A|={diag.get('mask_card_mean', 0):.2f}  "
                      f"rejection={rejection_rate:.3f}  "
                      f"eps={eps_now:.3f}")
    finally:
        log_handle.close()

    # ---- Save checkpoint ----
    save_dict = {
        "actor_state_dict": actor.state_dict(),
        "critic_state_dict": critic.state_dict(),
        "model_cfg": dict(
            state_dim=state_dim, critic_dim=critic_dim, hidden=args.hidden,
        ),
        "experiment_cfg": dict(
            configs=[n for n, _ in cfg_pairs],
            seed=args.seed,
            env_seed=env_seed,    # may differ from args.seed if --env-seed used
            init_seed=init_seed,  # may differ from args.seed if --init-seed used
            num_updates=args.num_updates,
            trainer="masked_ppo",
            augmented_reward=True,
        ),
        "trainer_cfg": dict(
            lr_actor=tcfg.lr_actor, lr_critic=tcfg.lr_critic,
            entropy_coef=tcfg.entropy_coef,
            num_updates=args.num_updates,
            rollout_length=tcfg.rollout_length,
            admission_reward_coef=tcfg.admission_reward_coef,
            epsilon_start=tcfg.epsilon_start, epsilon_final=tcfg.epsilon_final,
            n_step_horizon=tcfg.n_step_horizon,
            use_extended_state=tcfg.use_extended_state,
            # BC pre-training + KL anchor.
            bc_pretrain_steps=tcfg.bc_pretrain_steps,
            bc_pretrain_epochs=tcfg.bc_pretrain_epochs,
            bc_label_smoothing=tcfg.bc_label_smoothing,
            entropy_floor_coef=tcfg.entropy_floor_coef,
            entropy_floor_frac=tcfg.entropy_floor_frac,
            kl_prior_coef_start=tcfg.kl_prior_coef_start,
            kl_prior_coef_final=tcfg.kl_prior_coef_final,
            sg_threshold=args.sg_threshold,
            # CPAC auxiliary reward terms; chi is an optional extension.
            aux_omega_zeta=getattr(tcfg, "aux_omega_zeta", 0.0),
            aux_omega_delta=getattr(tcfg, "aux_omega_delta", 0.0),
            aux_omega_c=getattr(tcfg, "aux_omega_c", 0.0),
            aux_omega_chi=getattr(tcfg, "aux_omega_chi", 0.0),
            aux_chi_max=getattr(tcfg, "aux_chi_max", 3.0),
            aux_edge_local_zeta=bool(getattr(tcfg, "aux_edge_local_zeta", False)),
            # Reward normalizer.
            reward_normalize=bool(getattr(tcfg, "reward_normalize", False)),
            reward_norm_alpha=float(getattr(tcfg, "reward_norm_alpha", 0.99)),
            reward_norm_clip_std=float(getattr(tcfg, "reward_norm_clip_std", 5.0)),
            reward_norm_warmup_steps=int(getattr(tcfg, "reward_norm_warmup_steps", 100)),
            # n-step commit fix.
            nstep_fix=bool(getattr(tcfg, "nstep_fix", False)),
        ),
        "shaping_cfg": dict(
            beta=sw.beta, omega_b=sw.omega_b, kappa=sw.kappa, eta=sw.eta,
        ),
    }
    torch.save(save_dict, ckpt_path)

    elapsed = time.time() - t0
    print(f"[train] done in {elapsed:.1f}s ({elapsed / 60:.1f} min) -> {ckpt_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
