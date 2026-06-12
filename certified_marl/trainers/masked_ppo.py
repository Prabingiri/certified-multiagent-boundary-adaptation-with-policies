r"""Masked PPO trainer for CPAC.

The actor samples actions in the controller-centered frame. The environment
executes actions in canonical pair order ``(i, j)`` with ``i < j``, so the
sampled sign is flipped whenever the controller is endpoint ``j``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from certified_marl.env.csgrag import CSGRAGEnv, CSGRAGState
from certified_marl.models.flat_actor import FlatCritic, FlatEdgeActor
from certified_marl.objectives.shaping import ShapingWeights, psi_ij
from certified_marl.shield.feasibility_kernel import safe_action_set
from certified_marl.shield.matching import active_interfaces, controller_of
from certified_marl.trainers._common import (
    RolloutBuffer,
    Transition,
    clipped_ppo_loss,
    compute_gae,
    masked_log_probs_and_entropy,
)


@dataclass
class MaskedPPOConfig:
    """Hyperparameters for masked PPO training."""
    lr_actor: float = 3e-4
    lr_critic: float = 1e-3
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    epochs_per_update: int = 4
    minibatch_size: int = 64
    max_grad_norm: float = 0.5
    rollout_length: int = 256
    admission_reward_coef: float = 0.0
    epsilon_start: float = 0.0
    epsilon_final: float = 0.0
    n_step_horizon: int = 1
    use_extended_state: bool = True
    kl_prior_coef_start: float = 0.0
    kl_prior_coef_final: float = 0.0
    bc_pretrain_steps: int = 0
    bc_pretrain_epochs: int = 4
    bc_pretrain_lr: float = 1e-3
    bc_label_smoothing: float = 0.0
    entropy_floor_coef: float = 0.0
    entropy_floor_frac: float = 0.0
    aux_omega_zeta: float = 0.0
    aux_omega_delta: float = 0.0
    aux_omega_c: float = 0.0
    aux_omega_chi: float = 0.0
    aux_chi_max: float = 3.0
    aux_edge_local_zeta: bool = False
    reward_normalize: bool = False
    reward_norm_alpha: float = 0.99
    reward_norm_clip_std: float = 5.0
    reward_norm_warmup_steps: int = 100
    nstep_fix: bool = False


def actor_state_vec(state: CSGRAGState, i: int, j: int,
                    include_m_i: bool = False,
                    speed: float = 1.0) -> np.ndarray:
    """Build the controller-centered actor state for one active interface."""
    c = controller_of(i, j, [a.load_pressure_ewma for a in state.agents])
    cbar = j if c == i else i
    agent_c = state.agents[c]
    agent_cbar = state.agents[cbar]
    ifs = state.interfaces[(min(i, j), max(i, j))]
    if c == ifs.i:
        delta_c_cbar = ifs.band_ij.delta
        delta_cbar_c = ifs.band_ji.delta
    else:
        delta_c_cbar = ifs.band_ji.delta
        delta_cbar_c = ifs.band_ij.delta
    feats = [
        agent_c.load_pressure_ewma,
        float(agent_c.q_i),
        agent_c.latent_overload_ewma,
        agent_cbar.load_pressure_ewma,
        delta_c_cbar,
        delta_cbar_c,
    ]
    if include_m_i:
        from certified_marl.env.cs_lstf import least_slack_margin
        m_c = least_slack_margin(agent_c, float(state.t), speed=speed)
        U_bar_c = max(float(agent_c.U_bar), 1e-9)
        m_c_norm = float(np.clip(m_c / U_bar_c, -1.0, 1.0))
        feats.append(m_c_norm)
    return np.asarray(feats, dtype=np.float32)


def critic_state_vec(state: CSGRAGState, i: int, j: int,
                     include_m_i: bool = False,
                     speed: float = 1.0) -> np.ndarray:
    """Build the critic state from actor features plus endpoint context."""
    s_actor = actor_state_vec(state, i, j, include_m_i=include_m_i,
                              speed=speed)
    a_i = state.agents[i]
    a_j = state.agents[j]
    extra = np.array([
        float(a_j.q_i),
        a_i.U_current / max(a_i.U_bar, 1e-9),
        a_j.U_current / max(a_j.U_bar, 1e-9),
    ], dtype=np.float32)
    return np.concatenate([s_actor, extra])


def orient_safe_actions_for_controller(
    safe: list[int],
    *,
    controller: int,
    canonical_i: int,
) -> list[int]:
    """Convert canonical-frame safe actions into the actor's controller frame."""
    sign = 1 if controller == canonical_i else -1
    return [sign * int(a) for a in safe]


def mask_from_safe_actions(safe: list[int]) -> np.ndarray:
    """Map signed actions in {-1, 0, +1} to a length-3 float mask."""
    m = np.zeros(3, dtype=np.float32)
    for a in safe:
        m[a + 1] = 1.0
    return m


class _RolloutCollector:
    """Collect active-interface transitions for PPO updates."""

    def __init__(self, env: CSGRAGEnv,
                 shaping_w: ShapingWeights | None = None,
                 admission_reward_coef: float = 0.0,
                 epsilon: float = 0.0,
                 n_step_horizon: int = 1,
                 use_extended_state: bool = True,
                 prior_action_fn=None,
        ):
        self.env = env
        self.shaping_w = shaping_w or ShapingWeights()
        self.admission_reward_coef = float(admission_reward_coef)
        self.epsilon = float(epsilon)
        self.n_step_horizon = max(1, int(n_step_horizon))
        self.use_extended_state = bool(use_extended_state)
        self.prior_action_fn = prior_action_fn

    def collect(
        self,
        actor: FlatEdgeActor,
        critic: FlatCritic,
        device: torch.device,
        rollout_length: int,
        state: CSGRAGState | None = None,
    ) -> tuple[RolloutBuffer, CSGRAGState]:
        buf = RolloutBuffer()
        s = state if state is not None else self.env.reset()
        psi_hist: dict[tuple[int, int], list[float]] = {}
        pending_by_edge: dict[tuple[int, int], list[dict]] = {}
        n = self.n_step_horizon

        self._rollout_admitted = 0
        self._rollout_rejected = 0
        self._rollout_completed = 0
        self._rollout_resets = 0

        for _ in range(rollout_length):
            neighbors: dict[int, list[int]] = {i: [] for i in range(self.env.n)}
            for (i, j) in s.interfaces.keys():
                neighbors[i].append(j)
                neighbors[j].append(i)
            loads = [a.load_pressure_ewma for a in s.agents]
            active = active_interfaces(loads, neighbors)

            pending_actions: dict[tuple[int, int], int] = {}
            pending_records: list[dict] = []

            prior_actions_dict: dict[tuple[int, int], int] = {}
            if self.prior_action_fn is not None:
                try:
                    prior_actions_dict = self.prior_action_fn(s, np.random)
                    if prior_actions_dict is None:
                        prior_actions_dict = {}
                except Exception:
                    prior_actions_dict = {}

            _t_now_for_mask = float(s.t) * self.env.dt
            _active_obstacles_for_mask = (
                self.env._current_obstacles(_t_now_for_mask)
                if hasattr(self.env, "_current_obstacles")
                else self.env.obstacles
            )
            for (i, j) in active:
                ifs = s.interfaces[(min(i, j), max(i, j))]
                safe = safe_action_set(s, ifs, self.env.delta_step,
                                       _active_obstacles_for_mask,
                                       speed=self.env.speed)
                c = controller_of(i, j, loads)
                safe_controller = orient_safe_actions_for_controller(
                    safe, controller=c, canonical_i=i,
                )
                mask = mask_from_safe_actions(safe_controller)
                s_actor = actor_state_vec(
                    s, i, j,
                    include_m_i=self.use_extended_state,
                    speed=self.env.speed,
                )
                s_critic = critic_state_vec(
                    s, i, j,
                    include_m_i=self.use_extended_state,
                    speed=self.env.speed,
                )
                psi_before = psi_ij(s, i, j, self.shaping_w,
                                    speed=self.env.speed)

                with torch.no_grad():
                    sa = torch.as_tensor(s_actor, dtype=torch.float32,
                                         device=device).unsqueeze(0)
                    sc = torch.as_tensor(s_critic, dtype=torch.float32,
                                         device=device).unsqueeze(0)
                    logits = actor(sa)
                    mask_t = torch.as_tensor(mask, dtype=torch.float32,
                                             device=device).unsqueeze(0)
                    neg_inf = torch.finfo(logits.dtype).min
                    masked_logits = torch.where(
                        mask_t > 0, logits, torch.full_like(logits, neg_inf),
                    )
                    probs = torch.softmax(masked_logits, dim=-1).squeeze(0)
                    if (not torch.isfinite(probs).all()) or \
                            float(probs.sum().item()) < 1e-9:
                        safe_mask_vec = mask_t.squeeze(0)
                        if float(safe_mask_vec.sum().item()) > 0:
                            probs = safe_mask_vec / safe_mask_vec.sum()
                        else:
                            probs = torch.zeros_like(probs)
                            probs[1] = 1.0
                    if (self.epsilon > 0.0
                            and len(safe_controller) > 0
                            and np.random.random() < self.epsilon):
                        a_safe = int(np.random.choice(safe_controller))
                        action_idx = a_safe + 1
                    else:
                        action_idx = int(torch.multinomial(probs, 1).item())
                    action_signed = action_idx - 1
                    log_prob = torch.log(probs[action_idx] + 1e-12).item()
                    v = critic(sc).item()

                canonical_action = action_signed if c == i else -action_signed
                pending_actions[(min(i, j), max(i, j))] = canonical_action
                edge_key = (min(i, j), max(i, j))
                prior_canonical = int(prior_actions_dict.get(edge_key, 0))
                prior_action_signed = (prior_canonical if c == i
                                       else -prior_canonical)
                pending_records.append(dict(
                    i=i, j=j,
                    s_actor=s_actor, s_critic=s_critic, mask=mask,
                    action_idx=action_idx, action_signed=action_signed,
                    log_prob=log_prob, value=v, psi_before=psi_before,
                    prior_action_signed=prior_action_signed,
                ))


            s, info = self.env.step(pending_actions)

            self._rollout_admitted += int(info.get("admitted", 0))
            self._rollout_rejected += int(info.get("rejected", 0))
            self._rollout_completed += int(info.get("completed", 0))

            cross_admitted = info.get(
                "owner_cross_admitted_by_edge",
                info.get("cross_admitted_by_edge", {}),
            )

            for edge_key, count in cross_admitted.items():
                key = (min(edge_key), max(edge_key))
                pending = pending_by_edge.get(key)
                if pending:
                    pending[-1]["cross_admitted_at_tplus1"] = (
                        float(pending[-1].get("cross_admitted_at_tplus1", 0.0))
                        + float(count)
                    )

            for (ei, ej) in s.interfaces.keys():
                key = (min(ei, ej), max(ei, ej))
                psi_now = psi_ij(s, ei, ej, self.shaping_w, speed=self.env.speed)
                hist = psi_hist.setdefault(key, [])
                hist.append(psi_now)
                if not bool(getattr(self, "nstep_fix", False)):
                    if len(hist) > n + 1:
                        hist.pop(0)

            for rec in pending_records:
                key = (min(rec["i"], rec["j"]), max(rec["i"], rec["j"]))
                pending_by_edge.setdefault(key, []).append(dict(
                    rec=rec,
                    psi_before=rec["psi_before"],
                    snap_len=len(psi_hist[key]),
                    cross_admitted_at_tplus1=0.0,
                    done_at_t=info["done"],
                ))

            for key, queue in list(pending_by_edge.items()):
                hist_len = len(psi_hist.get(key, []))
                still_pending = []
                for p in queue:
                    delta = hist_len - p["snap_len"]
                    if delta >= n or info["done"]:
                        eff = min(n, max(1, delta))
                        future = psi_hist[key][p["snap_len"] : p["snap_len"] + eff]
                        psi_window = [p["psi_before"]] + list(future)
                        rec = p["rec"]
                        from certified_marl.objectives.shaping import (
                            pairwise_reward_nstep,
                        )
                        r = pairwise_reward_nstep(
                            psi_window, rec["action_signed"], n, self.shaping_w,
                        )
                        if self.admission_reward_coef > 0.0:
                            r = r + self.admission_reward_coef * float(
                                p.get("cross_admitted_at_tplus1", 0.0)
                            )
                        _aux_any = (
                            float(getattr(self, "aux_omega_zeta", 0.0)) > 0.0
                            or float(getattr(self, "aux_omega_delta", 0.0)) > 0.0
                            or float(getattr(self, "aux_omega_c", 0.0)) > 0.0
                            or float(getattr(self, "aux_omega_chi", 0.0)) > 0.0
                        )
                        if _aux_any:
                            from certified_marl.objectives.auxiliary_reward import (
                                AuxiliaryRewardWeights,
                                auxiliary_reward_terms,
                            )
                            _aux_w = AuxiliaryRewardWeights(
                                omega_zeta=float(getattr(self, "aux_omega_zeta", 0.0)),
                                omega_Delta=float(getattr(self, "aux_omega_delta", 0.0)),
                                omega_c=float(getattr(self, "aux_omega_c", 0.0)),
                                omega_chi=float(getattr(self, "aux_omega_chi", 0.0)),
                                intensity_dt=float(getattr(self.env, "dt", 1.0)),
                                chi_max=float(getattr(self, "aux_chi_max", 3.0)),
                                use_edge_local_zeta=bool(getattr(self, "aux_edge_local_zeta", False)),
                            )
                            r = r + auxiliary_reward_terms(
                                s, int(rec["i"]), int(rec["j"]), _aux_w,
                            )
                        if bool(getattr(self, "reward_normalize", False)) and \
                                getattr(self, "reward_normalizer", None) is not None:
                            r = float(self.reward_normalizer.update_and_normalize(r))
                        _buf_action = rec["action_signed"]
                        _buf_prior = rec.get("prior_action_signed",
                                             rec["action_signed"])
                        buf.add(Transition(
                            state_actor=rec["s_actor"],
                            state_critic=rec["s_critic"],
                            action=_buf_action,
                            mask=rec["mask"],
                            log_prob=rec["log_prob"],
                            reward=r,
                            value=rec["value"],
                            done=p["done_at_t"],
                            prior_action=_buf_prior,
                        ))
                    else:
                        still_pending.append(p)
                if still_pending:
                    pending_by_edge[key] = still_pending
                else:
                    del pending_by_edge[key]

            if info["done"]:
                psi_hist.clear()
                pending_by_edge.clear()
                self._rollout_resets += 1
                s = self.env.reset()

        return buf, s


class MaskedPPOTrainer:
    """CPAC masked PPO trainer."""

    def __init__(self, env: CSGRAGEnv, actor: FlatEdgeActor, critic: FlatCritic,
                 config: MaskedPPOConfig | None = None,
                 device: torch.device | None = None,
                 prior_action_fn=None):
        self.env = env
        self.actor = actor
        self.critic = critic
        self.cfg = config or MaskedPPOConfig()
        self.device = device or torch.device("cpu")
        self.actor.to(self.device)
        self.critic.to(self.device)
        self.opt_actor = torch.optim.Adam(self.actor.parameters(),
                                          lr=self.cfg.lr_actor)
        self.opt_critic = torch.optim.Adam(self.critic.parameters(),
                                           lr=self.cfg.lr_critic)
        self.collector = _RolloutCollector(
            env,
            admission_reward_coef=self.cfg.admission_reward_coef,
            epsilon=self.cfg.epsilon_start,
            n_step_horizon=getattr(self.cfg, "n_step_horizon", 1),
            use_extended_state=getattr(self.cfg, "use_extended_state", True),
            prior_action_fn=prior_action_fn,
        )
        self.collector.aux_omega_zeta = float(getattr(self.cfg, "aux_omega_zeta", 0.0))
        self.collector.aux_omega_delta = float(getattr(self.cfg, "aux_omega_delta", 0.0))
        self.collector.aux_omega_c = float(getattr(self.cfg, "aux_omega_c", 0.0))
        self.collector.aux_omega_chi = float(getattr(self.cfg, "aux_omega_chi", 0.0))
        self.collector.aux_chi_max = float(getattr(self.cfg, "aux_chi_max", 3.0))
        self.collector.aux_edge_local_zeta = bool(getattr(self.cfg, "aux_edge_local_zeta", False))
        self.collector.reward_normalize = bool(getattr(self.cfg, "reward_normalize", False))
        if self.collector.reward_normalize:
            from certified_marl.objectives.reward_normalizer import (
                RewardNormalizer,
                RewardNormalizerConfig,
            )
            _rn_cfg = RewardNormalizerConfig(
                alpha=float(getattr(self.cfg, "reward_norm_alpha", 0.99)),
                clip_std=float(getattr(self.cfg, "reward_norm_clip_std", 5.0)),
                warmup_steps=int(getattr(self.cfg, "reward_norm_warmup_steps", 100)),
            )
            self.collector.reward_normalizer = RewardNormalizer(_rn_cfg)
        else:
            self.collector.reward_normalizer = None
        self.collector.nstep_fix = bool(getattr(self.cfg, "nstep_fix", False))
        self.prior_action_fn = prior_action_fn
        self._kl_prior_coef: float = float(
            getattr(self.cfg, "kl_prior_coef_start", 0.0)
        )
        self._state: Optional[CSGRAGState] = None

    def collect(self) -> RolloutBuffer:
        buf, self._state = self.collector.collect(
            self.actor, self.critic, self.device,
            rollout_length=self.cfg.rollout_length, state=self._state,
        )
        return buf

    def update(self, buf: RolloutBuffer) -> dict[str, float]:
        data = buf.as_tensors(self.device)
        adv_gae, returns = compute_gae(
            data["rewards"], data["values"], data["dones"],
            last_value=0.0, gamma=self.cfg.gamma, lam=self.cfg.gae_lambda,
        )
        adv_gae = (adv_gae - adv_gae.mean()) / (adv_gae.std() + 1e-8)

        N = data["states_actor"].shape[0]
        idx = np.arange(N)
        diagnostics: dict[str, float] = {}
        for _ in range(self.cfg.epochs_per_update):
            np.random.shuffle(idx)
            for start in range(0, N, self.cfg.minibatch_size):
                mb = idx[start:start + self.cfg.minibatch_size]
                mb_t = torch.as_tensor(mb, dtype=torch.long, device=self.device)
                sa = data["states_actor"][mb_t]
                sc = data["states_critic"][mb_t]
                acts = data["actions"][mb_t]
                masks = data["masks"][mb_t]
                old_lp = data["old_log_probs"][mb_t]
                adv_gae_mb = adv_gae[mb_t]
                ret_mb = returns[mb_t]

                logits = self.actor(sa)
                new_lp, ent = masked_log_probs_and_entropy(logits, masks, acts)

                adv_for_ppo = adv_gae_mb

                loss_pi = clipped_ppo_loss(new_lp, old_lp, adv_for_ppo,
                                           self.cfg.clip_eps)
                loss_ent = -self.cfg.entropy_coef * ent.mean()
                mask_count = masks.sum(dim=-1).clamp_min(1.0)
                entropy_max = torch.log(mask_count)
                entropy_norm = torch.zeros_like(ent)
                multi_action = entropy_max > 1e-8
                if multi_action.any():
                    entropy_norm[multi_action] = (
                        ent[multi_action] / entropy_max[multi_action]
                    )
                floor_coef = float(getattr(self.cfg, "entropy_floor_coef", 0.0))
                floor_frac = float(getattr(self.cfg, "entropy_floor_frac", 0.0))
                if floor_coef > 0.0 and floor_frac > 0.0:
                    entropy_target = floor_frac * entropy_max
                    loss_entropy_floor = floor_coef * torch.relu(
                        entropy_target - ent
                    ).pow(2).mean()
                else:
                    loss_entropy_floor = torch.zeros((), device=self.device)

                loss_kl_prior = torch.zeros((), device=self.device)
                if (self.prior_action_fn is not None
                        and self._kl_prior_coef > 0.0):
                    neg_inf_kl = torch.finfo(logits.dtype).min
                    masked_logits_kl = torch.where(
                        masks > 0, logits,
                        torch.full_like(logits, neg_inf_kl),
                    )
                    cur_log_probs_kl = torch.log_softmax(
                        masked_logits_kl, dim=-1,
                    )
                    prior_acts = data["prior_actions"][mb_t]
                    nll = -cur_log_probs_kl.gather(
                        1, prior_acts.unsqueeze(-1),
                    ).squeeze(-1)
                    prior_safe = masks.gather(
                        1, prior_acts.unsqueeze(-1),
                    ).squeeze(-1)
                    nll = nll * prior_safe
                    loss_kl_prior = self._kl_prior_coef * nll.mean()

                v_pred = self.critic(sc)
                loss_v = self.cfg.value_coef * F.mse_loss(v_pred, ret_mb)

                self.opt_actor.zero_grad()
                (loss_pi + loss_ent + loss_kl_prior
                 + loss_entropy_floor).backward()
                torch.nn.utils.clip_grad_norm_(
                    self.actor.parameters(), self.cfg.max_grad_norm,
                )
                self.opt_actor.step()

                self.opt_critic.zero_grad()
                loss_v.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.critic.parameters(), self.cfg.max_grad_norm,
                )
                self.opt_critic.step()

                diagnostics = dict(
                    loss_pi=loss_pi.item(),
                    loss_v=loss_v.item(),
                    entropy=ent.mean().item(),
                    entropy_norm=entropy_norm.mean().item(),
                    mask_card_mean=mask_count.mean().item(),
                    mask_multi_rate=multi_action.to(torch.float32).mean().item(),
                    adv_mean=adv_for_ppo.mean().item(),
                    ret_mean=ret_mb.mean().item(),
                )
                if floor_coef > 0.0 and floor_frac > 0.0:
                    diagnostics["loss_entropy_floor"] = float(
                        loss_entropy_floor.item()
                    )
                diagnostics["kl_prior_coef"] = float(self._kl_prior_coef)
                if self.prior_action_fn is not None and self._kl_prior_coef > 0.0:
                    diagnostics["loss_kl_prior"] = float(loss_kl_prior.item())
        return diagnostics

    def pretrain_bc(self, n_steps: int, n_epochs: int = 4,
                    lr: float | None = None) -> dict[str, float]:
        """Supervised actor warm start from the prior policy."""
        if self.prior_action_fn is None or n_steps <= 0:
            return {"bc_skipped": 1.0}
        s = self._state if self._state is not None else self.env.reset()
        recs: list[dict] = []
        steps_done = 0
        from certified_marl.shield.feasibility_kernel import safe_action_set
        from certified_marl.shield.matching import active_interfaces, controller_of
        while steps_done < n_steps:
            neighbors: dict[int, list[int]] = {i: [] for i in range(self.env.n)}
            for (i, j) in s.interfaces.keys():
                neighbors[i].append(j)
                neighbors[j].append(i)
            loads = [a.load_pressure_ewma for a in s.agents]
            active = active_interfaces(loads, neighbors)
            try:
                prior_actions_dict = self.prior_action_fn(s, np.random) or {}
            except Exception:
                prior_actions_dict = {}
            _t_now_for_mask_bc = float(s.t) * self.env.dt
            _active_obstacles_bc = (
                self.env._current_obstacles(_t_now_for_mask_bc)
                if hasattr(self.env, "_current_obstacles")
                else self.env.obstacles
            )
            for (i, j) in active:
                ifs = s.interfaces[(min(i, j), max(i, j))]
                safe = safe_action_set(s, ifs, self.env.delta_step,
                                       _active_obstacles_bc,
                                       speed=self.env.speed)
                s_actor = actor_state_vec(
                    s, i, j,
                    include_m_i=self.collector.use_extended_state,
                    speed=self.env.speed,
                )
                edge_key = (min(i, j), max(i, j))
                prior_canonical = int(prior_actions_dict.get(edge_key, 0))
                c = controller_of(i, j, loads)
                safe_controller = orient_safe_actions_for_controller(
                    safe, controller=c, canonical_i=i,
                )
                mask_arr = mask_from_safe_actions(safe_controller)
                prior_signed = (prior_canonical if c == i
                                else -prior_canonical)
                prior_idx = prior_signed + 1
                recs.append(dict(s_actor=s_actor, mask=mask_arr,
                                 prior_idx=int(prior_idx)))
            actions = {(min(i, j), max(i, j)):
                       int(prior_actions_dict.get((min(i, j), max(i, j)), 0))
                       for (i, j) in s.interfaces.keys()}
            s, info = self.env.step(actions)
            steps_done += 1
            if info["done"]:
                s = self.env.reset()
        self._state = s

        if not recs:
            return {"bc_records": 0.0}
        sa_t = torch.as_tensor(
            np.stack([r["s_actor"] for r in recs]),
            dtype=torch.float32, device=self.device,
        )
        mask_t = torch.as_tensor(
            np.stack([r["mask"] for r in recs]),
            dtype=torch.float32, device=self.device,
        )
        labels = torch.as_tensor(
            [r["prior_idx"] for r in recs],
            dtype=torch.long, device=self.device,
        )
        bc_lr = float(lr if lr is not None
                      else getattr(self.cfg, "bc_pretrain_lr", 1e-3))
        opt_bc = torch.optim.Adam(self.actor.parameters(), lr=bc_lr)
        N = sa_t.shape[0]
        idx = np.arange(N)
        last_loss = float("nan")
        last_acc = float("nan")
        for ep in range(int(n_epochs)):
            np.random.shuffle(idx)
            for start in range(0, N, max(32, self.cfg.minibatch_size)):
                mb = idx[start:start + max(32, self.cfg.minibatch_size)]
                mb_t_ = torch.as_tensor(mb, dtype=torch.long,
                                        device=self.device)
                logits = self.actor(sa_t[mb_t_])
                neg_inf = torch.finfo(logits.dtype).min
                masked_logits = torch.where(
                    mask_t[mb_t_] > 0, logits,
                    torch.full_like(logits, neg_inf),
                )
                log_probs = torch.log_softmax(masked_logits, dim=-1)
                smooth = float(getattr(self.cfg, "bc_label_smoothing", 0.0))
                smooth = min(max(smooth, 0.0), 0.95)
                if smooth > 0.0:
                    safe = mask_t[mb_t_]
                    safe_count = safe.sum(dim=-1, keepdim=True).clamp_min(1.0)
                    target = smooth * safe / safe_count
                    target.scatter_add_(
                        1,
                        labels[mb_t_].unsqueeze(-1),
                        torch.full(
                            (len(mb_t_), 1),
                            1.0 - smooth,
                            dtype=target.dtype,
                            device=target.device,
                        ),
                    )
                    target = target * safe
                    target = target / target.sum(dim=-1, keepdim=True).clamp_min(1e-8)
                    loss = -(target * log_probs).sum(dim=-1).mean()
                else:
                    nll = -log_probs.gather(1, labels[mb_t_].unsqueeze(-1)
                                            ).squeeze(-1)
                    loss = nll.mean()
                opt_bc.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.actor.parameters(), self.cfg.max_grad_norm,
                )
                opt_bc.step()
                last_loss = float(loss.item())
                with torch.no_grad():
                    pred = log_probs.argmax(dim=-1)
                    last_acc = float((pred == labels[mb_t_]).float().mean().item())
        return {
            "bc_records": float(N),
            "bc_final_loss": last_loss,
            "bc_final_acc": last_acc,
            "bc_label_smoothing": float(getattr(
                self.cfg, "bc_label_smoothing", 0.0,
            )),
        }
