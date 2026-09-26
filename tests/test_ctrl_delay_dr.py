"""Tests for the config-gated MJLab ctrl-delay domain randomization.

These build a real (CPU) K1 MJLab env via ``build_ufo_mjlab_config`` +
``HumanoidVerseMjlabConfig.build()``, the same construction path
``tests/test_mjlab_ground_friction.py`` uses, so they exercise the actual
``DcMotorActuatorCfg``/``DelayBuffer`` wiring rather than a mock.

Vocabulary: a "tick" is one control step (``decimation`` physics steps, 4 for
K1 at 200 Hz physics / 50 Hz control). ``ctrl_delay_step_range`` is in ticks;
MJLab's ``delay_min_lag``/``delay_max_lag`` are in physics steps.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from humanoidverse.train import build_ufo_mjlab_config

REPO_ROOT = Path(__file__).resolve().parents[1]
K1_CONFIG = REPO_ROOT / "configs/robots/k1_22dof.yaml"
K1_MOTION_DATA = REPO_ROOT / "cache/motion_data/k1_lafan1/k1_lafan1_train_near10s_ufo.pkl"

pytestmark = pytest.mark.skipif(
    not (K1_CONFIG.exists() and K1_MOTION_DATA.exists()),
    reason="K1 robot config or cached lafan1 motion data not present",
)


def _build_core(num_envs: int, extra_overrides: list[str], seed: int = 3):
    train_cfg = build_ufo_mjlab_config(
        device="cpu",
        work_dir="/tmp/ufo_ctrl_delay_dr_test",
        num_envs=num_envs,
        num_env_steps=1,
        seed=seed,
        use_wandb=False,
        wandb_run_name=None,
        smoke=True,
        robot_config=K1_CONFIG,
        data_path=str(K1_MOTION_DATA),
    )
    env_cfg = train_cfg.env
    robot_xml = Path(env_cfg.mjcf_path)
    if not robot_xml.exists():
        pytest.skip(f"K1 MJCF not present: {robot_xml}")
    env_cfg = env_cfg.model_copy(update={"hydra_overrides": [*env_cfg.hydra_overrides, *extra_overrides]})
    env, _meta = env_cfg.build(num_envs=num_envs)
    return env.base_env


def _fixed_action_sequence(num_envs: int, num_dof: int, num_ticks: int, seed: int) -> list[torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    return [torch.randn(num_envs, num_dof, generator=gen) * 0.3 for _ in range(num_ticks)]


# ---------------------------------------------------------------------------
# (a) flag off: identical, deterministic, no delay actuators constructed.
# ---------------------------------------------------------------------------


def test_ctrl_delay_disabled_builds_no_delay_actuators_and_is_deterministic() -> None:
    torch.manual_seed(123)
    core_a = _build_core(2, [])
    assert core_a._ctrl_delay_hi_steps == 0
    assert core_a._ctrl_delay_actuators == []
    assert not any(getattr(act, "has_delay", False) for act in core_a.robot.actuators)

    actions = _fixed_action_sequence(2, core_a.num_dof, 6, seed=42)
    targets_a = []
    for action in actions:
        core_a.step(action)
        targets_a.append(core_a.robot.data.joint_pos_target.clone())

    torch.manual_seed(123)
    core_b = _build_core(2, [])
    targets_b = []
    for action in actions:
        core_b.step(action)
        targets_b.append(core_b.robot.data.joint_pos_target.clone())

    for ta, tb in zip(targets_a, targets_b):
        torch.testing.assert_close(ta, tb, atol=0.0, rtol=0.0)


# ---------------------------------------------------------------------------
# (b) flag on, forced lag=8 physics steps (ctrl_delay_step_range=[2,2]):
#     applied PD target equals the action generated 2 ticks earlier, and the
#     first `lag` physics steps after reset hold the "zero action" pose
#     (default_dof_pos + default_dof_pos_offset), not the stale 0-rad target
#     MJLab's own Entity.reset()/clear_state() would otherwise backfill with.
# ---------------------------------------------------------------------------


def test_ctrl_delay_forced_lag_applies_action_from_two_ticks_earlier() -> None:
    torch.manual_seed(7)
    core = _build_core(3, ["domain_rand.randomize_ctrl_delay=true", "domain_rand.ctrl_delay_step_range=[2,2]"])

    decimation = int(core.config.simulator.config.sim.control_decimation)
    assert decimation == 4
    assert core._ctrl_delay_lo_steps == 8
    assert core._ctrl_delay_hi_steps == 8
    assert core._ctrl_delay_actuators, "expected at least one delay-configured actuator"

    act = core._ctrl_delay_actuators[0]
    lags = act._delay_buffer.current_lags
    assert torch.all(lags == 8), f"forced [2,2] range must yield lag=8 physics steps for every env, got {lags}"

    (target_id,) = act.target_ids.tolist()
    dof_index = core._joint_ids.tolist().index(target_id)
    zero_action_target = (core.default_dof_pos + core.default_dof_pos_offset)[:, dof_index : dof_index + 1].clone()

    pre_delay: list[torch.Tensor] = []
    post_delay: list[torch.Tensor] = []
    orig_apply_delay = act.apply_delay

    def spy_apply_delay(cmd):
        pre_delay.append(cmd.position_target.clone())
        out = orig_apply_delay(cmd)
        post_delay.append(out.position_target.clone())
        return out

    act.apply_delay = spy_apply_delay

    num_ticks = 6
    actions = _fixed_action_sequence(3, core.num_dof, num_ticks, seed=99)
    for action in actions:
        core.step(action)

    assert len(post_delay) == num_ticks * decimation
    lag = 8
    for p in range(len(post_delay)):
        expected = zero_action_target if p < lag else pre_delay[p - lag]
        torch.testing.assert_close(post_delay[p], expected, atol=1e-6, rtol=0.0)

    # Sanity: the pre-delay (raw commanded) target is constant across the 4
    # physics substeps of a tick, and differs tick-to-tick (actions differ).
    for tick in range(num_ticks):
        sub = pre_delay[tick * decimation : (tick + 1) * decimation]
        for s in sub[1:]:
            torch.testing.assert_close(s, sub[0])


# ---------------------------------------------------------------------------
# (c) lag is constant within an episode and resampled across resets, within range.
# ---------------------------------------------------------------------------


def test_ctrl_delay_lag_constant_within_episode_and_resampled_on_reset() -> None:
    torch.manual_seed(11)
    core = _build_core(4, ["domain_rand.randomize_ctrl_delay=true", "domain_rand.ctrl_delay_step_range=[0,2]"])
    assert core._ctrl_delay_lo_steps == 0
    assert core._ctrl_delay_hi_steps == 8

    act = core._ctrl_delay_actuators[0]

    actions = _fixed_action_sequence(4, core.num_dof, 10, seed=5)
    lag_snapshots = []
    for action in actions:
        core.step(action)
        lag_snapshots.append(act._delay_buffer.current_lags.clone())
    for snap in lag_snapshots[1:]:
        torch.testing.assert_close(snap, lag_snapshots[0])

    observed_env0 = []
    for _ in range(40):
        core.reset_idx(torch.tensor([0], dtype=torch.long))
        observed_env0.append(int(act._delay_buffer.current_lags[0].item()))
    assert all(0 <= v <= 8 for v in observed_env0)
    assert len(set(observed_env0)) > 1, f"expected varying lags across resets, got {observed_env0}"


# ---------------------------------------------------------------------------
# (d) history/last_action always records the GENERATED action, unaffected by delay.
# ---------------------------------------------------------------------------


def test_ctrl_delay_history_and_last_action_record_generated_action() -> None:
    torch.manual_seed(21)
    core = _build_core(2, ["domain_rand.randomize_ctrl_delay=true", "domain_rand.ctrl_delay_step_range=[0,2]"])

    actions = _fixed_action_sequence(2, core.num_dof, 5, seed=13)
    for action in actions:
        expected = core._normalized_action(action.to(core.device))
        core.step(action)
        torch.testing.assert_close(core.actions, expected)

    obs = core.get_observation(to_numpy=False)
    torch.testing.assert_close(obs["last_action"], core._normalized_action(actions[-1].to(core.device)))
