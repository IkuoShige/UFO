from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import mujoco
import pytest
import torch

from humanoidverse.agents.envs.humanoidverse_mjlab import (
    _compose_humanoidverse_config,
    _randomize_ground_contact_friction,
    _strip_embedded_ground_planes,
    make_mjlab_ufo_env_cfg,
)
from humanoidverse.train import build_ufo_mjlab_config

REPO_ROOT = Path(__file__).resolve().parents[1]
K1_CONFIG = REPO_ROOT / "configs/robots/k1_22dof.yaml"


def test_strip_embedded_world_plane_keeps_robot_geometry() -> None:
    spec = mujoco.MjSpec()
    spec.worldbody.add_geom(name="embedded_ground", type=mujoco.mjtGeom.mjGEOM_PLANE)
    body = spec.worldbody.add_body(name="robot")
    body.add_geom(name="collision", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=(0.1, 0.0, 0.0))

    removed = _strip_embedded_ground_planes(spec)

    assert removed == ("embedded_ground",)
    assert [geom.name for geom in spec.geoms] == ["collision"]


def test_shared_friction_draw_is_applied_to_robot_and_terrain_per_environment() -> None:
    num_envs = 6
    friction = torch.full((num_envs, 4, 3), -1.0)
    robot = SimpleNamespace(indexing=SimpleNamespace(geom_ids=torch.tensor([0, 1, 2])))
    terrain = SimpleNamespace(indexing=SimpleNamespace(geom_ids=torch.tensor([3])))
    env = SimpleNamespace(
        num_envs=num_envs,
        device="cpu",
        scene={"robot": robot, "terrain": terrain},
        sim=SimpleNamespace(model=SimpleNamespace(geom_friction=friction)),
    )
    robot_cfg = SimpleNamespace(name="robot", geom_ids=slice(None))
    terrain_cfg = SimpleNamespace(name="terrain", geom_ids=slice(None))

    torch.manual_seed(7)
    _randomize_ground_contact_friction(
        env,
        torch.tensor([1, 3, 5]),
        ranges=(0.2, 0.6),
        robot_asset_cfg=robot_cfg,
        terrain_asset_cfg=terrain_cfg,
    )

    selected = friction[[1, 3, 5], :, 0]
    assert torch.all(selected >= 0.2)
    assert torch.all(selected <= 0.6)
    torch.testing.assert_close(selected, selected[:, :1].expand_as(selected))
    assert torch.all(friction[[0, 2, 4], :, 0] == -1.0)
    assert torch.all(friction[:, :, 1:] == -1.0)


@pytest.mark.parametrize("ranges", [(-0.1, 0.5), (0.7, 0.2)])
def test_shared_friction_draw_rejects_invalid_range(ranges: tuple[float, float]) -> None:
    env = SimpleNamespace(num_envs=1, device="cpu")
    with pytest.raises(ValueError, match="Invalid friction range"):
        _randomize_ground_contact_friction(
            env,
            None,
            ranges=ranges,
            robot_asset_cfg=None,
            terrain_asset_cfg=None,
        )


def test_friction_event_requests_per_world_model_storage() -> None:
    assert _randomize_ground_contact_friction.model_fields == ("geom_friction",)


@pytest.mark.skipif(not K1_CONFIG.exists(), reason="K1 robot config not present")
def test_k1_training_scene_has_one_ground_and_reference_friction_event() -> None:
    from mjlab.scene import Scene

    train_cfg = build_ufo_mjlab_config(
        device="cpu",
        work_dir="/tmp/ufo_k1_friction_test",
        num_envs=2,
        num_env_steps=1,
        seed=3,
        use_wandb=False,
        wandb_run_name=None,
        smoke=True,
        robot_config=K1_CONFIG,
    )
    env_cfg = train_cfg.env
    robot_xml = Path(env_cfg.mjcf_path)
    if not robot_xml.exists():
        pytest.skip(f"K1 MJCF not present: {robot_xml}")

    hv_cfg, _ = _compose_humanoidverse_config(
        num_envs=2,
        relative_config_path=env_cfg.relative_config_path,
        hydra_overrides=list(env_cfg.hydra_overrides),
        lafan_tail_path=env_cfg.lafan_tail_path,
        data_mix_weights=env_cfg.data_mix_weights,
        disable_obs_noise=env_cfg.disable_obs_noise,
        disable_domain_randomization=env_cfg.disable_domain_randomization,
        max_episode_length_s=env_cfg.max_episode_length_s,
        root_height_obs=env_cfg.root_height_obs,
        robot_training=env_cfg.robot_training,
    )
    mjlab_cfg = make_mjlab_ufo_env_cfg(
        hv_cfg,
        num_envs=2,
        seed=3,
        mjcf_path=env_cfg.mjcf_path,
        auto_reset=False,
        robot_training=env_cfg.robot_training,
    )
    model = Scene(mjlab_cfg.scene, device="cpu").compile()

    planes = [i for i, geom_type in enumerate(model.geom_type) if int(geom_type) == int(mujoco.mjtGeom.mjGEOM_PLANE)]
    assert len(planes) == 1
    assert model.geom(planes[0]).name == "terrain"
    event = mjlab_cfg.events["random_ground_contact_friction"]
    assert event.func is _randomize_ground_contact_friction
    assert event.params["ranges"] == (0.5, 1.25)
