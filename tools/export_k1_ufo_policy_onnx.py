#!/usr/bin/env python3
"""Export the K1 UFO FB-CPR checkpoint to ONNX for sim2sim / hardware deploy.

Produces, in ``<output_dir>``:

* ``<ModelName>.onnx``       -- meta policy: ``actor_obs`` -> ``action``
* ``<ModelName>.meta.json``  -- authoritative export spec (key order + dims)
* ``backward_encoder.onnx``  -- ``(state, last_action, privileged_state)`` -> ``z``
* ``deploy_spec.json``       -- everything the deploy node needs (obs layout, PD, joints)

Nothing here is inferred: the actor input key order and per-key dims come from
``_infer_policy_onnx_export_spec`` (humanoidverse/utils/helpers.py:319), and the
control constants come from the robot training YAML that trained the checkpoint.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import yaml

from humanoidverse.agents.load_utils import load_model_from_checkpoint_dir
from humanoidverse.export.backward_encoder import export_backward_encoder_from_model
from humanoidverse.utils.helpers import _infer_policy_onnx_export_spec, export_meta_policy_as_onnx

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = REPO_ROOT / "runs" / "ufo_fb_k1_5090_v2"
DEFAULT_ROBOT_CONFIG = REPO_ROOT / "configs" / "robots" / "k1_22dof.yaml"


def _match_joint_value(joint_name: str, value_by_substring: dict[str, float], default: float = 0.0) -> float:
    """Exact mirror of humanoidverse/agents/envs/humanoidverse_mjlab.py:147 (first substring wins)."""
    for key, value in value_by_substring.items():
        if key in joint_name:
            return float(value)
    return float(default)


def _resolve_xml_path(xml_path: str, config_path: Path) -> Path:
    """Mirror of humanoidverse/utils/robot_spec/mujoco_parser.py:29 (config dir, then repo root)."""
    raw = Path(xml_path).expanduser()
    if raw.is_absolute():
        return raw.resolve()
    for candidate in (config_path.parent / raw, REPO_ROOT / raw):
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"Robot XML not found for {xml_path!r} relative to {config_path} or {REPO_ROOT}")


def build_deploy_spec(robot_cfg_path: Path, export_spec: dict) -> dict:
    cfg = yaml.safe_load(robot_cfg_path.read_text())
    training = cfg["training"]
    control = training["control"]
    dof_names = list(cfg["control_joints"]["names"])

    stiffness = {k: float(v) for k, v in control["stiffness"].items()}
    damping = {k: float(v) for k, v in control["damping"].items()}
    effort_limits = [float(x) for x in control["effort_limit"]]
    velocity_limits = [float(x) for x in control["velocity_limit"]]
    default_angles = training["init_state"]["default_joint_angles"]

    kp = [_match_joint_value(n, stiffness) for n in dof_names]
    kd = [_match_joint_value(n, damping) for n in dof_names]
    action_scale = float(control["action_scale"])
    # humanoidverse_mjlab.py:433-438 -- action_rescale is True in the bfm_zero base config.
    per_joint_action_scale = [action_scale * effort_limits[i] / kp[i] for i in range(len(dof_names))]

    return {
        "checkpoint_run": str(DEFAULT_RUN),
        "robot_config": str(robot_cfg_path),
        "xml_path": str(_resolve_xml_path(str(cfg["xml_path"]), robot_cfg_path)),
        "dof_names": dof_names,
        "num_dof": len(dof_names),
        # --- actor ONNX signature (dumped, not inferred) ---
        "actor_input_keys": export_spec["actor_input_keys"],
        "actor_input_dims": export_spec["actor_input_dims"],
        "actor_input_dim": export_spec["actor_input_dim"],
        "z_dim": export_spec["z_dim"],
        "actor_obs_dim": export_spec["actor_obs_dim"],
        "output_action_dim": export_spec["output_action_dim"],
        # --- obs layout (humanoidverse_mjlab.py:926-970) ---
        "state_layout": [
            {"name": "dof_pos", "dim": len(dof_names), "scale": 1.0, "relative_to_default": True},
            {"name": "dof_vel", "dim": len(dof_names), "scale": 1.0},
            {"name": "projected_gravity", "dim": 3, "scale": 1.0},
            {"name": "base_ang_vel", "dim": 3, "scale": 0.25},
        ],
        "history_actor": {
            "steps": 4,
            "newest_first": True,
            "key_order": ["actions", "base_ang_vel", "dof_pos", "dof_vel", "projected_gravity"],
            "key_dims": {
                "actions": len(dof_names),
                "base_ang_vel": 3,
                "dof_pos": len(dof_names),
                "dof_vel": len(dof_names),
                "projected_gravity": 3,
            },
            "reset_value": 0.0,
        },
        "obs_scales": {"base_ang_vel": 0.25, "projected_gravity": 1.0, "dof_pos": 1.0, "dof_vel": 1.0, "actions": 1.0},
        "obs_normalizer_baked_into_onnx": True,
        # --- action -> PD target ---
        "normalize_action": True,
        "normalize_action_from": 1.0,
        "normalize_action_to": float(control["normalize_action_to"]),
        "action_clip_value": float(control["action_clip_value"]),
        "action_scale": action_scale,
        "action_rescale": True,
        "per_joint_action_scale": per_joint_action_scale,
        "default_joint_angles": [float(default_angles[n]) for n in dof_names],
        "kp": kp,
        "kd": kd,
        "effort_limit": effort_limits,
        "velocity_limit": velocity_limits,
        # --- rates ---
        "physics_fps": 200.0,
        "control_decimation": 4,
        "control_fps": 50.0,
        "init_root_pos": [float(x) for x in training["init_state"]["pos"]],
    }


def main(
    run_dir: Path = DEFAULT_RUN,
    output_dir: Path | None = None,
    robot_config: Path = DEFAULT_ROBOT_CONFIG,
    device: str = "cpu",
    opset: int = 13,
) -> None:
    run_dir = Path(run_dir).expanduser().resolve()
    checkpoint_dir = run_dir / "checkpoint"
    output_dir = Path(output_dir).expanduser().resolve() if output_dir else run_dir / "export_onnx"
    output_dir.mkdir(parents=True, exist_ok=True)

    model = load_model_from_checkpoint_dir(str(checkpoint_dir), device=device)
    model.eval()
    model.requires_grad_(False)

    export_spec = _infer_policy_onnx_export_spec(model, int(model.cfg.archi.z_dim))
    print("[INFO] export spec:", json.dumps(export_spec, indent=2, sort_keys=True))

    model_name = model.__class__.__name__
    policy_name = f"{model_name}.onnx"
    metadata = export_meta_policy_as_onnx(model, output_dir, policy_name, z_dim=int(model.cfg.archi.z_dim), opset_version=opset)
    (output_dir / f"{model_name}.meta.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")

    export_backward_encoder_from_model(model, output_dir / "backward_encoder.onnx", opset=opset)

    deploy_spec = build_deploy_spec(Path(robot_config).expanduser().resolve(), export_spec)
    deploy_spec["policy_onnx"] = policy_name
    deploy_spec["backward_encoder_onnx"] = "backward_encoder.onnx"
    (output_dir / "deploy_spec.json").write_text(json.dumps(deploy_spec, indent=2) + "\n")
    print(f"[INFO] Wrote deploy spec: {output_dir / 'deploy_spec.json'}")


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
