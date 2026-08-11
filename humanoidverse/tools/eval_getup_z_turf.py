"""Paired fixed-friction evaluation for frozen K1 get-up latents.

Candidates share initial conditions in tiled environment blocks.  Their block
assignment rotates between batches, reducing bias from per-world observation
noise and other persistent simulator state.  No model or optimizer is updated.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from humanoidverse.agents.load_utils import load_model_from_checkpoint_dir
from humanoidverse.mjlab_inference_utils import checkpoint_load_device, resolve_inference_robot_config
from humanoidverse.tools.eval_getup import (
    DEFAULT_DATA_PATH,
    DEFAULT_ROBOT_CONFIG,
    PROJECT_ROOT,
    MotionZSource,
    PoseBankConfig,
    RolloutConfig,
    StandCriterion,
    build_eval_env,
    build_z_provider,
)
from humanoidverse.tools.opt_getup_z import EvalContext, evaluate_population

FIXED_FRICTION_OVERRIDES = (
    "domain_rand.push_robots=False",
    "domain_rand.randomize_link_mass=False",
    "domain_rand.randomize_base_com=False",
    "domain_rand.randomize_default_dof_pos=False",
)


def load_specs(path: Path, names: list[str] | None) -> list[dict[str, Any]]:
    specs = json.loads(path.read_text())
    if not isinstance(specs, list):
        raise ValueError(f"Candidate file must contain a JSON list: {path}")
    by_name = {str(spec["name"]): spec for spec in specs}
    selected = list(by_name) if not names else names
    missing = [name for name in selected if name not in by_name]
    if missing:
        raise ValueError(f"Unknown candidates {missing}; available={sorted(by_name)}")
    return [by_name[name] for name in selected]


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    model_folder = args.model_folder.expanduser().resolve()
    model = load_model_from_checkpoint_dir(
        model_folder / "checkpoint", device=checkpoint_load_device(args.device)
    )
    model.to(args.device)
    model.eval()
    specs = load_specs(args.candidates, args.names)
    if not specs:
        raise ValueError(f"No candidates selected from {args.candidates}")
    names = [str(spec["name"]) for spec in specs]
    num_envs = len(specs) * args.conditions
    data_path = (PROJECT_ROOT / DEFAULT_DATA_PATH).resolve()
    robot_config = resolve_inference_robot_config(PROJECT_ROOT / DEFAULT_ROBOT_CONFIG, None)
    criterion = StandCriterion(hold_s=args.stand_hold_s)
    pose_cfg = PoseBankConfig(
        bucket="ood",
        ood_yaws=args.ood_yaws,
        ood_drop_height=args.ood_drop_height,
        ood_joint_noise=args.ood_joint_noise,
    )
    roll_cfg = RolloutConfig(
        settle_steps=args.settle_steps,
        episode_steps=args.episode_steps,
        action_latency_max=args.action_latency_max,
        obs_latency_max=args.obs_latency_max,
        record_qpos=False,
    )
    payload: dict[str, Any] = {
        "model_folder": str(model_folder),
        "candidates": names,
        "conditions_per_candidate": args.conditions,
        "batches": args.batches,
        "episodes_per_candidate": args.conditions * args.batches,
        "seed": args.seed,
        "frictions": args.frictions,
        "stand_criterion": asdict(criterion),
        "pose_config": asdict(pose_cfg),
        "rollout_config": asdict(roll_cfg),
        "disabled_domain_randomization": list(FIXED_FRICTION_OVERRIDES),
        "results": {},
    }

    for friction_index, friction in enumerate(args.frictions):
        overrides = [f"domain_rand.friction_range=[{friction},{friction}]", *FIXED_FRICTION_OVERRIDES]
        wrapped_env, core, _env_cfg, use_root_height_obs = build_eval_env(
            model_folder=model_folder,
            data_path=data_path,
            robot_config=robot_config,
            device=args.device,
            num_envs=num_envs,
            disable_dr=False,
            disable_obs_noise=False,
            seed=args.seed + friction_index,
            hydra_overrides=overrides,
        )
        motion_src = MotionZSource(
            core,
            model,
            use_root_height_obs=bool(use_root_height_obs),
            device=args.device,
        )
        z_pop = torch.stack(
            [
                build_z_provider(spec, model=model, motion_src=motion_src, device=args.device)
                .representative_z()
                .to(args.device)
                for spec in specs
            ]
        )
        ctx = EvalContext(
            wrapped_env=wrapped_env,
            core=core,
            model=model,
            crit=criterion,
            effort_limits=core.torque_limits.cpu().numpy(),
            checker=None,
            use_root_height_obs=bool(use_root_height_obs),
        )
        results = evaluate_population(
            ctx,
            z_pop,
            pose_cfg=pose_cfg,
            roll_cfg=roll_cfg,
            seed_key=[args.seed, friction_index, 7919],
            batches=args.batches,
            self_collision=False,
        )
        payload["results"][str(friction)] = dict(zip(names, results))
        print(f"\n[FRICTION {friction:.3f}] {args.conditions * args.batches} episodes/candidate")
        for name, result in sorted(
            zip(names, results), key=lambda item: (-item[1]["success_upright"], -item[1]["held_upright"])
        ):
            print(
                f"  {name:34s} upright={result['success_upright']:.3f} "
                f"strict={result['success']:.3f} rose={result['rose']:.3f} "
                f"held={result['held_upright']:.3f} fell_back={result['fell_back']:.3f}"
            )
        wrapped_env.close()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-folder", type=Path, default=PROJECT_ROOT / "runs/ufo_fb_k1_5090_v2")
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--names", nargs="*", default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--frictions", type=float, nargs="+", default=(0.05, 0.1, 0.2, 0.5, 1.0))
    parser.add_argument("--conditions", type=int, default=16)
    parser.add_argument("--batches", type=int, default=4)
    parser.add_argument("--seed", type=int, default=97)
    parser.add_argument("--settle-steps", type=int, default=50)
    parser.add_argument("--episode-steps", type=int, default=250)
    parser.add_argument("--stand-hold-s", type=float, default=2.0)
    parser.add_argument("--action-latency-max", type=int, default=2)
    parser.add_argument("--obs-latency-max", type=int, default=1)
    parser.add_argument("--ood-yaws", type=int, default=12)
    parser.add_argument("--ood-drop-height", type=float, default=0.35)
    parser.add_argument("--ood-joint-noise", type=float, default=0.1)
    args = parser.parse_args()
    if args.conditions <= 0 or args.batches <= 0:
        parser.error("--conditions and --batches must be positive")
    if any(value < 0.0 for value in args.frictions):
        parser.error("--frictions values must be non-negative")
    return args


def main() -> None:
    evaluate(parse_args())


if __name__ == "__main__":
    main()
