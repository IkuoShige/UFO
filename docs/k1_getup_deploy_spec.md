# K1 UFO policy — deploy interface contract

Authoritative sim2sim / sim2real contract for running the UFO Behavior Foundation
Model (FB-CPR) on the Booster K1 22-DOF humanoid outside the training stack.

Checkpoint: `runs/ufo_fb_k1_5090_v2/checkpoint/` — `FBcprAuxModel`, z_dim 256, 192M steps.
Every claim below is traced to a file:line. Where the working assumptions going into
this work were wrong, that is called out explicitly.

Related files:

| Purpose | Path |
| --- | --- |
| Export policy + backward encoder + `deploy_spec.json` | `UFO/tools/export_k1_ufo_policy_onnx.py` |
| Build the latent z-bank | `UFO/tools/build_k1_z_bank.py` |
| Standalone MuJoCo sim2sim (no ROS2) | `UFO/tools/k1_ufo_sim2sim.py` |
| ONNX ≡ PyTorch equivalence test | `UFO/tests/test_k1_ufo_onnx_equivalence.py` |
| Deploy-runtime cross-check vs training code | `UFO/tests/test_k1_ufo_deploy_runtime.py` |
| Deploy constants (generated) | `booster_k1_locomotion/booster_k1_locomotion/k1_ufo_constants.py` |
| Deploy runtime (obs / action / z) | `booster_k1_locomotion/booster_k1_locomotion/ufo_policy_runtime.py` |
| ROS2 node | `booster_k1_locomotion/booster_k1_locomotion/ufo_policy_node.py` |
| Launch | `booster_k1_locomotion/launch/k1_ufo_sim.launch.py` |

---

## 0. Corrections to the going-in assumptions

Four assumptions that seemed safe turned out to be wrong. Any of them would have
produced a policy that runs without error and behaves badly.

| Assumption | Reality | Source |
| --- | --- | --- |
| `state` = `[base_ang_vel, projected_gravity, dof_pos, dof_vel]` | **`[dof_pos, dof_vel, projected_gravity, base_ang_vel]`** — joints first, IMU last | `humanoidverse/agents/envs/humanoidverse_mjlab.py:957` |
| `history_actor` is 4 steps × 72 grouped **by step** | Grouped **by key**, keys in **sorted** (alphabetical) order, and within each key **newest first** | `humanoidverse_mjlab.py:938-946`, `humanoidverse/envs/env_utils/history_handler.py:38-45` |
| `dof_pos` might be absolute | **Relative to the default pose**: `dof_pos - (default_dof_pos + default_dof_pos_offset)` | `humanoidverse_mjlab.py:928` |
| `action_scale` is the flat `0.5` from the config | `action_rescale: true`, so the effective per-joint scale is `0.5 · effort_limit_i / kp_i` — ranging **0.28 to 1.77**, a 6× spread | `humanoidverse_mjlab.py:433-438`, `configs/robots/k1_22dof.yaml` |

One more, unrelated to the actor: the training scene contains **two coincident
ground planes** (§7).

---

## 1. ONNX signature

Dumped from the real export spec (`_infer_policy_onnx_export_spec`,
`humanoidverse/utils/helpers.py:319`), not inferred:

```
actor_obs (batch, 616)  ->  action (batch, 22)
```

`actor_obs` is the concatenation of the actor input keys **in `actor_input_keys`
order**, followed by `z` at the tail (`helpers.py:378-392`):

| slice | key | dim |
| --- | --- | --- |
| `[0:50]` | `state` | 50 |
| `[50:72]` | `last_action` | 22 |
| `[72:360]` | `history_actor` | 288 |
| `[360:616]` | `z` | 256 |

`actor_input_keys = ["state", "last_action", "history_actor"]` comes verbatim from
`cfg.archi.actor.input_filter.key` and is **not** re-sorted (`helpers.py:244-260`).
`privileged_state` (343) is a critic/backward-map input only and is **not** an actor input.

* Output is `dist.mean` of a `TruncatedNormal` whose mean is `tanh(...)`, so
  **`action ∈ [-1, 1]`** (`humanoidverse/agents/nn_models.py:534`, `fb/model.py:128`).
* The running observation normalizer is **baked into the graph**
  (`model.act` → `actor` → `_normalize`, `fb/model.py:100-118`). The deploy node
  applies the env-side `obs_scales` and then **must not normalize again**.

Backward encoder (`backward_encoder.onnx`): `(state, privileged_state) -> z`.
Note `last_action` is **absent** from the graph — `torch.onnx.export` prunes it because
FBcprAux's backward map filters to `key=["state", "privileged_state"]`
(`checkpoint/model/config.json`). Fixed the resulting crash in
`humanoidverse/export/backward_encoder.py:_verify_backward_encoder_onnx`, which
previously fed all three names unconditionally.

---

## 2. `state` (50)

Built in `get_observation` (`humanoidverse_mjlab.py:957`):

```python
state = cat([dof_pos, dof_vel, projected_gravity, base_ang_vel])
```

| slice | field | dim | scale | notes |
| --- | --- | --- | --- | --- |
| `[0:22]` | `dof_pos` | 22 | 1.0 | **offset from default pose** |
| `[22:44]` | `dof_vel` | 22 | **1.0** | not 0.1 — see below |
| `[44:47]` | `projected_gravity` | 3 | 1.0 | gravity unit vector in trunk frame |
| `[47:50]` | `base_ang_vel` | 3 | **0.25** | trunk-frame angular velocity |

* **`dof_pos` is relative**: `dof_pos_rel = dof_pos - (default_dof_pos + default_dof_pos_offset)`
  (`humanoidverse_mjlab.py:928`). `default_dof_pos` is `robot.init_state.default_joint_angles`
  (`humanoidverse_mjlab.py:260-262, 666`). `default_dof_pos_offset` is a domain-randomization
  term that is forced to zero for this run — `cfg.domain_rand.randomize_default_dof_pos = False`
  (`humanoidverse_mjlab.py:341`, `_randomize_default_dof_pos_offset` at `:832`). **At deploy it is 0.**
* **`dof_vel` scale is 1.0**, from `obs_scales` in `humanoidverse/config/obs/bfm_zero_obs.yaml`.
  (The unrelated locomotion policy in `k1_constants.py` uses 0.1 — do not copy it.)
* **`base_ang_vel` scale is 0.25**, the only non-unity scale in the whole observation.
* `base_ang_vel = quat_rotate_inverse(base_quat, root_ang_vel_world)`
  (`humanoidverse_mjlab.py:850`) — i.e. **trunk-frame** angular velocity. An IMU gyro
  already reports this frame; feed it **unscaled** and let the runtime apply 0.25.
  Verified against MuJoCo directly (`k1_ufo_sim2sim.py::_assert_body_frame_ang_vel`):
  a free joint's `qvel[3:6]` is body-frame angular velocity and equals the rotated
  world value. The K1 MJCF's `gyro` sensor is on the `imu` site, which sits at the
  Trunk origin with no rotation, so IMU frame == Trunk frame.
* `projected_gravity = quat_rotate_inverse(base_quat, [0,0,-1])` (`humanoidverse_mjlab.py:851`).
  Formula: `humanoidverse/utils/torch_utils.py:281` with `w_last=True`.

Observation noise (`noise_scales` in `bfm_zero_obs.yaml`) is **training-only**:
`_apply_obs_scale_noise` zeroes it when `is_evaluating` (`humanoidverse_mjlab.py:782-791`).
Deploy adds no noise.

---

## 3. `last_action` (22)

`raw_obs["actions"] = self.actions` (`humanoidverse_mjlab.py:930`), where `self.actions`
is set in `step()` as `self.actions[:] = self._normalized_action(actions)`
(`humanoidverse_mjlab.py:1032`).

**It is the post-scaling, post-clipping action (range ±5), not the raw network output.**
Feed back `clip(a_net · 5.0, ±5.0)`, not `a_net`.

Timing: the vector env computes the observation *after* stepping physics
(`humanoidverse_mjlab.py:1246`), so `last_action` at tick *t* is the action that was
just applied during tick *t*'s physics. Zero on the first tick after reset
(`reset_idx` sets `self.actions[env_ids] = 0.0`, `:1124`).

---

## 4. `history_actor` (288)

```python
history_config = self.config.obs.obs_auxiliary["history_actor"]
for key in sorted(history_config.keys()):                       # SORTED
    t = self.history_handler.query(key)[:, :history_config[key]]
    history_tensors.append(t.reshape(t.shape[0], -1))
history_actor = cat(history_tensors, dim=1)
```
(`humanoidverse_mjlab.py:938-946`)

`sorted(["base_ang_vel","projected_gravity","dof_pos","dof_vel","actions"])`
⇒ **`["actions", "base_ang_vel", "dof_pos", "dof_vel", "projected_gravity"]`**, which is
*not* the order they appear in `bfm_zero_obs.yaml`.

| slice | key | layout |
| --- | --- | --- |
| `[0:88]` | `actions` | 4 × 22 |
| `[88:100]` | `base_ang_vel` | 4 × 3 |
| `[100:188]` | `dof_pos` | 4 × 22 |
| `[188:276]` | `dof_vel` | 4 × 22 |
| `[276:288]` | `projected_gravity` | 4 × 3 |

(offsets are relative to the start of the `history_actor` block, i.e. `actor_obs[72:360]`)

**Step order is newest first.** `HistoryHandler.add` does
`history[:, 1:] = history[:, :-1]; history[:, 0] = value`
(`humanoidverse/envs/env_utils/history_handler.py:38-42`), so index 0 is the most
recent entry. Each key's block reads `[t-1, t-2, t-3, t-4]`.

> There are two `HistoryHandler` classes in the repo. The env imports the one from
> `humanoidverse/envs/env_utils/` (`humanoidverse_mjlab.py:28`), **not** the
> `agents/envs/utils/` variant, which stores oldest-first and has a different
> `query()`. Getting this wrong reverses the time axis of 288 of the 360 inputs.

Other properties:
* **Values stored are already scaled** — the history is fed from `obs_data`, which has
  passed through `_apply_obs_scale_noise` (`humanoidverse_mjlab.py:929-936, 950-952`).
  `obs_scales["history_actor"] = 1.0`, so there is no second scaling.
* **Read before write**: the history is queried, then the current tick is appended
  (`:940` vs `:950`), so `history_actor` holds strictly past ticks.
* **Zero at reset**: `history_handler.reset()` zeroes the buffers
  (`history_handler.py:30-35`), called from `reset_idx` (`humanoidverse_mjlab.py:1126`).
  The deploy node must zero history and `last_action` whenever the policy is (re)activated.

Cross-checked numerically against the real `HistoryHandler` in
`tests/test_k1_ufo_deploy_runtime.py::test_history_actor_matches_training_history_handler`.

---

## 5. Action → PD target

```
a_net   = onnx(actor_obs)                                  # tanh mean, [-1, 1]
a       = clip(a_net * 5.0 / 1.0, -5.0, +5.0)              # _normalized_action
q_des_i = a_i * (0.5 * effort_limit_i / kp_i) + default_i  # JointPositionAction
```

* `_normalized_action` (`humanoidverse_mjlab.py:1018-1021`):
  `normalize_action: true`, `normalize_action_from: 1.0`, `normalize_action_to: 5.0`,
  `action_clip_value: 5.0`. The clip is a no-op given a tanh input, but keep it —
  `a` is what feeds `last_action` and the action history.
* Per-joint scale (`humanoidverse_mjlab.py:415-438` and `_action_target_scale` at `:265`):
  `action_scale = 0.5`, and because `action_rescale` is true it is multiplied by
  `effort_limit_i / kp_i`. Effective scales range **0.280** (ankles) to **1.772** (arms);
  knee is 0.331, giving a commandable knee range of `±5 × 0.331 + 0.63` ≈ `[-1.03, 2.29]` rad,
  which matches the intent recorded in `configs/robots/k1_22dof.yaml`.
* `+ default_i` comes from mjlab's `JointPositionActionCfg(use_default_offset=True)`
  (`humanoidverse_mjlab.py:493-499`, `mjlab/envs/mdp/actions/actions.py:218`), where the
  offset is `default_joint_pos` = `init_state.default_joint_angles`.
* `_mjlab_action_input` (`humanoidverse_mjlab.py:1023-1027`) adds
  `default_dof_pos_offset / action_target_scale`, which is **0** here (§2).

**A zero action commands exactly the default pose.** Useful as the safe idle target.

PD law, at the **physics** rate with the target held across decimation
(`mjlab/actuator/pd_actuator.py:compute`, `mjlab/entity/entity.py:844`):

```
tau = kp * (q_des - q) + kd * (0 - qd)
```

then the DC-motor clamp (`mjlab/actuator/dc_actuator.py:137-163`), with
`saturation_effort == effort_limit`:

```
v_corner = velocity_limit * (1 + effort_limit / saturation_effort)   # == 2 * velocity_limit
v        = clip(qd, ±v_corner)
tau      = clip(tau, max(sat*(-1 - v/vlim), -effort_limit),
                     min(sat*( 1 - v/vlim),  effort_limit))
```

---

## 6. Rates

| Quantity | Value | Source |
| --- | --- | --- |
| physics timestep | 0.005 s (200 Hz) | `simulator.config.sim.fps = 200` → `MujocoCfg(timestep=1/200)`, `humanoidverse_mjlab.py:583` |
| decimation | 4 | `simulator.config.sim.control_decimation`, `humanoidverse_mjlab.py:566` |
| control rate | **50 Hz** (0.02 s) | 200 / 4 |

The MJCF's own `<option timestep="0.001" impratio="10" .../>` is **discarded**:
`MjSpec.attach` does not propagate child `<option>` fields (mjlab warns about this,
`mjlab/scene/scene.py:232`) and `MujocoCfg.apply` overwrites them. Training ran with
mjlab's defaults: `implicitfast`, Newton, `impratio=1.0`, pyramidal cone,
`iterations=100`, `tolerance=1e-8`, `ls_iterations=50`, `ls_tolerance=0.01`.

---

## 7. Scene: two coincident ground planes

mjlab attaches the robot MJCF into a parent spec that already carries a
`TerrainEntityCfg(terrain_type="plane")` ground (`humanoidverse_mjlab.py:571`,
`mjlab/scene/scene.py:236,264`). `spec_fn` deletes only the actuators
(`humanoidverse_mjlab.py:396-401`), so `K1_22dof.xml`'s own `<geom name="ground">`
survives the attach. The compiled training model therefore contains **both**:

| geom | condim | friction | solref |
| --- | --- | --- | --- |
| `robot/ground` (from K1_22dof.xml) | 1 | `0.4 0.005 0.0001` | `0.001 1` (from the MJCF `<default>`) |
| `terrain` (from mjlab) | 3 | `1.0 0.005 0.0001` | MuJoCo default `0.02 1` |

Verified by replicating the attach in plain MuJoCo. MuJoCo takes the elementwise max
of the two geoms' friction and `max(condim)` per contact pair, so the effective foot
friction is ~1.0, but every foot contact is duplicated, which stiffens the ground.
`tools/k1_ufo_sim2sim.py --single-ground` A/B tests this; get-up succeeds either way,
so it is not load-bearing for this skill — but it is a latent training-scene bug and
should be fixed deliberately, not by accident.

`booster_k1_locomotion`'s `mujoco_sim_node.py` loads a single XML, so pointing it at
`K1_22dof.xml` gives the single-plane variant. `assets/rfc_assets/` is an
uninitialized git submodule (empty), so the default scene in `k1_sim.launch.py` does
not exist on this box; `k1_ufo_sim.launch.py` defaults to `K1_22dof.xml` instead,
which already has a ground plane.

---

## 8. Joint order

Verified **mechanically**, not by eye
(`tests/test_k1_ufo_deploy_runtime.py::test_joint_order_matches_mujoco_model`):

```
UFO control_joints.names  ==  MJCF hinge-joint order  ==  MJCF <actuator> order
                          ==  k1_constants.JOINT_NAMES (locomotion policy)
```

and the hinge joints occupy `qpos[7:29]` / `qvel[6:28]` contiguously in that order,
which is exactly how `mujoco_sim_node.py` indexes them (`qpos[7 + i]`, `qvel[6 + i]`)
and what the booster SDK's serial motor index assumes.

```
0  AAHead_yaw            8  Right_Elbow_Pitch     16 Right_Hip_Pitch
1  Head_pitch            9  Right_Elbow_Yaw       17 Right_Hip_Roll
2  ALeft_Shoulder_Pitch  10 Left_Hip_Pitch        18 Right_Hip_Yaw
3  Left_Shoulder_Roll    11 Left_Hip_Roll         19 Right_Knee_Pitch
4  Left_Elbow_Pitch      12 Left_Hip_Yaw          20 Right_Ankle_Pitch
5  Left_Elbow_Yaw        13 Left_Knee_Pitch       21 Right_Ankle_Roll
6  ARight_Shoulder_Pitch 14 Left_Ankle_Pitch
7  Right_Shoulder_Roll   15 Left_Ankle_Roll
```

**No index remapping is needed anywhere in the pipeline.**

---

## 9. Gains and default pose — do not mix with `k1_constants.py`

`k1_constants.py` describes an unrelated locomotion policy. Its joint *order* matches;
nothing else does. Using its gains with UFO's action targets would be violent.

| | UFO (`k1_ufo_constants.py`) | locomotion (`k1_constants.py`) |
| --- | --- | --- |
| kp legs | 30.2 / 21.4 / 17.8 / 60.4 / 35.7 | 200 / 200 / 200 / 200 / 50 |
| kp arms | 3.95 | 20 |
| kd legs | 3.6 / 2.56 / 2.13 / 4.81 / 4.26 | 5.0 / 5.0 / 5.0 / 5.0 / 3.0 |
| action scale | per-joint 0.280 – 1.772 | flat 0.25 |
| obs dim | 616 (with z) | 79 |
| default knee | 0.63 | 0.4 |
| default shoulder roll | ∓1.3 | ∓1.35 |

UFO gains come from `configs/robots/k1_22dof.yaml` (`kp = armature·(2πf)²`,
`kd = 2ζ·armature·2πf`; legs f=4 Hz ζ=1.5 with knee ζ=1.0, arms/head f=10 Hz ζ=2).
Default pose is the symmetrised median upright-frame pose of the retargeted LAFAN1 K1
dataset; nominal standing root height 0.53 m.

---

## 10. Latent `z` — runtime-swappable, never baked in

The BFM actor is one network for **all** skills (get-up → walk → run → shoot → …);
the skill is selected purely by the 256-dim `z`. `z` must therefore stay a runtime
input. Do not specialize the ONNX for one skill.

* `archi.norm_z = true`, so every `z` must satisfy `‖z‖ = sqrt(256) = 16`
  (`project_z`, `fb/model.py:124-127`). A mean or linear blend of latents is
  **off-manifold until re-projected** — `ZBank`/`ZController` re-project on load,
  on injection, and on every blend tick.
* z-bank format (`.npz`): `z/<skill>` → `(256,)`, `seq/<name>` → `(T, 256)`.
* Runtime control (`ufo_policy_node.py`):
  * `/ufo/skill` (`std_msgs/String`) — `"<skill>"`, `"seq:<name>"`, or `"<skill>!"` for
    an immediate unblended switch;
  * `/ufo/z` (`std_msgs/Float32MultiArray`) — inject a raw 256-dim latent;
  * z-bank **file watch** — the `.npz` is hot-reloaded on mtime change;
  * `/ufo/status` (`std_msgs/String`, JSON) — active skill, blending flag, available skills.
* Transitions are **slerped on the z-sphere** over `blend_steps` ticks (default 25 = 0.5 s)
  and re-projected each tick, so the action never jumps.
* Sequence mode replays a `seq/<name>` latent track open-loop at the control rate.

---

## 11. Verification status

**Numerically verified** (`tests/`, 16 tests, all passing):
* ONNX ≡ PyTorch `model.act(..., mean=True)`: max abs error **< 1e-4** over 100 random
  616-dim inputs, and per-sample at batch 1.
* Backward encoder ONNX ≡ `project_z(model.backward_map(...))`: max abs **1.4e-6**;
  output norms = 16.
* Deploy `history_actor` == the training `HistoryHandler`'s output, over 8 ticks.
* `projected_gravity` == `quat_rotate_inverse(..., w_last=True)`.
* Action pipeline, `dof_pos` relative-ness, z-manifold invariance through blends and
  sequences, joint-order chain, constants vs `deploy_spec.json`.

**Behaviourally verified** (`tools/k1_ufo_sim2sim.py`, plain MuJoCo, no ROS2):
from a fallen pose, with the interim `getup` latent, the robot **stands up in 0.64 s
and holds the stand** — final root height 0.530 m against a nominal 0.53 m,
uprightness 1.00, standing for 94 % of a 10 s rollout. Robust across face-up /
face-down starts (0.64 s / 0.78–0.82 s) and single / double ground plane.
Videos: `runs/ufo_fb_k1_5090_v2/export_onnx/getup_sim2sim*.mp4`.

**Not verified:**
* Nothing has been run under ROS2. `ufo_policy_node.py` byte-compiles and its runtime
  is exercised by the sim2sim loop and the unit tests, but the DDS path
  (`booster_robotics_sdk` LowState/LowCmd), the launch file and `colcon build` are
  untested here — see §12.
* No hardware. The gains and default pose in `configs/robots/k1_22dof.yaml` are marked
  `review_status: draft` and have never been validated on a real K1.
* `mujoco_sim_node.py` clamps torque at `TORQUE_LIMITS` but does **not** implement the
  DC-motor velocity derating that training used (§5). Small at low joint speeds,
  larger during fast get-up transients.
* `mujoco_sim_node.py` hardcodes the spawn height to `qpos[2] = 0.68`; UFO's nominal
  standing height is 0.53. The robot drops ~0.15 m on start.
* The interim `getup` latent is derived from `zs_17.pkl`
  (fallAndGetUp2_subject3 tracking latents, frames 100–140 averaged then re-projected).
  WS-A is searching for a better one; swap it into `tools/build_k1_z_bank.py`.

---

## 12. Reproducing

```bash
cd ~/ufo_ws/UFO
# Not in pyproject.toml: only the export/verify/sim2sim tooling needs these.
uv pip install onnxruntime pytest imageio-ffmpeg

uv run python tools/export_k1_ufo_policy_onnx.py     # ONNX + deploy_spec.json
uv run python tools/build_k1_z_bank.py               # z_bank.npz
uv run python -m pytest tests/test_k1_ufo_onnx_equivalence.py tests/test_k1_ufo_deploy_runtime.py -q
uv run python tools/k1_ufo_sim2sim.py --z-name getup --seconds 10 --video /tmp/getup.mp4
```

> ROS2 is sourced in the login shell and puts `/opt/ros/jazzy/lib/python3.12/site-packages`
> on `PYTHONPATH`, which breaks UFO's Python 3.10 venv (pytest autoloads
> `launch_testing`). Prefix UFO commands with
> `env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`.
> Headless rendering needs `MUJOCO_GL=egl` and `MUJOCO_EGL_DEVICE_ID=0`
> (set automatically by `k1_ufo_sim2sim.py`), plus `imageio-ffmpeg` since there is no
> system `ffmpeg`.

ROS2 side (untested):

```bash
# ROS2's Python 3.12 has numpy but NOT onnxruntime; the node cannot start without it.
python3 -m pip install --user onnxruntime

colcon build --packages-select booster_k1_locomotion
ros2 launch booster_k1_locomotion k1_ufo_sim.launch.py
ros2 topic pub -1 /ufo/enable std_msgs/String '{data: start}'
ros2 topic pub -1 /ufo/skill  std_msgs/String '{data: getup}'
ros2 topic echo /ufo/status
```
