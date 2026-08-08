# Booster K1 起き上がりポリシー — 調査・計画

## 0. 最終目的（サッカーエージェント）

起き上がりは単体のゴールではなく、**サッカーエージェント**のスキル #1。
習得順: 起き上がり → 歩行 → 走行 → instep shoot → inside kick → loop shoot → goal saving (keeper)。

UFO の BFM を基盤に選ぶ理由は、**1つの actor を共有し、スキルは潜在変数 z で切り替える**ことで
動作間の遷移が滑らかになるため。この性質が本プロジェクトの中核資産であり、
**設計判断はすべてこの性質を壊さない方を選ぶ**。

### スキルの射程（現時点の証拠つき）

188.8M step の tracking eval（`humanoidverse_tracking_eval.csv`）より:

| スキル | 状態 | 根拠 |
| --- | --- | --- |
| 起き上がり | **射程内** | fallAndGetUp 0.986 / ground 0.997 / pushAndFall 0.990 |
| 歩行 | **射程内** | walk 1.000 |
| 走行 | **射程内** | run 0.992（sprint 0.972、hip-pitch 速度制限が上限） |
| キック様動作 | **要確認** | fight1 0.998 / fightAndSports 0.998 と追従は良好。ただし当該クリップに実際に蹴り動作が含まれるかは未検証 |
| ボール接触スキル | **射程外** | 下記の制約を参照 |

### 射程外である理由（正直に）

- actor の入力は 360 次元に固定で、**ボール状態のチャネルが無い**。
  env 側にも object / ball のサポートが無い（scene は plane + robot のみ）。
- したがって **どんな z を見つけても、凍結した BFM はボールを認識できない**。
  完璧な「キック」z が得られても、それは開ループのキックであり、
  ボール相対のタイミングや狙いは z 探索だけでは到達できない。

### 想定アーキテクチャ（将来）

    ボール/ゲーム認識の高レベル方策  ──(低頻度で z を出力)──▶  凍結 BFM actor (50Hz)  ──▶  action(22)

学習時 `--update-z-every-step 100`（50Hz で 2 秒相当）で z を更新していたこと、
および毎フレーム z のリプレイも機能することから、
高レベル側は 0.5〜2 秒周期（あるいは z を滑らかに補間）で in-distribution に収まる。

実現に必要なもの（**今回は着手しない。名前をつけておくだけ**）:
1. env へのボールエンティティ追加（mjlab の `EntityCfg` で可能だが env コード変更が必要）
2. z 空間上の小さな RL ループ（UFO には該当機構が無い — 新規実装）
3. キック/ダイブの事前分布を厚くするなら、サッカー系 mocap の K1 リターゲットと BFM の継続学習

keeper のダイブは pushAndFall / 起き上がりと相性が良い（倒れる→受け身→起き上がり）。

---

対象チェックポイント: `runs/ufo_fb_k1_5090_v2/checkpoint/`（FB-CPR / `FBcprAuxModel`, z_dim=256, 192M env steps）
sim2sim 先: `../booster_k1_locomotion`

## 0.5 「だんだん上手くなる」仕組み（凍結BFMの天井を破る）

凍結 BFM は LAFAN1 の事前分布が性能の天井になる。サッカー上級者を目指すなら
**レパートリー自体が拡張する仕組み**が要る。UFO のコード構造を確認した結果、
最上位の手段が**アーキテクチャ変更なしで実現できる**ことが分かった。

### 上達の梯子（安い順）

| 層 | 何が良くなるか | UFO への変更 | 忘却リスク |
| --- | --- | --- | --- |
| L1 z 探索（CEM/RL over z） | 既存レパートリーからの**選択**のみ。能力は上がらない | なし | なし |
| L2 階層RL（凍結BFM + z を出す高レベル） | **タスク遂行**が大きく向上（ボール認識・タイミング・狙い） | 高レベル学習ループ（新規）+ ボール entity | なし |
| L3 残差ポリシー（Δa を学習、BFM 凍結） | **運動能力そのもの**が向上。z ごとにゲート可能 | 残差ネット + 学習ループ | 低（ベース無傷） |
| L4 正則化付き fine-tune | 運動能力が向上 | resume + タスク報酬 | 中〜高（他 z の劣化を要監視） |
| **L5 自己生成データで BFM 継続学習** | **レパートリー自体が拡張＝真の上達** | **データ変換 + resume のみ** | 低 |

### L5 が成立する根拠（コード実測）

- FB-CPR は `replay_buffer["expert_slicer"]`（事前分布＝モーションデータ）と
  `replay_buffer["train"]`（オンライン RL）を**分離して持つ**（`agents/fb_cpr/agent.py:171-213`）。
  discriminator が z 条件付きで expert と policy を判別する構造（CPR）。
- expert buffer はモーションライブラリから構築される
  （`load_expert_trajectories_from_motion_lib`, `training/workspace.py:556-563`）。
- 学習の再開は `work_dir/checkpoint/train_status.json` があれば**自動**
  （`create_agent_or_load_checkpoint`, `training/workspace.py:203`）。

したがって自己改善ループは:

    現ポリシーでタスク報酬つきロールアウト
      → 成功軌道を RobotState CSV 化してデータセットに追加
      → 192M チェックポイントから継続学習
      → BFM の事前分布に「上手くなった動き」が入る
      → 次ラウンドはそこから始まる

これは expert iteration / 自己模倣を BFM に適用した形で、
**「練習して型が身につき、その上でより高度な戦術が乗る」**という人間の上達と同じ構造。

### 副次的な利点

自己生成軌道は **K1 自身が物理的に実行した**動きなので、
リターゲット LAFAN1 が抱える問題（接地貫通 4〜11cm、hip-pitch 速度飽和 — `docs/vastai_k1.md`）が原理的に無い。
つまり自己生成データは事前分布を**増やすだけでなく綺麗にする**。

### L2 と L5 は競合せず、フライホイールを組む

L2（高レベルが今のレパートリーで狙い・タイミングを学ぶ）→ その成功軌道を L5 で BFM に還元
→ 次ラウンドでは「良いキック」が BFM のネイティブな挙動になり L2 の仕事が楽になる → 繰り返し。

**今回のスコープでは着手しない。** ただし WS-A の評価ハーネスには
成功エピソードを RobotState CSV で書き出す機能を入れてある（このループのデータ源になる）。

---

## 0.6 locomotion policy への引き継ぎ（実測で判明した主因）

起き上がり後に既存 locomotion policy（`policy_180843_19999.onnx`, obs 49 → action 12,
`booster_k1_locomotion` の `feat/dual_walk` が現行）へ渡す。
UFO 側と locomotion 側の定数を突き合わせた結果、**主要な障害は姿勢差ではなく PD ゲインの不連続**だった。

### 既定姿勢の差は小さい（脚 最大 0.13 rad）

| 関節 | UFO 既定 | loco 既定 | Δ |
| --- | --- | --- | --- |
| Hip_Pitch | -0.33 | -0.26 | +0.07 |
| Hip_Roll | ±0.13 | 0.00 | 0.13 |
| Hip_Yaw | ±0.11 | 0.00 | 0.11 |
| Knee_Pitch | 0.63 | 0.52 | -0.11 |
| Ankle_Pitch | -0.21 | -0.26 | -0.05 |
| Ankle_Roll | ±0.08 | 0.00 | 0.08 |

### ゲインの差は大きい（脚 1.4〜11.2 倍）

| 関節 | kp UFO | kp loco | 倍率 |
| --- | --- | --- | --- |
| Hip_Yaw | 17.8 | 200 | **11.2×** |
| Hip_Roll | 21.4 | 200 | **9.4×** |
| Hip_Pitch | 30.2 | 200 | **6.6×** |
| Knee_Pitch | 60.4 | 200 | 3.3× |
| Ankle_Pitch/Roll | 35.7 | 50 | 1.4× |
| 腕 | 3.95 | 30 | 7.6× |
| 頭 | 7.9 | 6 | 0.76× |

切替の瞬間、同じ関節誤差が最大 11 倍のトルクを生む。**瞬時切替は不可**。

### 本当の姿勢ギャップは BFM の挙動側にある

BFM が起き上がり後に収束する実測姿勢は **膝 0.15 rad**（ほぼ直脚）で、
これは loco 既定 0.52 からも、UFO 自身の既定 0.63 からも遠い。
つまり config 同士の不一致ではなく、**BFM がそういう立ち方をする**という話。
ここが z スケジューリング / PD ブレンドで詰める対象。

### 引き継ぎ手順（順序が重要）

1. UFO ノードで起き上がり（`getup` z、0.64 s）
2. **UFO の低ゲインのまま**、脚を loco 既定姿勢へ寄せる（z 二相化＋残差 PD ブレンド）
3. 姿勢が寄ってから **ゲインを 0.3〜0.5 s かけてランプ**（UFO → loco）
4. LowCmd の発行権を locomotion ノードへ移す

姿勢が合わないままゲインを上げると強く引き合うので、2 → 3 の順序は逆にできない。

### 実装上の制約

- **LowCmd を発行するノードは常に 1 つ**。UFO ノードと locomotion ノードが同時に
  DDS へ書くと競合する。arbiter か明示的な enable/disable が必要。
- UFO 用の launch は locomotion 側の `k1_getup_sim.launch.py` に寄せず独立させる
  （obs が 616 次元で全く別物のため）。`feat/ufo-policy` の `launch/k1_ufo_sim.launch.py`。

---

## 1. 調査結果（Survey）

### 1.1 BFM は既に起き上がりを持っている

v2 ランは `env.config.lie_down_init=True` / `lie_down_init_prob=0.3` で学習済み。
`humanoidverse_tracking_eval.csv` を最終チェックポイント（188.8M step）で絞り込むと:

| クリップ群 | クリップ数 | 平均 proximity | 最悪 |
| --- | --- | --- | --- |
| fallAndGetUp\* | 96 | 0.986 | 0.861 |
| ground\* | 86 | 0.997 | 0.956 |
| pushAndFall\* | 34 | 0.990 | 0.867 |
| walk\* | 295 | 1.000 | 0.988 |
| 全体 | 1692 | 0.996 | — |

起き上がり系は追従できている。したがって課題は「起き上がりを学習させる」ことではなく、
**参照モーション無しで閉ループに起き上がる潜在変数 z を BFM から取り出すこと**。

### 1.2 起き上がり z が既にディスク上にある

`k1_lafan1_full_ufo.pkl`（77 モーション）のインデックス対応:
`16=fallAndGetUp2_subject2`, `17=fallAndGetUp2_subject3`, `27=ground2_subject2`,
`55=pushAndFall1_subject4`, `65=walk1_subject1`。

`runs/ufo_fb_k1_5090_v2/tracking_inference*/zs_{16,17,27,55}.pkl` は
これら起き上がりクリップの毎フレーム z 列そのもの。追加計算ゼロで使える最初の候補。

### 1.3 デプロイ側インタフェース

- actor obs = `state`(50) + `last_action`(22) + `history_actor`(288) = 360、末尾に z(256) を連結して **616**、出力 action 22。
- `privileged_state`(343) は critic 専用でデプロイ不要。
- obs 正規化器は ONNX に焼き込まれる（`model.act` → `actor` → `_normalize`）。
  一方 `obs_scales`（`base_ang_vel: 0.25`）は env 側適用 → ノードが再現する必要がある。
- `export_meta_policy_as_onnx`（`humanoidverse/utils/helpers.py:340`）と
  backward encoder エクスポートの両方が既に存在。

### 1.4 sim2sim 先の状態

`booster_k1_locomotion` は MuJoCo sim ノード ←DDS→ ONNX ポリシーノードを Python / C++ 両実装で持つ。
ただし既存 `k1_constants.py` は**別物のロコモーションポリシー用**（PD 200/50、`ACTION_SCALE=0.25`、
obs 79 次元に gait phase と cmd_vel）。UFO ポリシーには一切流用できない。
`assets/rfc_assets` submodule は未初期化だが `xml_path` は ROS パラメータなので、
学習に使った `booster_assets/robots/K1/K1_22dof.xml` をそのまま指せる（sim2sim gap を最小化できる）。
joint 順は UFO の `control_joints.names` と `k1_constants.JOINT_NAMES` が一致して見えるが要検証。

## 2. アプローチ

安い候補から順に測る。凝った手法は、安い手が足りないと**データで示されてから**。

- **Phase 0/1**: 起き上がり評価ハーネスを作り、z 候補を比較（これが本丸の成果物）
- **Phase 2**: 固定 z で速度・ロバスト性が足りない場合のみ着手（条件付き）。
  優先順は **(a) z スケジュール / 二相 z → (b) z インタフェースを保ったまま BFM へ正則化付きファインチューン**。
  ファインチューンした場合は **他スキルの z（walk/run）での性能劣化を必ず確認**する
  — 見えないところで多スキル計画が壊れる。
  **固定 z への蒸留は既定案から外す**（共有潜在空間という中核資産を捨てる方向のため）。
  専用の転倒リフレックスなど、最後の手段としてのみ残す。
- **Phase 3**: sim2sim（z に依存しないので並行実行）
- **Phase 4**: ロバスト性スイープと報告

### z 候補（コスト順）

1. 既存 `zs_*.pkl` の起き上がり区間 — 平均 z（定数）と時系列リプレイ（開ループ）
2. `reward_inference` の `move-ego-0-0`（非G1ガードを通る、コード変更不要）
3. K1 goal JSON からの goal-z
4. 上記が不足なら z 球面上の CEM 探索

定数 z は閉ループで頑健だが遅い可能性、時系列リプレイは自然で速いが滑ると時刻同期がずれる。
このトレードオフをハーネスで定量化する。

## 3. 作業分割

| WS | 担当 | 内容 | 依存 |
| --- | --- | --- | --- |
| A | Opus 5 | 起き上がり評価ハーネス + z 候補比較（ローカル3090） | なし |
| B | Opus 5 | obs/action 規約の確定 + ONNX エクスポート + sim2sim ノード | なし（z は後差し） |
| C | Sonnet 5 | K1 goal JSON 生成 + goal-z 出力 | なし（成果物を A が消費） |

### 評価ハーネスの設計要点

- 転倒姿勢バンクは **in-distribution**（学習時 `lie_down_init` を厳密再現: root z=0.5、x軸±90°回転）と
  **OOD**（仰向け/うつ伏せ/横向き × yaw 掃引）を分離して報告する。平均してはいけない。
- 指標: 成功率、立ち上がり時間、立位保持安定性、トルク飽和、望ましくない接触。
- DR スイープ: 摩擦、質量、外乱 push、観測/動作遅延。

## 4. 実行方針の決定事項

- Phase 0/1 のロールアウトは**ローカル RTX 3090**で実行（推論規模）。共有GPUのため実行前に `nvidia-smi` 確認。
  フル学習ランは明示許可なしに起動しない。
- Phase 2 は**条件付き** — Phase 1 のハーネス結果が目標未達のときのみ着手。固定 z への蒸留は既定案ではない。
- z は**実行時に差し替え可能**であること（デプロイノードの必須要件）。z-bank（`skill_name -> z`）形式で
  スキルライブラリを育てる。`runs/getup_eval/z_bank.pt` が最初のエントリ。
- 評価ハーネスは使い捨てにしない — 歩行・走行が同じパイプラインを通る。
