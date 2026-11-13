# 本地 LeRobot v3 训练

两份数据位于 `/home/djhpc/data/`：

| 配置 | 数据目录 | GPU | 全局 batch |
| --- | --- | --- | --- |
| `pi0_force_lora_huanggua` | `huanggua1103_25hz_3cam_lerobotv3` | 0,1 | 4 |
| `pi0_force_lora_board` | `board_25hz_3cam_lerobotv3_clean` | 2,3 | 4 |

黄瓜数据原先散放的 `data`、`images`、`meta`、`videos`、`visualizations`
已全部移入上面的黄瓜数据目录。两份数据是 25 Hz，分别有 41 / 62 个 episode，
15,392 / 20,067 帧。

## 输入和动作

适配器直接读取本地 v3 的 Parquet、episode 范围和视频分片，无需转换为 v2，
也无需升级环境中旧版的 LeRobot。视频使用 CPU TorchCodec 解码，解码器按 worker
缓存；每个 episode 的视频起始时间由 v3 元数据提供。

| 原始字段 | 模型输入 |
| --- | --- |
| `observation.joint`（7 维） | `state[:7]`，机器人关节状态 |
| `observation.tau_ext`（7 维） | `state[7:14]`，通过 7 维投影注入 MoE |
| `observation.images.front` | `base_0_rgb` |
| `observation.images.right_wrist` | `right_wrist_0_rgb` |
| `action.joint`（7 维绝对关节角） | 相对当前关节状态的 7 维增量，再补零至 32 维 |

虽然目录名包含 `3cam`，当前元数据和视频实际只有 front、right_wrist 两路。
`left_wrist_0_rgb` 补零且 mask 为 false。以后增加第三路时可设置
`--data.left-wrist-image-key observation.images.left_wrist`，同时重新计算对应统计。

动作 horizon 为 50，即 2 秒。动作序列只在当前 episode 内取样，末尾重复最后
一个动作；力矩只取当前帧。七个关节全部做 delta 转换，第七维不当作夹爪。
需要绝对关节动作时，可将配置的 `use_delta_joint_actions` 改为 false，并重算统计。
输出变换会把模型预测的增量还原为绝对关节角。

board 数据源的任务文本是 `cut_cucumber`，所以配置明确覆盖为 `Wipe the board.`；
黄瓜配置使用 `Peel the cucumber.`。

## 启动

```bash
conda activate forcevla
cd /home/djhpc/dp_training/dp_hirol_main/ForceVLA
```

在两个终端分别执行：

```bash
CUDA_VISIBLE_DEVICES=0,1 bash scripts/train_lerobot_v3.sh huanggua
```

```bash
CUDA_VISIBLE_DEVICES=2,3 bash scripts/train_lerobot_v3.sh board
```

每条命令只有一个 JAX 进程，使用两张卡做 FSDP，数据 batch 也在两卡间分片。
不使用 `torchrun`。配置是全局 batch 4，即每卡 2；EMA 和在线 WandB 已关闭，
训练 50,000 步，每 2,000 步保存，保留每 10,000 步的历史 checkpoint。

LoRA 冻结的是语言模型和动作专家的基础权重；SigLIP 视觉编码器和力感知
MoE 仍在全量训练。两张 24 GiB RTX 4090 使用全局 batch 8 时已有首步 OOM，
因此默认降至 4。参数和 Adam 状态的静态大小不能代表训练峰值显存。
当前四专家 MoE 的 token 分组要求全局 batch 为 4 的倍数，不能降到 2。

已用真实黄瓜数据、官方 π₀ base 权重及双卡 FSDP 验证 batch 4 连续两个训练
step，loss 为 6.31 / 5.59，未发生 OOM。JAX 记录的每卡活动分配峰值约
12.4 GiB，分配器内存池约 21.2 GiB；`nvidia-smi` 显示的占用包含内存池。
这是短程验证，未运行完整 50,000 步。

如果之前在第一个 step 就 OOM，原目录还没有可恢复的 checkpoint，可用新实验名重启：

```bash
CUDA_VISIBLE_DEVICES=0,1 bash scripts/train_lerobot_v3.sh huanggua --exp-name=huanggua_tau_ext_b4
CUDA_VISIBLE_DEVICES=2,3 bash scripts/train_lerobot_v3.sh board --exp-name=board_tau_ext_b4
```

脚本在统计缺失时自动运行 `compute_norm_stats.py`，v3 的统计计算跳过视频解码。
首次训练会自动下载官方 π₀ base 权重和 tokenizer，需要网络，缓存为两任务共享。
默认实验名的 checkpoint 分别写入 `checkpoints/pi0_force_lora_huanggua/huanggua_tau_ext/`
和 `checkpoints/pi0_force_lora_board/board_tau_ext/`；使用上面的重启命令时，末级目录
分别为 `huanggua_tau_ext_b4/` 和 `board_tau_ext_b4/`。已有 checkpoint 时，使用相同
实验名并追加 `--resume` 接着训练。

手动重算统计：

```bash
JAX_PLATFORMS=cpu python scripts/compute_norm_stats.py --config-name pi0_force_lora_huanggua
JAX_PLATFORMS=cpu python scripts/compute_norm_stats.py --config-name pi0_force_lora_board
```

修改数据路径、动作表示或 tau_ext 预处理后，需重新计算统计，不能沿用 wrench 的统计。
