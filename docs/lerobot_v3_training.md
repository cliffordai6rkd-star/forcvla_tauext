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
这里已经使用黑图补齐缺失视角；黑图经归一化和 SigLIP 编码后不等于零特征。
对现有 checkpoint，应保留相同填充和 mask，不能只在推理时复制其他视角或把 mask 改成 true。

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
MoE 会在路由前将 token 数补齐至专家数的倍数，路由后裁回原长度，因此也支持
batch 1 / 2。训练时全局 batch 仍须能被可见设备数整除；这里保留已验证的 batch 4。

已用真实黄瓜数据、官方 π₀ base 权重及双卡 FSDP 验证 batch 4 连续两个训练
step，loss 为 6.31 / 5.59，未发生 OOM。JAX 记录的每卡活动分配峰值约
12.4 GiB，分配器内存池约 21.2 GiB；`nvidia-smi` 显示的占用包含内存池。
上述显存数据来自短程验证。

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

## 单样本推理验证

推理在每个去噪步骤联合重算 prefix 和 suffix，使用与 `compute_loss` 相同的
注意力 mask、位置和力融合公式；图像与文本的初始编码只需计算一次。现有注意力
实现对完全被 mask 的 query 行仍产生输出，后续 MoE 也读取这些行，故对现有
checkpoint 不能只缓存 prefix 的 LLM 输出。联合重算保留训练时的计算语义，代价是
推理速度下降；此次没有更改训练的 mask 语义，也没有改变参数名称或形状。

标准输入有 817 个 MoE token。训练 batch 4 共 3268 个 token，原路由恰好是
每组 1 个 token、每专家容量 4。单样本仅补齐到 820 个再自动选组，会产生每组
5 个 token；若全选同一专家，其中 1 个会因容量不足被丢弃。现在推理固定每组
1 个 token（单样本 820 组），路由后裁回 817 个，沿用训练时的独立路由方式。
训练模式的分组算法保留，原 batch 4 的路由输出与旧路径逐元素一致。

回归测试覆盖缺失相机、文本 padding 及二者同时存在时的联合前向一致性，另用
全部选择同一专家的构造输入验证 token 不再因单样本分组而丢弃，并对比 batch
1 / 4 的路由输出。实机任务成功率仍需单独评估。

已对黄瓜、擦板的 `*_tau_ext_b4/49999` 各取一个真实样本，用相同初始 noise
比较单样本推理与 batch 4 联合前向参考（2 个去噪步骤）。真实输入为前视/右腕
mask true、左腕 false，文本 6/48 个 token 有效。两份还原后 7 关节动作的最大
误差分别为 0.00078 / 0.00152 rad，低于预先设定的 0.005 rad 容差；bfloat16
跨 batch 计算仍非逐位一致。两份 `Policy.infer` 均输出 `(50, 7)` 且数值有限。

## 推理服务

在 `ForceVLA` 目录启动擦板服务：

```bash
CUDA_VISIBLE_DEVICES=0 \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
/home/djhpc/miniconda3/envs/forcevla/bin/python scripts/serve_policy.py \
  --port 8000 \
  policy:checkpoint \
  --policy.config=pi0_force_lora_board \
  --policy.dir=checkpoints/pi0_force_lora_board/board_tau_ext_b4/49999
```

本地黄瓜/擦板配置默认在监听端口前完成一次离线预热，第一次 JAX 编译可能需要
几十秒。等到 `Policy warmup finished` 和 `server listening` 后再启动机器人客户端。
预热只使用合成观测，保留采样随机数状态，不连接机器人。诊断冷启动时可在
`policy:checkpoint` 前加 `--no-warmup`。

服务用单独线程顺序执行推理，网络事件循环继续处理 Ping/Pong、连接与健康检查，
因此首次编译或慢请求不会因占用网络线程而阻塞心跳。仍需将客户端首次请求的
等待时间设得足够长（如 120 秒）；请求等待超时和 WebSocket 心跳超时是不同的设置。
断线日志包含 code、reason、请求编号及发生阶段；每条请求记录排队和推理耗时。
`/healthz` 返回 HTTP 200 表示服务可连接，不代表实机策略效果已经验证。

客户端单次发送以下字段，不提前归一化，也不要增加 batch 维：

```python
observation = {
    "state": joint_angles_rad,          # (7,)，J1..J7
    "tau_ext": compensated_tau_ext_nm,  # (7,)，匹配训练的动力学补偿和因果滤波
    "image": front_rgb_uint8,           # HWC RGB
    "right_wrist_image": wrist_rgb_uint8,
    "prompt": "Wipe the board.",
}
```

本地配置缺省时会使用训练任务文本，显式提供的 prompt 优先。右腕必须用
`right_wrist_image`；服务会拒绝缺少右腕、额外传入 `wrist_image`（左腕槽）或
错误批次维的请求。左腕由适配器自动补黑且 mask false。浮点图像只接受有限的
`[0, 1]`，uint8 图像接受 `[0, 255]`；服务不做 BGR 到 RGB 的自动转换。

服务在归一化前补做 `sum(abs(tau_ext)) < 14 N·m` 时七维清零的 contact gate，
等于 14 时保留；已经做过 gate 的输入不受影响。客户端仍须提供正确的外力矩，
不能用未经补偿的电机力矩代替。NaN/Inf 状态、力矩和动作会被拒绝。
返回动作是 `(50, 7)` 的绝对关节角 rad，不能再次叠加当前关节角。

WebSocket metadata 会提供相机字段、任务文本、单位、门限和动作形状。消息上限
默认 32 MiB，建议客户端发送 224×224 的图像；需要调整时使用启动参数
`--max-message-size-mib`。代码更新后须重启原服务进程才能生效。

服务回归已使用两份现有 `49999` 权重、默认 10 步和真实数据观测验证：黄瓜关闭
预热时首请求约 44 秒，其间 173 次显式 Ping/Pong 全部成功，最大往返约 145 ms；
擦板启用预热后首条网络请求约 0.44 秒。两份服务均连续返回有限的 `(50, 7)` 动作。
这些是本机离线结果，未向机器人发送动作。
