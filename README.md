# ForceVLA: Enhancing VLA Models with a Force-aware MoE for Contact-rich Manipulation
## 
ForceVLA is based on the [π₀ model](https://www.physicalintelligence.company/blog/pi0), a flow-based diffusion vision-language-action model (VLA)； Both training and inference are based on π₀.

## Requirements

To run the models in this repository, you will need an NVIDIA GPU with at least the following specifications. These estimations assume a single GPU, but you can also use multiple GPUs with model parallelism to reduce per-GPU memory requirements by configuring `fsdp_devices` in the training config. Please also note that the current training script does not yet support multi-node training.

| Mode               | Memory Required | Example GPU        |
| ------------------ | --------------- | ------------------ |
| Inference          | > 8 GB          | RTX 4090           |
| Fine-Tuning (LoRA) | > 22.5 GB       | RTX 4090           |
| Fine-Tuning (Full) | > 70 GB         | A100 (80GB) / H100 |

The repo has been tested with Ubuntu 22.04, we do not currently support other operating systems.


## dataset
https://huggingface.co/datasets/qiaojunyu/ForceVLA-real-data

## Installation

When cloning this repo, make sure to update submodules:

```bash
conda create -n forcevla python=3.11 -y

```

```bash
python -m pip install --upgrade pip setuptools wheel
conda install -c nvidia cuda-toolkit=12.8
```

```bash
cd lerobot/
conda install ffmpeg=7.1.1 -c conda-forge
pip install -e .
```

```bash
cd ./openpi
pip install -e .
```

```bash
cd dlimp/
pip install -e .
```

```bash
cd packages/
cd openpi-client/
pip install -e .
```

```bash
cd flaxformer/
pip install -e .
```
## train policy
Local LeRobot v3 joint/torque datasets: see [dual-GPU training instructions](docs/lerobot_v3_training.md).

The guidance input is now seven-dimensional external joint torque, read from
`observation.tau_ext` (override with `--data.tau-ext-key` if needed). The first
seven channels of legacy `observation.state` remain `xyz + rpy + gripper`;
the local v3 configs instead read all seven joints from `observation.joint`. The input
transform builds `[proprioception (7), tau_ext (7), padding]`; legacy wrench
channels appended to the source state are discarded. Inference inputs provide
`state` and `tau_ext` separately, as in `make_forcevla_example()`.

Recompute normalization statistics on the torque dataset before training; old
wrench statistics and six-input ForceVLA guidance weights are incompatible.
The standard pi0 base checkpoint can still initialize training.

```bash
export HF_LEROBOT_HOME="xxxxxx"
python scripts/compute_norm_stats.py --config-name forcevla_lora 
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9  python scripts/train.py forcevla_lora --exp-name=my_experiment --overwrite  --batch_size 32 --save_interval 2000 --keep_period 10000
```
