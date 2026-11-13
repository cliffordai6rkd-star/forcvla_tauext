#!/usr/bin/env bash
set -euo pipefail

if [[ $# -eq 0 || ${1:-} == --help || ${1:-} == -h ]]; then
    echo "Usage: bash scripts/train_lerobot_v3.sh {huanggua|board} [train.py options]"
    exit 0
fi

task=$1
shift
case "$task" in
    huanggua)
        config=pi0_force_lora_huanggua
        dataset=huanggua1103_25hz_3cam_lerobotv3
        default_gpus=0,1
        ;;
    board)
        config=pi0_force_lora_board
        dataset=board_25hz_3cam_lerobotv3_clean
        default_gpus=2,3
        ;;
    *)
        echo "Unknown task: $task (choose huanggua or board)" >&2
        exit 2
        ;;
esac

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
python_bin=${FORCEVLA_PYTHON:-/home/djhpc/miniconda3/envs/forcevla/bin/python}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-$default_gpus}
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}
export XLA_PYTHON_CLIENT_PREALLOCATE=${XLA_PYTHON_CLIENT_PREALLOCATE:-false}
export TF_CPP_MIN_LOG_LEVEL=${TF_CPP_MIN_LOG_LEVEL:-2}
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}

stats_path="assets/$config/$dataset/norm_stats.json"
if [[ ! -f "$stats_path" ]]; then
    echo "Computing torque/joint normalization statistics: $config"
    JAX_PLATFORMS=cpu "$python_bin" scripts/compute_norm_stats.py --config-name "$config"
fi

echo "Training $config on CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES (default global batch 4, FSDP 2; CLI options may override)"
exec "$python_bin" scripts/train.py "$config" --exp-name="${task}_tau_ext" "$@"
