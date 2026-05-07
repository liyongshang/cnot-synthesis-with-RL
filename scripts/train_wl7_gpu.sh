#!/usr/bin/env bash
# 一键在 GPU 上训练 walk_length=7（N=5 线形拓扑）BC 策略。
# 用法（在项目根目录）:
#   chmod +x scripts/train_wl7_gpu.sh
#   ./scripts/train_wl7_gpu.sh
# 可选环境变量:
#   CUDA_VISIBLE_DEVICES=0   # 默认 0
# 追加参数会传给 imitation_learning.py，例如更大 batch:
#   ./scripts/train_wl7_gpu.sh --batch-size 512

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

exec python imitation_learning.py \
  --dataset data/line_n5_wl7_post_pm.pt \
  --checkpoint checkpoints/imitation_policy_wl7.pt \
  --device cuda \
  --batch-size 256 \
  --seed 0 \
  "$@"
