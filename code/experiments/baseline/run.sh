#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

echo "=========================================="
echo "基线复现实验"
echo "工作目录: $(pwd)"
echo "=========================================="

if grep -q "YOUR_API_KEY" experiments/baseline/config.yaml; then
    echo "[警告] config.yaml 中 API Key 未配置，将以 dry-run 模式运行"
    DRY_RUN="--dry-run"
else
    DRY_RUN=""
fi

mkdir -p results/exp
mkdir -p experiments/baseline/logs
mkdir -p experiments/baseline/figs

echo "开始运行..."
python src/main.py \
    --config experiments/baseline/config.yaml \
    --split test \
    --output-dir results/exp \
    $DRY_RUN \
    "$@"

echo "=========================================="
echo "实验完成！结果保存在 results/exp/"
echo "=========================================="
