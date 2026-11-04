#!/bin/bash
# TRELLIS.2 red_brick 全流程（mac mini 一体化）：
#   1) 本机 ComfyUI 产裁切图（BiRefNet + ImageCropToMask，与 GPU 基准同配方）
#   2) 杀掉 ComfyUI 释放 16GB 统一内存
#   caffeinate 防休眠 + 自构建驱动 fp32/MPS 跑 shape 管线出 GLB
# 用法: bash run_redbrick.sh [--dtype fp32|bf16|fp16] [--seed N] [额外 driver 参数...]
set -uo pipefail

PY=/Library/Frameworks/Python.framework/Versions/3.12/bin/python3
BASE="$HOME/ComfyUI"
DIR="$(cd "$(dirname "$0")" && pwd)"
OUT="$DIR/out"
mkdir -p "$OUT"

DTYPE=fp32
SEED=12345
while [ $# -gt 0 ]; do
  case "$1" in
    --dtype) DTYPE="$2"; shift 2;;
    --seed) SEED="$2"; shift 2;;
    *) break;;
  esac
done

echo "== [1/3] ComfyUI 可用性检查 =="
if ! curl -s --max-time 4 http://127.0.0.1:8188/system_stats >/dev/null 2>&1; then
  echo "ComfyUI 未运行，headless 启动…"
  export PYTORCH_ENABLE_MPS_FALLBACK=1
  cd "$BASE" || exit 1
  nohup $PY main.py --listen 127.0.0.1 --port 8188 --lowvram --preview-method none > "$HOME/comfyui-run.log" 2>&1 &
  for _ in $(seq 1 40); do
    sleep 3
    curl -s --max-time 4 http://127.0.0.1:8188/system_stats >/dev/null 2>&1 && break
  done
  cd "$DIR"
fi

echo "== [2/3] 产裁切图 =="
CROP=$($PY "$DIR/crop_redbrick.py" | grep -E '^CROP_PATH=' | head -1 | cut -d= -f2-)
[ -n "$CROP" ] && [ -f "$CROP" ] || { echo "FATAL: 裁切图未产出"; exit 1; }
ls -l "$CROP"

echo "== [3/3] 释放内存 + 驱动全流程 =="
pkill -f "main.py --listen" 2>/dev/null || true
sleep 6
pgrep -fl "main.py --listen" && echo "WARN: ComfyUI 仍在运行" || echo "ComfyUI 已停"

export PYTORCH_ENABLE_MPS_FALLBACK=1
# 注意：不设 PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0（部署文档 §4.2 整机 stall 前科）
cd "$DIR" || exit 1
caffeinate -dim -w $$ >/dev/null 2>&1 &
caffeinate_pid=$!
$PY "$DIR/trellis2_mac_driver.py" \
  --image "$CROP" \
  --out "$OUT/red_brick_${DTYPE}.glb" \
  --device mps --dtype "$DTYPE" --seed "$SEED" \
  --struct-dtype bf16 --struct-attention sub_quad \
  --tex-dtype bf16 --tex-size 1024 \
  --vae-device cpu \
  --dump "$OUT/dump" \
  "$@" 2>&1 | tee "$OUT/driver.log"
rc=${PIPESTATUS[0]}
kill "$caffeinate_pid" 2>/dev/null
ls -l "$OUT"/*.glb 2>/dev/null
exit $rc
