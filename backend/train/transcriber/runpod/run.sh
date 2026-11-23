#!/usr/bin/env bash
# 학습. 먼저 짧은 시험(SMOKE=1)으로 한 걸음 시간을 보고, 본 학습은 MAX_MIN(분)이 지나면 저장하고 멈춘다.
#   SMOKE=1 bash run.sh
#   MAX_MIN=480 bash run.sh
set -euo pipefail
W=/workspace
cd $W/cowrite/backend/train/transcriber
mkdir -p $W/runs
COMMON="--base ../../app/midi_io/basic_pitch/nmp_cowrite.onnx --banks $W/banks --slakh $W/slakh --surge $W/surge_clean \
  --slakh_share 0.5 --surge_share 0.3 --batch ${BATCH:-64} --workers ${WORKERS:-$(( $(nproc) - 2 ))} --lr ${LR:-3e-4}"
if [ "${SMOKE:-0}" = 1 ]; then
  python big.py $COMMON --out $W/runs/smoke --steps 200 --save_every 100
else
  python big.py $COMMON --out $W/runs/${RUN:-r1} --steps ${STEPS:-50000} --save_every ${SAVE:-1000} --max_minutes ${MAX_MIN:-480} \
    2>&1 | tee -a $W/runs/${RUN:-r1}.log
fi
