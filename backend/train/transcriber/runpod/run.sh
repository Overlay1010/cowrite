#!/usr/bin/env bash
# 학습. 먼저 짧은 시험(SMOKE=1)으로 한 걸음 시간을 보고, 본 학습은 MAX_MIN(분)이 지나면 저장하고 멈춘다.
#   SMOKE=1 bash run.sh
#   MAX_MIN=480 bash run.sh
# 데이터는 있는 것만 쓴다: /workspace/slakh(변환 끝난 Slakh) · /workspace/surge_clean(Surge 루프), 없으면 합성 루프.
set -euo pipefail
W=/workspace
cd $W/cowrite/backend/train/transcriber
mkdir -p $W/runs
DATA=""
[ -f $W/slakh/.ready ] && DATA="$DATA --slakh $W/slakh --slakh_share 0.5"
ls $W/surge_clean/*.npz >/dev/null 2>&1 && DATA="$DATA --surge $W/surge_clean --surge_share 0.3"
echo "데이터:${DATA:- 합성 루프만}"
COMMON="--base ../../app/midi_io/basic_pitch/nmp_cowrite.onnx --banks $W/banks $DATA \
  --batch ${BATCH:-64} --workers ${WORKERS:-$(( $(python -c "import big; print(big.cpu_count())") - 1 ))} --lr ${LR:-3e-4}"
if [ "${SMOKE:-0}" = 1 ]; then
  python big.py $COMMON --out $W/runs/smoke --steps 200 --save_every 100 2>&1 | grep -v Warning | tee $W/runs/smoke.log
else
  python big.py $COMMON --out $W/runs/${RUN:-r1} --steps ${STEPS:-50000} --save_every ${SAVE:-1000} --max_minutes ${MAX_MIN:-480} \
    2>&1 | grep -v Warning | tee -a $W/runs/${RUN:-r1}.log
fi
