#!/usr/bin/env bash
# RunPod Pod(PyTorch 템플릿, /workspace = 볼륨)에서 한 번: 코드 · 패키지 · Slakh2100 준비.
#   TRACKS=600 bash setup.sh   # Hugging Face 사본에서 600곡 받기 + 변환 (토큰: ~/.cache/huggingface/token 또는 HF_TOKEN)
# Zenodo 원본(tar.gz)은 곡 파일이 뒤섞여 앞부분만으로는 온전한 곡이 없고, 끊기면 이어 받을 수 없어서 쓰지 않는다.
# "패키지 준비 끝"이 나오면 다른 터미널에서 SMOKE=1 run.sh를 같이 돌려도 된다(Slakh 없이 합성 · Surge로).
set -euo pipefail
W=/workspace
cd $W
[ -d cowrite ] || git clone -b develop https://github.com/Overlay1010/cowrite.git
cd cowrite/backend/train/transcriber
git pull -q || true
# Ubuntu 24.04 이미지는 시스템 pip를 막는다(PEP 668). 컨테이너 전용 Python이라 허용
PIP_BREAK_SYSTEM_PACKAGES=1 PIP_ROOT_USER_ACTION=ignore pip install -q -r runpod/requirements-gpu.txt
[ -d $W/banks ] || python fetch_samples.py $W/banks
echo "패키지 준비 끝 $(date +%H:%M)"

echo "Slakh 받기 · 변환 시작 $(date +%H:%M)"
CPUS=$(python -c "import big; print(big.cpu_count())")
# 받기는 네트워크를 기다리는 시간이 많아 CPU 수보다 많이 띄운다
python slakh_hf.py $W/slakh --tracks ${TRACKS:-600} --procs $(( CPUS * 2 ))
N=$(ls $W/slakh/*.notes.npy 2>/dev/null | wc -l)
du -sh $W/slakh
if [ "$N" -lt 100 ]; then echo "변환된 파일이 $N개뿐 — 멈춤"; exit 1; fi
touch $W/slakh/.ready
rm -rf $W/slakh_raw
echo "준비 끝 $(date +%H:%M) — 이제 run.sh"
