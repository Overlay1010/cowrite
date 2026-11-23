#!/usr/bin/env bash
# RunPod Pod(PyTorch 템플릿, /workspace = 네트워크 볼륨)에서 한 번: 코드 · 패키지 · Slakh2100 준비.
#   bash setup.sh            # Slakh 전체 (104GB 받기 + 변환)
#   MAX_GB=30 bash setup.sh  # 받는 양을 줄여 일부 곡만 (tar.gz는 앞에서부터 풀리므로 도중에 끊어도 받은 곡은 쓸 수 있다)
set -euo pipefail
W=/workspace
cd $W
[ -d cowrite ] || git clone https://github.com/Overlay1010/cowrite.git
cd cowrite/backend/train/transcriber
pip install -q -r runpod/requirements-gpu.txt
[ -d $W/banks ] || python fetch_samples.py $W/banks

URL="https://zenodo.org/records/4599666/files/slakh2100_flac_redux.tar.gz?download=1"
mkdir -p $W/slakh_raw
if [ ! -f $W/slakh_raw/.done ]; then
  # 받으면서 바로 푼다 (압축 파일을 따로 두지 않아 디스크를 아낌). test · omitted는 풀지 않는다.
  if [ -n "${MAX_GB:-}" ]; then
    curl -L --retry 5 -s "$URL" | head -c $((MAX_GB * 1024 * 1024 * 1024)) | tar xz -C $W/slakh_raw --exclude='*/test/*' --exclude='*/omitted/*' || true
  else
    curl -L --retry 5 -s "$URL" | tar xz -C $W/slakh_raw --exclude='*/test/*' --exclude='*/omitted/*'
  fi
  touch $W/slakh_raw/.done
fi
SRC=$(dirname "$(find $W/slakh_raw -maxdepth 3 -type d -name train | head -1)")
python slakh_data.py prepare "$SRC" $W/slakh --procs "$(nproc)"
du -sh $W/slakh
echo "준비 끝 — 이제 run.sh"
