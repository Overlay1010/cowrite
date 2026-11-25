#!/usr/bin/env bash
# RunPod Pod(PyTorch 템플릿, /workspace = 볼륨)에서 한 번: 코드 · 패키지 · Slakh2100 준비.
#   bash setup.sh            # Slakh 전체 (104GB 받기 + 변환)
#   MAX_GB=30 bash setup.sh  # 받는 양을 줄여 앞쪽 곡만 (tar.gz는 앞에서부터 풀리므로 도중에 끊어도 받은 곡은 쓸 수 있다)
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

URL="https://zenodo.org/records/4599666/files/slakh2100_flac_redux.tar.gz?download=1"
mkdir -p $W/slakh_raw
if [ ! -f $W/slakh_raw/.done ]; then
  echo "Slakh 받는 중 $(date +%H:%M) — 진행은 du -sh $W/slakh_raw"
  # 받으면서 바로 푼다 (압축 파일을 따로 두지 않아 디스크를 아낌). test · omitted는 풀지 않는다.
  if [ -n "${MAX_GB:-}" ]; then
    curl -L --retry 5 -s "$URL" | head -c $((MAX_GB * 1024 * 1024 * 1024)) | tar xz -C $W/slakh_raw --exclude='*/test/*' --exclude='*/omitted/*' || true
  else
    curl -L --retry 5 -s "$URL" | tar xz -C $W/slakh_raw --exclude='*/test/*' --exclude='*/omitted/*'
  fi
  touch $W/slakh_raw/.done
fi
echo "Slakh 받기 끝 $(date +%H:%M), 변환 시작"
SRC=$(dirname "$(find $W/slakh_raw -maxdepth 3 -type d -name train | head -1)")
# nproc은 서버 전체(48)를 보여 준다 — 컨테이너에 실제로 주어진 수만큼
python slakh_data.py prepare "$SRC" $W/slakh --procs "$(python -c "import big; print(big.cpu_count())")"
du -sh $W/slakh
touch $W/slakh/.ready
rm -rf $W/slakh_raw   # 변환이 끝나면 원본은 지운다 (디스크)
echo "준비 끝 $(date +%H:%M) — 이제 run.sh"
