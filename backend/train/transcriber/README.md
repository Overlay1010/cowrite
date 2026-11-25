# 오디오 → 음표 모델 이어서 학습

앱이 쓰는 `app/midi_io/basic_pitch/nmp_cowrite.onnx`를 만드는 코드다. 앱 실행에는 필요 없다(torch도 앱에는 안 들어간다).
왜 · 어떻게 정했는지는 [docs/audio-to-midi.md](../../../docs/audio-to-midi.md).

| 파일 | 하는 일 |
| --- | --- |
| `synth_data.py` | 샘플팩 스타일 루프 합성: 무작위 코드 · 아르페지오 · 멜로디 · 베이스 → 신스(polyBLEP 톱니 · 사각 · FM · 슈퍼소 …) 또는 FluidR3 녹음으로 렌더 → 리버브 · 딜레이 · 코러스 · 포화 · 로파이 · 드럼. 정답은 친 MIDI 음뿐(메아리 · 잔향 · 드럼은 무시하도록) |
| `fetch_samples.py` | FluidR3 GM(CC BY 3.0) 중 음 높이가 있는 악기 65개를 3반음마다 받아 22050Hz wav로 |
| `bp_torch.py` | `nmp.onnx` → PyTorch (onnx2torch, 오디오 축 반사 패딩을 고쳐 끼움). 원래 ONNX와 출력 차이 1e-6 이하 |
| `train.py` | CQT를 고정하고 뒤쪽 합성곱 6층만 학습(음 · 시작점 · 윤곽 BCE). 체크포인트마다 `.pt`와 `.onnx` |
| `patch_onnx.py` | 학습한 가중치를 원래 `nmp.onnx` 그래프에 끼운다. PyTorch로 다시 내보내면 같은 모델이 7배 느려서 |
| `surge_render.py` | 무료 신스 Surge XT(GPL-3.0) 프리셋으로 정답 딸린 루프를 미리 렌더 (프리셋 검사 `scan` → `render`). `--cym` 없이 돌리면 구매한 샘플팩이 하나도 안 섞인다 |
| `cymatics_data.py` | 구매한 샘플팩(samples/private)의 A 묶음만 학습 목록으로 (B는 평가용으로 남김). 이 PC에서만 쓰고 RunPod에는 올리지 않는다 |
| `big.py` | Basic Pitch 위에 보정층(합성곱 + 트랜스포머, 267만 개)을 얹은 큰 모델. 처음엔 Basic Pitch와 같은 출력에서 출발. GPU · bf16, `--slakh` · `--surge`, `--max_minutes`로 시간 상한 |
| `slakh_data.py` | Slakh2100(CC BY 4.0) → 곡마다 전체 믹스 + 악기 1~3개 작은 묶음 (22050Hz int16 .npy + 정답), 학습 중 메모리 매핑으로 구간만 읽음 |
| `slakh_hf.py` | Slakh을 Hugging Face 사본(`WhiRik/slakh_flac`, redux와 같은 1,559곡 · 폴더 구조, 접근 동의 필요)에서 곡 단위로 받아 바로 변환하고 원본은 지운다. Zenodo tar.gz는 곡 파일이 뒤섞여 앞부분만으로는 온전한 곡이 없고 끊기면 이어 받을 수 없어서 |
| `runpod/` | RunPod GPU에서: `setup.sh`(코드 · 패키지 · Slakh 받으면서 풀기 · 변환) → `SMOKE=1 run.sh`(200걸음) → `run.sh`(본 학습). 이 PC에서는 `watch.py`가 몇 분마다 체크포인트 · 로그를 `runs_gpu/`로 받아 오고, 잔액이 바닥에 가까우면 다 받은 뒤 Pod를 지운다 |

```bash
python -m venv venv && venv\Scripts\pip install -r requirements.txt   # Python 3.12에서 확인
venv\Scripts\python fetch_samples.py banks
venv\Scripts\python train.py --onnx ../../app/midi_io/basic_pitch/nmp.onnx --out run --steps 1000 --lr 1e-4 --banks banks --save_every 100
# 체크포인트를 재서 고른다 (backend에서): python -m eval.transcribe <run/stepN.onnx>
```

- 데이터는 매번 무작위로 만들어서(시드에 시각이 섞임) 다시 돌리면 숫자가 조금 다르다.
- 오래 학습하면 합성 소리에 맞춰져 실제 루프 점수가 떨어진다(학습률 5e-4로는 500걸음 뒤부터). 체크포인트마다 재서 고른다.
- 평가 샘플팩(samples/private)의 B 묶음은 학습에 넣지 않는다. A 묶음은 `cymatics_data.py`로 시험만 했고 앱 모델에는 안 썼다(출시 전 라이선스 확인 필요).

## GPU 학습 (RunPod)

학습만 GPU를 빌려 하고, 결과 ONNX(약 11MB)는 앱이 지금처럼 CPU에서 돌린다. 구매한 샘플팩은 올리지 않는다 — 데이터는 Slakh2100(공개) · Surge 렌더(`--cym` 없이) · 합성 루프뿐이고, 체크포인트를 내려받아 이 PC에서 `python -m eval.transcribe <stepN.onnx>`로 잰다.

```bash
# 이 PC: Surge 루프 (샘플팩 MIDI 없이)
python surge_render.py render "<Surge Synth Team>" surge_presets.json surge_clean --loops 10000 --procs 5
# Pod (/workspace = 네트워크 볼륨): surge_clean을 /workspace/surge_clean으로 올린 뒤
TRACKS=600 bash runpod/setup.sh  # HF 토큰을 ~/.cache/huggingface/token에 둔 뒤
SMOKE=1 bash runpod/run.sh      # 한 걸음 시간 확인
MAX_MIN=480 bash runpod/run.sh  # 본 학습, 1000걸음마다 저장
# 이 PC: 학습하는 동안 계속 켜 둔다 (결과가 이 PC에 쌓여 Pod가 지워져도 남음)
python runpod/watch.py <pod-id> --ssh "root@IP -p PORT" --floor 2
```

- Slakh 출처: Manilow et al., "Cutting Music Source Separation Some Slakh: A Dataset to Study the Impact of Training Data Quality and Quantity", WASPAA 2019 (CC BY 4.0).
- 2026-11-25 RunPod 짧은 시험(공용 RTX 4090, 캐나다): 200걸음 21초(한 걸음 약 0.1초, 배치 64), `watch.py`가 체크포인트를 받아 오고 앱 변환 코드가 그대로 읽음. GPU에서만 나던 손실 nan(bf16에서 1 - 1e-4 → 1)과 CPU 수 오인(48개로 보고 실제 6.8개)을 고침. Zenodo 받는 속도는 초당 0.3~5MB로 들쭉날쭉.
- BabySlakh(앞 20곡)로 이 PC에서 확인: 변환 20곡 33초 · 411MB, 정답 시작점과 모델 시작점 차이 중간 5.5ms, `big.py`가 CPU에서 4걸음 학습 → 저장 → ONNX 내보내기, 앱 변환 코드가 그 ONNX를 그대로 읽음.

