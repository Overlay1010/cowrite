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

```bash
python -m venv venv && venv\Scripts\pip install -r requirements.txt   # Python 3.12에서 확인
venv\Scripts\python fetch_samples.py banks
venv\Scripts\python train.py --onnx ../../app/midi_io/basic_pitch/nmp.onnx --out run --steps 1000 --lr 1e-4 --banks banks --save_every 100
# 체크포인트를 재서 고른다 (backend에서): python -m eval.transcribe <run/stepN.onnx>
```

- 데이터는 매번 무작위로 만들어서(시드에 시각이 섞임) 다시 돌리면 숫자가 조금 다르다.
- 오래 학습하면 합성 소리에 맞춰져 실제 루프 점수가 떨어진다(학습률 5e-4로는 500걸음 뒤부터). 체크포인트마다 재서 고른다.
- 평가 샘플팩(samples/private)은 정답을 재는 데만 쓰고 학습에는 넣지 않는다.
