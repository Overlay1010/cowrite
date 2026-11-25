"""Slakh2100 (CC BY 4.0, 여러 악기 곡 2,100개 + 악기마다 MIDI) → 학습용 파일, 그리고 학습 중 무작위로 꺼내기.

  python slakh_data.py prepare <slakh2100_flac_redux 폴더> <출력 폴더> --procs 8

- train · validation 폴더만 쓰고 test는 남긴다(나중에 공개 기준으로 재 볼 수 있게), 저자가 학습에 쓰지 말라는 omitted는 뺀다.
- 곡마다 22050Hz 모노 int16으로 두 개를 만든다:
  - 전체 믹스(드럼 포함) — 정답은 드럼이 아닌 악기의 MIDI 음 전부
  - 작은 묶음: 드럼 아닌 악기 1~3개(반은 드럼도 함께) — 샘플팩 루프처럼 악기가 적은 소리. 정답은 고른 악기의 음
- 오디오는 .npy(메모리 매핑으로 창만 읽음), 정답은 (시작초, 끝초, 음) .npy.
- 정답은 stem마다 원래 Basic Pitch로 옥타브를 확인해 소리에 맞게 옮긴다(octave_check). 곡마다 .check.json에 기록.
"""
from __future__ import annotations

import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

SR = 22050
SPLITS = ("train", "validation")


def _read(path: Path) -> np.ndarray:
    import soundfile as sf
    from math import gcd
    from scipy.signal import resample_poly

    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if sr != SR:
        g = gcd(SR, sr)
        x = resample_poly(x, SR // g, sr // g).astype(np.float32)
    return x


def _notes(path: Path) -> list[tuple[float, float, int]]:
    import pretty_midi

    pm = pretty_midi.PrettyMIDI(str(path))
    return [(n.start, n.end, n.pitch) for inst in pm.instruments if not inst.is_drum for n in inst.notes if n.end > n.start]


# --- 정답 옥타브 확인 ---------------------------------------------------------------------------
# Slakh 정답 MIDI는 악기에 따라 소리와 옥타브가 다르다(BabySlakh 베이스 22개 중 21개가 한 옥타브 높게 적힘 — 악보 관례대로
# 적고 렌더 음색은 한 옥타브 낮게 울림). 그대로 학습하면 모델이 베이스를 배음 자리로 적어 조성이 흔들린다(69 → 61/88).
# 그래서 stem마다 원래 Basic Pitch(nmp.onnx, 학습 안 한 것)로 정답 음 자리의 활성도를 재 옥타브를 고른다.
JUDGE = Path(__file__).resolve().parents[2] / "app" / "midi_io" / "basic_pitch" / "nmp.onnx"
SHIFTS = (-24, -12, 0, 12, 24)
GAIN = 1.5  # 옮긴 자리 활성도가 그대로보다 이 배 넘게 크고
MARGIN = 0.1  # 이만큼 넘게 클 때만 옮긴다 (배음 때문에 애매하면 그대로)
HEARD = 0.15  # 가장 맞는 옥타브의 평균 활성도가 이보다 작으면 소리와 안 맞는 stem
EXCERPT = 30.0  # 초
# 창 나누기는 app/midi_io/transcribe.py(basic-pitch 0.4.0, Apache 2.0)와 같다. 앱 모듈은 pydantic이 필요해 Pod에서 따로 둔다
_FFT_HOP, _FPS = 256, SR // 256
_WIN = SR * 2 - _FFT_HOP
_OV_FRAMES = 30
_OV = _OV_FRAMES * _FFT_HOP
_WIN_FRAMES = _FPS * 2
_session = None


def _judge_session():
    global _session
    if _session is None:
        import onnxruntime as ort

        opt = ort.SessionOptions()
        opt.intra_op_num_threads = 1  # 곡마다 프로세스를 따로 띄우므로
        _session = ort.InferenceSession(str(JUDGE), opt, providers=["CPUExecutionProvider"])
    return _session


def note_activation(audio: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """원래 Basic Pitch의 음 활성도 (프레임, 88)와 프레임 시각(초)."""
    padded = np.concatenate([np.zeros(_OV // 2, np.float32), audio.astype(np.float32)])
    wins = np.stack([np.pad(padded[i : i + _WIN], (0, _WIN - len(padded[i : i + _WIN]))) for i in range(0, len(padded), _WIN - _OV)])[..., None]
    s = _judge_session()
    out = np.concatenate([s.run(["StatefulPartitionedCall:1"], {"serving_default_input_2:0": wins[i : i + 16]})[0] for i in range(0, len(wins), 16)])
    out = out[:, _OV_FRAMES // 2 : -_OV_FRAMES // 2, :].reshape(-1, 88)[: int(len(audio) * _FPS / SR)]
    n = np.arange(len(out))
    offset = (_FFT_HOP / SR) * (_WIN_FRAMES - _WIN / _FFT_HOP) + 0.0018  # Basic Pitch의 창 보정값
    return out, n * _FFT_HOP / SR - offset * np.floor(n / _WIN_FRAMES)


def octave_check(audio: np.ndarray, notes: list[tuple[float, float, int]]) -> tuple[int, float, float]:
    """(옮길 반음, 고른 옥타브의 평균 활성도, 그대로의 평균 활성도). 음마다 시작 뒤 0.3초 안 최대 활성도의 평균.
    빠르게 하려고 음이 가장 많은 EXCERPT초 구간만 본다 (곡 전체와 같은 결론인지 BabySlakh로 확인)."""
    if not notes:
        return 0, 0.0, 0.0
    starts_all = np.array(sorted(s for s, _, _ in notes))
    i = int(np.argmax(np.searchsorted(starts_all, starts_all + EXCERPT) - np.arange(len(starts_all))))
    t0 = max(0.0, float(starts_all[i]) - 0.5)
    seg = audio[int(t0 * SR) : int((t0 + EXCERPT + 1) * SR)]
    notes = [(s - t0, e - t0, p) for s, e, p in notes if t0 <= s < t0 + EXCERPT]
    act, times = note_activation(seg)
    if not notes or not len(act):
        return 0, 0.0, 0.0
    starts = np.searchsorted(times, [s for s, _, _ in notes])
    ends = np.searchsorted(times, [min(e, s + 0.3) for s, e, _ in notes], side="right")
    score = {}
    for k in SHIFTS:
        vals = []
        for (s, e, p), a, b in zip(notes, starts, ends):
            col = p + k - 21
            vals.append(float(act[a : max(b, a + 1), col].max()) if 0 <= col < 88 and a < len(act) else 0.0)
        score[k] = float(np.mean(vals))
    best = max(SHIFTS, key=lambda k: (score[k], -abs(k)))
    if best != 0 and score[best] > GAIN * score[0] and score[best] - score[0] > MARGIN:
        return best, score[best], score[0]
    return 0, score[0], score[0]


def _save(out: Path, name: str, audio: np.ndarray, notes: list) -> None:
    peak = float(np.abs(audio).max())
    if peak < 1e-4 or not notes:
        return
    np.save(out / f"{name}.audio.npy", np.round(audio / peak * 0.9 * 32767).astype(np.int16))
    np.save(out / f"{name}.notes.npy", np.array(sorted(notes), dtype=np.float32))


def prepare_track(job: tuple[str, str, int]) -> str:
    try:
        return _prepare_track(job)
    except Exception as e:  # 받다가 끊은 마지막 곡 · 깨진 파일은 건너뛴다
        return f"{Path(job[0]).name} 건너뜀: {type(e).__name__} {e}"


def _prepare_track(job: tuple[str, str, int]) -> str:
    import yaml

    track, out, seed = Path(job[0]), Path(job[1]), job[2]
    name = track.name
    if (out / f"{name}.check.json").exists():  # 정답 확인까지 끝난 곡 (믹스를 뺀 곡은 .notes.npy가 없다)
        return f"{name} 있음"
    rng = np.random.default_rng(seed)
    meta = yaml.safe_load((track / "metadata.yaml").read_text(encoding="utf-8"))
    stems = meta.get("stems", {})
    pitched, drums = [], []
    for sid, info in stems.items():
        # metadata의 audio_rendered · midi_saved는 BabySlakh에서 실제 파일과 안 맞아 파일이 있는지로 본다 (macOS ._ 파일 빼고)
        wav = next((w for w in (track / "stems").glob(f"{sid}.*") if not w.name.startswith("._")), None)
        if wav is None:
            continue
        if info.get("is_drum"):
            drums.append(wav)
        elif (track / "MIDI" / f"{sid}.mid").exists():
            pitched.append((wav, track / "MIDI" / f"{sid}.mid"))
    if not pitched:
        return f"{name} 악기 없음"
    mix_file = next(m for m in track.glob("mix.*") if not m.name.startswith("._"))
    audio = {w: _read(w) for w, _ in pitched}
    notes, check = {}, {}
    for w, m in pitched:
        raw = _notes(m)
        k, act, act0 = octave_check(audio[w], raw)
        notes[w] = [(s, e, p + k) for s, e, p in raw if 0 <= p + k <= 127]
        check[w] = {"shift": k, "act": round(act, 3), "act0": round(act0, 3), "rms": float(np.sqrt(np.mean(audio[w] ** 2)))}
    good = [(w, m) for w, m in pitched if check[w]["act"] >= HEARD]
    loud = max(c["rms"] for c in check.values())
    # 소리와 안 맞는 stem이 크게 들리면 전체 믹스는 정답이 빠진 소리가 돼 쓰지 않는다 (작은 묶음은 맞는 stem에서만)
    bad_loud = [w for w, _ in pitched if check[w]["act"] < HEARD and check[w]["rms"] > 0.15 * loud]
    if not bad_loud:
        _save(out, name, _read(mix_file), [n for v in notes.values() for n in v])
    (out / f"{name}.check.json").write_text(
        __import__("json").dumps({"mix": not bad_loud, "stems": {Path(w).stem: c for w, c in check.items()}}), encoding="utf-8"
    )
    if not good:
        return f"{name} 소리와 맞는 악기 없음"
    k = int(rng.integers(1, min(3, len(good)) + 1))
    pick = [good[i] for i in rng.choice(len(good), k, replace=False)]
    parts = [audio[w] * float(rng.uniform(0.4, 1.0)) for w, _ in pick]
    if drums and rng.random() < 0.5:
        parts.append(_read(drums[int(rng.integers(0, len(drums)))]) * float(rng.uniform(0.2, 0.7)))
    n = max(len(p) for p in parts)
    sub = np.zeros(n, np.float32)
    for p in parts:
        sub[: len(p)] += p
    _save(out, name + "_sub", sub, [x for w, _ in pick for x in notes[w]])
    moved = sum(c["shift"] != 0 for c in check.values())
    return f"{name} 악기 {len(pitched)}개 (옥타브 옮김 {moved} · 안 맞음 {len(pitched) - len(good)}), 믹스 {'씀' if not bad_loud else '뺌'}, 작은 묶음 {k}개"


def prepare(src: Path, out: Path, procs: int) -> None:
    out.mkdir(parents=True, exist_ok=True)
    tracks = sorted(t for s in SPLITS for t in (src / s).glob("Track*") if t.is_dir())
    if not tracks:  # BabySlakh처럼 나눔 폴더 없이 곡만 있는 경우
        tracks = sorted(t for t in src.glob("Track*") if t.is_dir())
    print(f"곡 {len(tracks)}개", flush=True)
    jobs = [(str(t), str(out), i) for i, t in enumerate(tracks)]
    with Pool(procs) as pool:
        for i, msg in enumerate(pool.imap_unordered(prepare_track, jobs), 1):
            if i % 50 == 0 or i == len(jobs):
                print(f"{i}/{len(jobs)} {msg}", flush=True)


class Slakh:
    """학습 중 꺼내기: 길이에 비례해 곡을 고르고 seconds초 구간의 오디오(float32)와 그 구간 기준 정답."""

    def __init__(self, folder: Path):
        self.items = []
        for a in sorted(folder.glob("*.audio.npy")):
            nfile = a.with_name(a.name.replace(".audio.npy", ".notes.npy"))
            if nfile.exists():
                self.items.append((a, nfile, np.load(a, mmap_mode="r").shape[0]))
        if not self.items:
            raise FileNotFoundError(f"{folder}에 *.audio.npy가 없어요")
        lengths = np.array([n for _, _, n in self.items], dtype=np.float64)
        self.p = lengths / lengths.sum()
        self._cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def __len__(self) -> int:
        return len(self.items)

    def _open(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        if i not in self._cache:
            if len(self._cache) > 256:
                self._cache.clear()
            a, nfile, _ = self.items[i]
            self._cache[i] = (np.load(a, mmap_mode="r"), np.load(nfile))
        return self._cache[i]

    def excerpt(self, rng: np.random.Generator, seconds: float = 6.0) -> tuple[np.ndarray, list[tuple[float, float, int]]]:
        audio, notes = self._open(int(rng.choice(len(self.items), p=self.p)))
        n = int(seconds * SR)
        st = int(rng.integers(0, max(1, len(audio) - n)))
        x = np.asarray(audio[st : st + n], dtype=np.float32) / 32767.0
        x *= float(rng.uniform(0.3, 1.0))
        t0, t1 = st / SR, (st + len(x)) / SR
        keep = notes[(notes[:, 1] > t0) & (notes[:, 0] < t1)]
        return x, [(float(s) - t0, float(e) - t0, int(p)) for s, e, p in keep]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prepare"])
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--procs", type=int, default=8)
    a = ap.parse_args()
    prepare(Path(a.src), Path(a.out), a.procs)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
