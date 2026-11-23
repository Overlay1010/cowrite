"""Slakh2100 (CC BY 4.0, 여러 악기 곡 2,100개 + 악기마다 MIDI) → 학습용 파일, 그리고 학습 중 무작위로 꺼내기.

  python slakh_data.py prepare <slakh2100_flac_redux 폴더> <출력 폴더> --procs 8

- train · validation 폴더만 쓰고 test는 남긴다(나중에 공개 기준으로 재 볼 수 있게), 저자가 학습에 쓰지 말라는 omitted는 뺀다.
- 곡마다 22050Hz 모노 int16으로 두 개를 만든다:
  - 전체 믹스(드럼 포함) — 정답은 드럼이 아닌 악기의 MIDI 음 전부
  - 작은 묶음: 드럼 아닌 악기 1~3개(반은 드럼도 함께) — 샘플팩 루프처럼 악기가 적은 소리. 정답은 고른 악기의 음
- 오디오는 .npy(메모리 매핑으로 창만 읽음), 정답은 (시작초, 끝초, 음) .npy.
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
    if (out / f"{name}.notes.npy").exists():
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
    notes = {w: _notes(m) for w, m in pitched}
    _save(out, name, _read(mix_file), [n for v in notes.values() for n in v])

    k = int(rng.integers(1, min(3, len(pitched)) + 1))
    pick = [pitched[i] for i in rng.choice(len(pitched), k, replace=False)]
    parts = [_read(w) * float(rng.uniform(0.4, 1.0)) for w, _ in pick]
    if drums and rng.random() < 0.5:
        parts.append(_read(drums[int(rng.integers(0, len(drums)))]) * float(rng.uniform(0.2, 0.7)))
    n = max(len(p) for p in parts)
    sub = np.zeros(n, np.float32)
    for p in parts:
        sub[: len(p)] += p
    _save(out, name + "_sub", sub, [x for w, _ in pick for x in notes[w]])
    return f"{name} 악기 {len(pitched)}개, 작은 묶음 {k}개"


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
