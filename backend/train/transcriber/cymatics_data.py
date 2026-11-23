"""샘플팩(samples/private, 저장소에 없음)을 학습 데이터로 (2026-11-22, 사용자 결정).

평가(eval/transcribe.py)와 같은 A/B 묶음을 써서 **B 묶음 곡은 학습에 넣지 않는다** — B로 정직하게 잰다.
같은 곡의 MIDI가 따로 들어 있는 경우도 곡 이름으로 막는다.

- 실제 짝: A 묶음의 wav · mid 짝 중 정답 MIDI와 가장 맞는 stem(그리고 완성 루프는 쓰지 않는다 — 정답 MIDI에 없는
  악기가 섞여 있어서 그 소리를 "음 없음"으로 배우게 된다)
- MIDI만: 오디오 없이 들어 있는 MIDI 팩 — 실제 곡의 코드 · 멜로디를 synth_data의 신스 · 녹음으로 렌더한다

  python cymatics_data.py ../../../samples/private cym.pkl   # 목록 만들기 (backend 코드로 MIDI를 읽는다)
"""
from __future__ import annotations

import pickle
import re
import sys
import zlib
from pathlib import Path

import numpy as np


def group(name: str) -> str:
    return "AB"[zlib.crc32(name.encode()) % 2]  # eval/transcribe.py와 같게


def song(name: str) -> str:
    n = re.sub(r"^Cymatics( Gems Vol \d+)? - ", "", name)
    return re.split(r" - | MIDI", n)[0].strip().lower()


def build(root: Path, out: Path) -> None:
    backend = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(backend))
    from app.midi_io import transcribe as T
    from app.midi_io.convert import midi_bytes_to_project
    from eval.transcribe import _long, match, pairs

    found = pairs(root)
    held_out = {song(w.stem) for _, w in found if group(w.stem) == "B"}
    paired = {m.resolve() for m, _ in found}

    def notes_of(mid: Path):
        p = midi_bytes_to_project(mid.read_bytes(), "m")
        return [(n.start, n.duration, n.pitch, n.velocity / 127) for t in p.tracks if not t.is_drum for n in t.notes]

    real = []
    for mid, wav in found:
        if group(wav.stem) != "A":
            continue
        bpm = T.bpm_from_name(wav.stem)
        ref = notes_of(mid)
        if not bpm or not ref:
            continue
        tol = 0.05 * bpm / 60
        # 정답 악기 stem 고르기: 지금 모델로 읽어 음이름(옥타브 무시)이 가장 맞는 것. 음 높이로 고르면 모델이 잘 읽는
        # 쉬운 stem만 남는다. 음이름도 거의 안 맞으면 정답에 없는 악기라 버린다
        best, best_f1 = None, 0.15
        ref_pc = sorted((b, p % 12) for b, _, p, _ in ref)
        for s in mid.parent.glob("*.wav"):
            if s.name.startswith("._"):
                continue
            ev = T.transcribe(T.load_audio(s.read_bytes()))
            f1 = match(ref_pc, sorted((e.start * bpm / 60, e.pitch % 12) for e in ev), tol)[0]
            if f1 > best_f1:
                best, best_f1, best_ev = s, f1, ev
        if best is None:
            continue
        # stem이 정답 MIDI보다 2~4배 긴 곳이 많다. 같은 루프를 되풀이하면 정답도 되풀이해 붙이고, 첫 사이클 뒤에 다른 연주로
        # 바뀌면(음이름 F1이 첫 사이클의 6할 아래) 거기서 자른다
        dur = len(T.load_audio(best.read_bytes())) / T.SAMPLE_RATE
        one = [(b * 60 / bpm, (b + d) * 60 / bpm, p) for b, d, p, _ in ref]
        end = max(t for _, t, _ in one)
        k = max(1, round(dur / end))
        per = dur / k if dur / k <= end * 1.15 else end
        got = [(e.start, e.pitch % 12) for e in best_ev]
        labels, first = [], None
        for c in range(k):
            cyc = [(a + c * per, b + c * per, p) for a, b, p in one]
            f = match(sorted((a, p % 12) for a, _, p in cyc), [g for g in got if c * per - 0.1 <= g[0] < (c + 1) * per], 0.05)[0]
            first = f if first is None else first
            if c and f < 0.6 * first:
                break
            labels += cyc
        real.append({"audio": str(best), "labels": labels, "name": wav.stem, "f1": best_f1, "end": max(t for _, t, _ in labels)})
        print(f"짝 {wav.stem[:50]:50s} {best.name[len(wav.stem):][:20]:20s} 음이름 F1 {best_f1:.2f} 사이클 {len(labels) // len(one)}/{k}", flush=True)

    midis = []
    for m in _long(root).rglob("*.mid"):
        if "__MACOSX" in m.parts or m.name.startswith("._") or m.resolve() in paired or song(m.stem) in held_out:
            continue
        try:
            ns = notes_of(m)
        except Exception:
            continue
        if len(ns) >= 4:
            midis.append({"name": m.stem, "notes": ns})
    print(f"실제 짝(A) {len(real)} · MIDI만 {len(midis)} · 뺀 B 곡 {len(held_out)}")
    pickle.dump({"real": real, "midis": midis}, open(out, "wb"))


# ---- 학습 때 쓰는 쪽 (torch 환경, backend 코드 없이) ----

_DATA: dict | None = None
_AUDIO: dict[str, np.ndarray] = {}


def load(path: Path) -> None:
    global _DATA
    _DATA = pickle.load(open(path, "rb"))


def _read(path: str, sr: int) -> np.ndarray:
    if path not in _AUDIO:
        import soundfile as sf
        from scipy.signal import resample_poly

        p = path if path.startswith("\\\\?\\") else "\\\\?\\" + path
        x, r = sf.read(p, dtype="float32", always_2d=True)
        x = x.mean(1)
        if r != sr:
            from math import gcd

            g = gcd(r, sr)
            x = resample_poly(x, sr // g, r // g).astype(np.float32)
        _AUDIO[path] = x
    return _AUDIO[path]


def real_loop(rng, sr: int):
    """A 묶음 실제 stem과 정답."""
    e = _DATA["real"][int(rng.integers(0, len(_DATA["real"])))]
    labels = list(e["labels"])
    # 정답 MIDI가 오디오보다 짧은 곳이 있다(같은 루프를 되풀이한 오디오) — 정답이 없는 뒤쪽을 "음 없음"으로 배우지 않게 자른다
    x = _read(e["audio"], sr)[: int((e["end"] + 0.3) * sr)]
    gain = float(rng.uniform(0.3, 0.95))
    peak = np.abs(x).max()
    return (x / peak * gain if peak > 0 else x).astype(np.float32), labels


def midi_music(rng):
    """MIDI 팩의 곡 하나를 synth_data.make_music과 같은 모양으로: (bpm, 초, 파트). 조를 무작위로 옮긴다."""
    notes = []
    while not notes:  # 옮긴 뒤 피아노 범위를 벗어나거나 아주 짧은 음뿐인 곡이면 다시 고른다
        e = _DATA["midis"][int(rng.integers(0, len(_DATA["midis"])))]
        shift = int(rng.integers(-5, 7))
        notes = [(s, d, p + shift, v) for s, d, p, v in e["notes"] if 21 <= p + shift <= 108 and d > 0.05]
    bpm = float(rng.uniform(70, 170))
    end = max(s + d for s, d, _, _ in notes)
    # 너무 길면 앞 8마디만
    beats = min(end, 32.0)
    notes = [(s, min(d, beats - s), p, v) for s, d, p, v in notes if s < beats]
    low = [n for n in notes if n[2] < 48]
    high = [n for n in notes if n[2] >= 48]
    parts = [{"role": "chords", "notes": high}] if high else []
    if low:
        parts.append({"role": "bass", "notes": low})
    return bpm, beats * 60 / bpm, parts


if __name__ == "__main__":
    build(Path(sys.argv[1]), Path(sys.argv[2]))
