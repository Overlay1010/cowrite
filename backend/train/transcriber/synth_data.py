"""샘플팩 스타일 루프 합성기: 무작위 음악(코드 · 아르페지오 · 멜로디 · 베이스) → 신스 · 녹음 샘플로 렌더 → 효과.
정답은 친 MIDI 음 (딜레이 메아리 · 리버브 꼬리 · 옥타브 겹침 · 드럼은 정답에 없음 = 무시하도록 배움).

make_loop(rng) -> (audio float32 22050Hz mono, notes [(start_s, end_s, pitch)])
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from scipy.signal import butter, lfilter, fftconvolve

SR = 22050
SAMPLED_SHARE = 0.5  # 녹음 샘플로 렌더할 파트 비율 (나머지는 신스)
MAJOR = [0, 2, 4, 5, 7, 9, 11]
MINOR = [0, 2, 3, 5, 7, 8, 10]

# ---------------------------------------------------------------- 음악 내용


def scale_pitches(tonic: int, mode: list[int], lo: int, hi: int) -> list[int]:
    return [p for p in range(lo, hi + 1) if (p - tonic) % 12 in mode]


def diatonic_chord(tonic: int, mode: list[int], degree: int, size: int) -> list[int]:
    """degree 0~6, 3·5·7·9음 쌓기 → 음이름(0~11)."""
    return [(tonic + mode[(degree + 2 * k) % 7] + 12 * ((degree + 2 * k) // 7)) % 12 for k in range(size)]


def voice(pcs: list[int], lo: int, hi: int, rng) -> list[int]:
    """음이름들을 lo~hi 안에 배치 (촘촘 · 넓게 섞어서)."""
    base = rng.integers(lo, max(lo + 1, hi - 12))
    out = []
    cur = base
    for i, pc in enumerate(pcs):
        p = cur + ((pc - cur) % 12)
        if i > 0 and rng.random() < 0.3:
            p += 12
        if p > hi:
            p -= 12
        out.append(int(p))
        cur = p
    return sorted(set(out))


def rhythm(kind: str, beats: float, rng) -> list[tuple[float, float]]:
    """(시작 박, 길이 박)."""
    out = []
    if kind == "hold":
        return [(0.0, beats)]
    if kind == "half":
        return [(b, 2.0) for b in np.arange(0, beats, 2.0)]
    if kind == "quarter":
        return [(b, 0.9) for b in np.arange(0, beats, 1.0)]
    if kind == "stab":
        cells = [0, 0.75, 1.5, 2.5, 3, 3.5]
        for b in np.arange(0, beats, 4.0):
            for c in cells:
                if rng.random() < 0.6 and b + c < beats:
                    out.append((b + c, rng.choice([0.25, 0.5, 0.75])))
        return out or [(0.0, 0.5)]
    if kind == "eighth":
        return [(b, 0.45) for b in np.arange(0, beats, 0.5)]
    if kind == "sixteenth":
        return [(b, 0.22) for b in np.arange(0, beats, 0.25)]
    if kind == "triplet":
        return [(b, 0.3) for b in np.arange(0, beats, 1 / 3)]
    raise ValueError(kind)


def make_music(rng) -> tuple[float, float, list[dict]]:
    """(bpm, 초 길이, 파트들). 파트 = {"role", "notes": [(start_beat, dur_beat, pitch, vel)]}."""
    bpm = float(rng.uniform(60, 180))
    tonic = int(rng.integers(0, 12))
    mode = MAJOR if rng.random() < 0.4 else MINOR
    bars = int(rng.choice([2, 4]))
    chord_len = float(rng.choice([2.0, 4.0]))
    beats = bars * 4.0
    degrees = [int(rng.integers(0, 7)) for _ in range(int(beats / chord_len))]
    if rng.random() < 0.6:
        degrees[0] = 0
    roles = ["chords"] + [r for r in ("melody", "bass", "arp", "bell") if rng.random() < 0.45]
    if rng.random() < 0.25:  # 한 파트만 (stem처럼)
        roles = [str(rng.choice(["chords", "melody", "arp", "bell"]))]
    parts = []
    for role in roles:
        notes = []
        if role in ("chords", "arp"):
            size = int(rng.choice([3, 4, 4, 5]))
            kind = str(rng.choice(["hold", "half", "quarter", "stab"])) if role == "chords" else str(rng.choice(["eighth", "sixteenth", "triplet", "eighth"]))
            lo, hi = (48, 79) if role == "chords" else (55, 88)
            for i, d in enumerate(degrees):
                v = voice(diatonic_chord(tonic, mode, d, size), lo, hi, rng)
                for s, dur in rhythm(kind, chord_len, rng):
                    if role == "chords":
                        for p in v:
                            notes.append((i * chord_len + s, dur, p))
                    else:  # 펼침: 오르기 · 오르내리기, 가끔 무작위 음
                        order = v + v[-2:0:-1] if rng.random() < 0.4 else v
                        idx = int(rng.integers(0, len(v))) if rng.random() < 0.2 else int(round(s * 4)) % len(order)
                        notes.append((i * chord_len + s, dur, order[idx]))
        elif role == "melody" or role == "bell":
            lo, hi = (60, 86) if role == "melody" else (72, 96)
            pool = scale_pitches(tonic, mode, lo, hi)
            pos = int(rng.integers(0, len(pool)))
            t = 0.0
            while t < beats:
                dur = float(rng.choice([0.25, 0.5, 0.5, 0.75, 1.0, 1.5, 2.0]))
                if rng.random() < 0.2:
                    t += dur  # 쉼
                    continue
                pos = int(np.clip(pos + rng.choice([-2, -1, -1, 0, 1, 1, 2, 3, -3]), 0, len(pool) - 1))
                notes.append((t, min(dur, beats - t) * rng.uniform(0.6, 1.0), pool[pos]))
                t += dur
        elif role == "bass":
            kind = str(rng.choice(["hold", "half", "quarter", "eighth", "stab"]))
            for i, d in enumerate(degrees):
                root = int(28 + (tonic + mode[d]) % 12 + 12 * rng.integers(0, 2))  # E1~D#3
                for s, dur in rhythm(kind, chord_len, rng):
                    notes.append((i * chord_len + s, dur, root))
        vel = rng.uniform(0.5, 1.0)
        parts.append({"role": role, "notes": [(s, d, p, float(np.clip(vel + rng.normal(0, 0.08), 0.2, 1))) for s, d, p in notes if d > 0.05]})
    return bpm, beats * 60 / bpm, parts


# ---------------------------------------------------------------- 음색


def blep_saw(cycles: np.ndarray, dt: float) -> np.ndarray:
    """에일리어싱을 줄인 톱니파 (polyBLEP). cycles = 위상(주기 단위)."""
    t = cycles % 1.0
    x = 2 * t - 1
    dt = min(max(dt, 1e-6), 0.5)
    lo = t < dt
    a = t[lo] / dt
    x[lo] -= a + a - a * a - 1
    hi = t > 1 - dt
    b = (t[hi] - 1) / dt
    x[hi] -= b * b + b + b + 1
    return x


def adsr(n: int, a: float, d: float, s: float, r: float, held: int) -> np.ndarray:
    env = np.zeros(n, dtype=np.float32)
    ai, di, ri = max(1, int(a * SR)), max(1, int(d * SR)), max(1, int(r * SR))
    held = min(held, n)
    seg = np.concatenate([np.linspace(0, 1, ai, endpoint=False), np.linspace(1, s, di, endpoint=False)])
    if len(seg) >= held:
        env[:held] = seg[:held]
        last = seg[held - 1] if held else 0
    else:
        env[: len(seg)] = seg
        env[len(seg) : held] = s
        last = s
    rel = last * np.exp(-np.arange(n - held) / (ri / 5))
    env[held:] = rel
    return env


class Synth:
    def __init__(self, rng, role: str):
        self.wave = str(rng.choice(["saw", "square", "tri", "sine", "fm", "supersaw", "pluck", "organ", "bell"]))
        pad = role == "chords" and rng.random() < 0.4
        self.a = float(rng.uniform(0.15, 0.8)) if pad else float(rng.choice([0.002, 0.005, 0.01, 0.03, 0.08]))
        self.d = float(rng.uniform(0.05, 0.6))
        self.s = float(rng.uniform(0.2, 1.0)) if self.wave not in ("pluck", "bell") else 0.0
        self.r = float(rng.uniform(0.05, 1.2 if pad else 0.5))
        self.cut = float(rng.uniform(800, 9000))
        self.detune = float(rng.uniform(0.05, 0.25))
        self.fm_ratio = float(rng.choice([1.0, 2.0, 3.0, 3.5, 0.5]))
        self.fm_index = float(rng.uniform(0.5, 4))
        self.octave_up = rng.random() < 0.25
        self.sub = rng.random() < 0.12
        self.vibrato = float(rng.uniform(0, 0.004)) if rng.random() < 0.3 else 0.0

    def osc(self, f: float, n: int, rng) -> np.ndarray:
        t = np.arange(n) / SR
        ph = 2 * np.pi * f * t
        if self.vibrato:
            ph = ph + 2 * np.pi * f * self.vibrato * np.sin(2 * np.pi * 5.5 * t) / 5.5
        w = self.wave
        dt = f / SR
        if w == "saw":
            x = blep_saw(ph / (2 * np.pi), dt)
        elif w == "square":
            x = 0.7 * (blep_saw(ph / (2 * np.pi), dt) - blep_saw(ph / (2 * np.pi) + 0.5, dt))
        elif w == "tri":
            x = 2 * np.abs(2 * ((ph / (2 * np.pi)) % 1) - 1) - 1
        elif w == "sine":
            x = np.sin(ph)
        elif w == "fm":
            x = np.sin(ph + self.fm_index * np.exp(-t * 3) * np.sin(ph * self.fm_ratio))
        elif w == "supersaw":
            x = sum(blep_saw(ph * (1 + self.detune * 0.01 * k) / (2 * np.pi) + rng.random(), dt * (1 + self.detune * 0.01 * k)) for k in (-2, -1, 0, 1, 2)) / 3
        elif w == "pluck":
            x = blep_saw(ph / (2 * np.pi), dt) * np.exp(-t * 6)
        elif w == "organ":
            x = np.sin(ph) + 0.5 * np.sin(2 * ph) + 0.3 * np.sin(3 * ph) + 0.2 * np.sin(4 * ph)
        else:  # bell
            x = np.sin(ph) + 0.6 * np.sin(ph * 2.76) * np.exp(-t * 4) + 0.4 * np.sin(ph * 5.4) * np.exp(-t * 7)
        if self.octave_up:
            x = x + 0.4 * np.sin(2 * ph)
        if self.sub:
            x = x + 0.4 * np.sin(ph / 2)
        return x.astype(np.float32)

    def render(self, pitch: int, dur: float, vel: float, rng) -> np.ndarray:
        f = 440 * 2 ** ((pitch - 69) / 12)
        tail = self.r + 0.05
        n = int((dur + tail) * SR)
        x = self.osc(f, n, rng) * adsr(n, self.a, self.d, self.s, self.r, int(dur * SR))
        if self.wave in ("saw", "square", "supersaw", "pluck"):
            b, a = butter(2, min(self.cut, SR / 2 - 100) / (SR / 2))
            x = lfilter(b, a, x)
        return (x * vel).astype(np.float32)


class Sampled:
    """FluidR3 GM (CC BY 3.0) · tonejs 녹음: 가장 가까운 녹음 음을 재생 속도로 옮김."""

    def __init__(self, bank: dict[int, np.ndarray], rng):
        self.bank = bank
        self.keys = np.array(sorted(bank))

    def render(self, pitch: int, dur: float, vel: float, rng) -> np.ndarray:
        k = int(self.keys[np.argmin(np.abs(self.keys - pitch))])
        src = self.bank[k]
        ratio = 2 ** ((pitch - k) / 12)
        n_out = int(min(len(src) / ratio, (dur + 0.4) * SR))
        idx = np.arange(n_out) * ratio
        x = np.interp(idx, np.arange(len(src)), src).astype(np.float32)
        fade = np.ones(n_out, dtype=np.float32)
        off = int(dur * SR)
        if off < n_out:
            fade[off:] = np.exp(-np.arange(n_out - off) / (0.08 * SR))
        return x * fade * vel


# ---------------------------------------------------------------- 효과


def reverb(x: np.ndarray, rng) -> np.ndarray:
    rt = float(rng.uniform(0.3, 3.0))
    n = int(rt * SR)
    ir = rng.normal(0, 1, n).astype(np.float32) * np.exp(-6.9 * np.arange(n) / n)
    b, a = butter(1, float(rng.uniform(2000, 8000)) / (SR / 2))
    ir = lfilter(b, a, ir).astype(np.float32)
    ir /= np.sqrt((ir ** 2).sum()) + 1e-9
    wet = fftconvolve(x, ir)[: len(x)]
    mix = float(rng.uniform(0.1, 0.5))
    return ((1 - mix) * x + mix * wet).astype(np.float32)


def delay(x: np.ndarray, bpm: float, rng) -> np.ndarray:
    d = int(60 / bpm * float(rng.choice([0.5, 0.75, 1.0, 0.25])) * SR)
    fb, mix = float(rng.uniform(0.2, 0.55)), float(rng.uniform(0.1, 0.35))
    out = x.copy()
    echo = x * mix
    k = d
    while k < len(x) and np.abs(echo).max() > 1e-3:
        out[k:] += echo[: len(x) - k]
        echo = echo * fb
        k += d
    return out


def chorus(x: np.ndarray, rng) -> np.ndarray:
    n = len(x)
    lfo = (0.003 + 0.002 * np.sin(2 * np.pi * float(rng.uniform(0.3, 1.5)) * np.arange(n) / SR)) * SR
    idx = np.clip(np.arange(n) - lfo, 0, n - 1)
    return (0.7 * x + 0.5 * np.interp(idx, np.arange(n), x)).astype(np.float32)


def lofi(x: np.ndarray, rng) -> np.ndarray:
    b, a = butter(2, float(rng.uniform(1500, 5000)) / (SR / 2))
    y = lfilter(b, a, x)
    if rng.random() < 0.5:
        y = np.round(y * 64) / 64  # 비트 깎기
    wow = 1 + 0.002 * np.sin(2 * np.pi * 0.5 * np.arange(len(y)) / SR)
    y = np.interp(np.clip(np.cumsum(wow) - 1, 0, len(y) - 1), np.arange(len(y)), y)
    return y.astype(np.float32)


def drums(n: int, bpm: float, rng) -> np.ndarray:
    out = np.zeros(n, dtype=np.float32)
    beat = int(60 / bpm * SR)
    t = np.arange(int(0.25 * SR)) / SR
    kick = np.sin(2 * np.pi * (50 + 120 * np.exp(-t * 30)) * t) * np.exp(-t * 12)
    snare = rng.normal(0, 0.5, len(t)) * np.exp(-t * 25)
    hat = rng.normal(0, 0.3, int(0.05 * SR)) * np.exp(-np.arange(int(0.05 * SR)) / SR * 80)
    b, a = butter(2, 7000 / (SR / 2), "high")
    hat = lfilter(b, a, hat)
    for i in range(0, n, beat):
        if (i // beat) % 2 == 0:
            out[i : i + len(kick)] += kick[: n - i]
        else:
            out[i : i + len(snare)] += snare[: n - i]
        for h in (i, i + beat // 2):
            if h < n:
                out[h : h + len(hat)] += hat[: n - h]
    return out


# ---------------------------------------------------------------- 묶기


_BANKS: list[dict[int, np.ndarray]] = []


def load_banks(folder: Path) -> None:
    """folder/<악기>/<midi번호>.wav 들 (fetch_samples.py가 만든 것)."""
    import soundfile as sf

    for d in sorted(p for p in folder.iterdir() if p.is_dir()):
        bank = {}
        for f in d.glob("*.wav"):
            x, sr = sf.read(f, dtype="float32")
            if x.ndim > 1:
                x = x.mean(1)
            bank[int(f.stem)] = x
        if len(bank) >= 3:
            _BANKS.append(bank)


def make_loop(rng, music=None) -> tuple[np.ndarray, list[tuple[float, float, int]]]:
    """music: (rng) → (bpm, 초, 파트) — 없으면 make_music(무작위 곡)."""
    bpm, seconds, parts = (music or make_music)(rng)
    n = int(seconds * SR)
    mix = np.zeros(n + SR, dtype=np.float32)
    labels = []
    for part in parts:
        inst = Sampled(_BANKS[int(rng.integers(0, len(_BANKS)))], rng) if _BANKS and rng.random() < SAMPLED_SHARE else Synth(rng, part["role"])
        buf = np.zeros(n + SR, dtype=np.float32)
        jitter = float(rng.uniform(0, 0.012))
        for sb, db, p, v in part["notes"]:
            s = sb * 60 / bpm + rng.normal(0, jitter)
            s = max(0.0, s)
            d = db * 60 / bpm
            if s >= seconds:
                continue
            x = inst.render(p, d, v, rng)
            i = int(s * SR)
            m = min(len(x), len(buf) - i)
            buf[i : i + m] += x[:m]
            labels.append((s, min(s + d, seconds), p))
        peak = np.abs(buf).max()
        if peak > 0:
            buf /= peak
        if rng.random() < 0.3:
            buf = chorus(buf, rng)
        if rng.random() < 0.4:
            buf = delay(buf, bpm, rng)
        if rng.random() < 0.6:
            buf = reverb(buf, rng)
        if rng.random() < 0.3:
            buf = np.tanh(buf * float(rng.uniform(1.5, 4))).astype(np.float32)
        mix += buf * float(rng.uniform(0.3, 1.0))
    if rng.random() < 0.25:
        mix += drums(len(mix), bpm, rng) * float(rng.uniform(0.2, 0.8))
    if rng.random() < 0.2:
        mix = lofi(mix, rng)
    if rng.random() < 0.15:
        mix += rng.normal(0, float(rng.uniform(0.002, 0.02)), len(mix)).astype(np.float32)
    mix = mix[:n]
    peak = np.abs(mix).max()
    mix = (mix / peak * float(rng.uniform(0.3, 0.95))) if peak > 0 else mix
    return mix.astype(np.float32), labels


if __name__ == "__main__":
    import soundfile as sf
    import sys

    rng = np.random.default_rng(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(".")
    for i in range(4):
        a, lab = make_loop(rng)
        sf.write(out / f"syn_{i}.wav", a, SR)
        print(i, f"{len(a) / SR:.1f}s", len(lab), "notes", sorted({p for _, _, p in lab})[:8])
