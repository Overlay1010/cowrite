"""Surge XT(무료 신스, GPL-3.0) 프리셋으로 학습용 루프를 미리 렌더한다 (2026-11-22).

synth_data.py의 단순 신스보다 실제 샘플팩에 가까운 소리를 내려고 만들었다. 곡 내용은 synth_data.make_music(무작위)과
cymatics_data.midi_music(샘플팩 MIDI의 실제 코드 · 멜로디)을 섞는다. 정답은 친 MIDI 음뿐이다(프리셋 효과 · 잔향은 무시하도록).

Surge XT는 포터블 zip을 풀어 쓴다(https://github.com/surge-synthesizer/releases-xt, 설치 없음). pedalboard(GPL-3.0)로 VST3를
불러 MIDI를 렌더하고, 프리셋(.fxp)은 플러그인 상태(JUCE VST3 상태 안의 Surge 패치)를 바꿔 끼워 불러온다.
렌더한 오디오는 학습 데이터일 뿐 앱에 들어가지 않는다.

  python surge_render.py scan  <Surge Synth Team 폴더> presets.json          # 프리셋마다 C4를 쳐 보고 쓸 것만 고른다
  python surge_render.py render <Surge Synth Team 폴더> presets.json out --cym cym.pkl --loops 3000 --procs 4
"""
from __future__ import annotations

import argparse
import json
import re
import struct
import sys
import time
from multiprocessing import Process
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import synth_data as G  # noqa: E402

SR = 22050
# 한 음을 쳐서 그 음이 나는 것만 쓴다 — 효과음 · 드럼 · 시퀀스 · 코드(한 건반에 여러 음) · 키 분할 프리셋은 뺀다
ROLES = {
    "bass": {"Basses", "Bass"},
    "chords": {"Pads", "Keys", "Polysynths", "Plucks", "Synths", "Strings", "Organs", "Brass", "Guitars"},
    "melody": {"Leads", "Plucks", "Keys", "Bells", "Mallets", "Winds", "Synths", "Brass", "Guitars", "Strings"},
}
_AL = ".ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+"  # JUCE MemoryBlock base64
_ALB = np.frombuffer(_AL.encode(), dtype=np.uint8)


def _jdec(s: str) -> bytes:
    n, _, enc = s.partition(".")
    val = 0
    for i, ch in enumerate(enc):
        val |= _AL.index(ch) << (6 * i)
    return val.to_bytes((6 * len(enc) + 7) // 8, "little")[: int(n)]


def _jenc(b: bytes) -> str:
    """6비트씩 낮은 비트부터 — 3바이트(24비트)가 4글자. 큰 정수로 한 번에 하면 큰 프리셋(웨이브테이블 1MB)에서 너무 느리다."""
    n = len(b)
    pad = (-n) % 3
    a = np.frombuffer(b + bytes(pad), dtype=np.uint8).reshape(-1, 3).astype(np.uint32)
    v = a[:, 0] | (a[:, 1] << 8) | (a[:, 2] << 16)
    idx = np.stack([(v >> s) & 63 for s in (0, 6, 12, 18)], 1).ravel()[: (n * 8 + 5) // 6]
    return f"{n}." + bytes(_ALB[idx]).decode()


class Surge:
    def __init__(self, folder: Path):
        from pedalboard import load_plugin

        self.p = load_plugin(str(folder / "Surge XT.vst3" / "Contents" / "x86_64-win" / "Surge XT.vst3"))
        st = self.p.raw_state
        size = struct.unpack("<I", st[4:8])[0]
        self.xml, self.tail = st[8 : 8 + size].decode(), st[8 + size :]
        self.m = re.search(r"<IComponent>(.*?)</IComponent>", self.xml)
        assert _jdec(self.m.group(1))[:4] == b"sub3"

    def load(self, fxp: Path) -> None:
        body = fxp.read_bytes()[60:]  # fxChunkSetCustom 머리(60바이트) 뒤가 Surge 패치('sub3')
        x = (self.xml[: self.m.start(1)] + _jenc(body) + self.xml[self.m.end(1) :]).encode()
        self.p.raw_state = b"VC2!" + struct.pack("<I", len(x)) + x + self.tail

    def render(self, notes: list[tuple[float, float, int, float]], seconds: float) -> np.ndarray:
        """notes: (시작 초, 끝 초, 음, 세기 0~1)."""
        from mido import Message

        ev = []
        for s, e, p, v in notes:
            ev.append((s, 1, Message("note_on", note=int(p), velocity=int(np.clip(v * 127, 1, 127)), time=s)))
            ev.append((e, 0, Message("note_off", note=int(p), velocity=0, time=e)))
        ev.sort(key=lambda t: (t[0], t[1]))
        out = self.p([m for _, _, m in ev], duration=seconds, sample_rate=SR, num_channels=2, reset=True)
        return out.mean(0).astype(np.float32)


def presets(folder: Path) -> list[tuple[Path, str]]:
    data = folder / "SurgeXTData"
    out = []
    for f in sorted(data.rglob("*.fxp")):
        rel = f.relative_to(data).parts
        if rel[0] == "patches_factory":
            cat = rel[1]
        elif rel[0] == "patches_3rdparty" and len(rel) > 3:
            cat = rel[2]
        else:
            continue
        if any(cat in v for v in ROLES.values()):
            out.append((f, cat))
    return out


def scan(folder: Path, out: Path) -> None:
    """C2 · C4 · C5를 하나씩 쳐서 (1) 소리가 나고 (2) 스펙트럼의 가장 센 음이름이 친 음(C)이거나 그 5도(배음)인 것만 남긴다."""
    s = Surge(folder)
    good = []
    for i, (f, cat) in enumerate(presets(folder)):
        try:
            s.load(f)
            ok = True
            for p in (36, 60, 72) if cat not in ("Basses", "Bass") else (28, 36, 48):
                a = s.render([(0.05, 0.85, p, 0.8)], 1.2)
                seg = a[int(0.1 * SR) : int(0.8 * SR)]
                if not np.isfinite(seg).all() or np.sqrt((seg**2).mean()) < 1e-3:
                    ok = False
                    break
                spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg))))
                freqs = np.fft.rfftfreq(len(seg), 1 / SR)
                band = (freqs > 25) & (freqs < 5000)
                midi = np.round(69 + 12 * np.log2(freqs[band] / 440)).astype(int)
                chroma = np.bincount(midi % 12, weights=spec[band] ** 2, minlength=12)
                if int(chroma.argmax()) not in (p % 12, (p + 7) % 12) or chroma[p % 12] < 0.25 * chroma.sum():
                    ok = False
                    break
        except Exception as e:  # 깨진 프리셋
            print("건너뜀", f.name, e, flush=True)
            ok = False
        if ok:
            good.append({"path": str(f), "cat": cat})
        if i % 100 == 0:
            print(f"{i} 개 확인 · 쓸 것 {len(good)}", flush=True)
    json.dump(good, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=0)
    print("쓸 프리셋", len(good))


def _worker(wid: int, folder: str, preset_file: str, out: str, cym: str | None, loops: int, midi_share: float) -> None:
    rng = np.random.default_rng(wid * 7919 + int(time.time()) % 100000)
    surge = Surge(Path(folder))
    good = json.load(open(preset_file, encoding="utf-8"))
    by_role = {r: [g for g in good if g["cat"] in cats] for r, cats in ROLES.items()}
    if cym:
        import cymatics_data as C

        C.load(Path(cym))
    outdir = Path(out)
    for k in range(loops):
        name = outdir / f"w{wid}_{k:05d}"
        if name.with_suffix(".npz").exists():
            continue
        music = C.midi_music if cym and rng.random() < midi_share else G.make_music
        bpm, seconds, parts = music(rng)
        seconds = min(seconds, 24.0)
        n = int(seconds * SR)
        mix = np.zeros(n, dtype=np.float32)
        labels = []
        for part in parts:
            role = "bass" if part["role"] == "bass" else "chords" if part["role"] in ("chords", "arp") else "melody"
            pick = by_role[role][int(rng.integers(0, len(by_role[role])))]
            surge.load(Path(pick["path"]))
            jitter = float(rng.uniform(0, 0.01))
            notes = []
            for sb, db, p, v in part["notes"]:
                s = max(0.0, sb * 60 / bpm + rng.normal(0, jitter))
                e = min(s + db * 60 / bpm, seconds - 0.01)
                if s < seconds - 0.05 and e > s:
                    notes.append((s, e, p, v))
            if not notes:
                continue
            buf = surge.render(notes, seconds)[:n]
            peak = np.abs(buf).max()
            if not np.isfinite(buf).all() or peak < 1e-4:
                continue
            buf /= peak
            if rng.random() < 0.25:
                buf = G.reverb(buf, rng)[:n]
            if rng.random() < 0.2:
                buf = G.delay(buf, bpm, rng)[:n]
            mix[: len(buf)] += buf * float(rng.uniform(0.3, 1.0))
            labels += [(s, e, p) for s, e, p, _ in notes]
        if not labels:
            continue
        if rng.random() < 0.3:
            mix += G.drums(n, bpm, rng)[:n] * float(rng.uniform(0.2, 0.8))
        if rng.random() < 0.15:
            mix = G.lofi(mix, rng)[:n]
        peak = np.abs(mix).max()
        mix = mix / peak * float(rng.uniform(0.3, 0.95)) if peak > 0 else mix
        np.savez_compressed(name.with_suffix(".npz"), audio=mix.astype(np.float16), labels=np.array(labels, dtype=np.float32))
        if k % 50 == 0:
            print(f"작업자 {wid}: {k}/{loops}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["scan", "render"])
    ap.add_argument("surge")
    ap.add_argument("presets")
    ap.add_argument("out", nargs="?")
    ap.add_argument("--cym", default=None)
    ap.add_argument("--loops", type=int, default=3000)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--midi_share", type=float, default=0.6)
    a = ap.parse_args()
    if a.cmd == "scan":
        scan(Path(a.surge), Path(a.presets))
        return
    Path(a.out).mkdir(parents=True, exist_ok=True)
    per = (a.loops + a.procs - 1) // a.procs
    ps = [Process(target=_worker, args=(w, a.surge, a.presets, a.out, a.cym, per, a.midi_share)) for w in range(a.procs)]
    for p in ps:
        p.start()
    for p in ps:
        p.join()


if __name__ == "__main__":
    main()
