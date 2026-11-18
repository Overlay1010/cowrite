"""오디오 → MIDI 평가 (LLM 없음). `python -m eval.transcribe` (backend에서)

샘플팩(samples/private, 저장소에 없음)에서 같은 이름의 루프 wav와 mid를 짝지어 mid를 정답으로 쓴다.
- 미디는 템포가 120으로 박힌 게 많아 박(틱 ÷ 해상도)으로 비교하고, 오디오는 초 × 이름의 BPM ÷ 60.
- 완성 루프 wav는 여러 악기가 섞이고 mid는 그중 한 악기라서 정답 음을 찾은 비율(재현율)만 본다.
- 같은 폴더의 악기별 stem wav는 하나씩 바꿔 mid와 가장 잘 맞는 stem의 F1 (mid 길이 안 음표만).
  샘플팩 신스는 mid보다 한두 옥타브 위아래로 울리는 게 많아 옥타브를 옮겨 맞춘 F1도 함께.
- 조성: 변환한 완성 루프 / 정답 mid를 같은 엔진으로 읽어 이름의 조와 비교.
- 템포: 이름의 BPM을 가리고 짐작한 값.
"""

from __future__ import annotations

import io
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

from app.harmony.analyze import SHARP, analyze
from app.midi_io import transcribe as T
from app.midi_io.convert import MidiParseError, midi_bytes_to_project
from eval.harmony import MIREX, key_relation

PRIVATE = Path(__file__).resolve().parents[2] / "samples" / "private"
TOL_SECONDS = 0.05  # 시작점이 이 안이면 맞음 (mir_eval 기본값)
KEY = re.compile(r"([A-G])([#b]?) (Maj|Min)\s*$", re.IGNORECASE)


def _long(p: Path) -> Path:
    """Windows 경로 길이 제한(260자)을 넘는 샘플팩 폴더가 있다."""
    return Path("\\\\?\\" + str(p.resolve())) if os.name == "nt" else p


def key_of_name(name: str) -> tuple[int, str] | None:
    m = KEY.search(name)
    if not m:
        return None
    pc = SHARP.index(m[1].upper()) + (1 if m[2] == "#" else -1 if m[2] == "b" else 0)
    return pc % 12, "major" if m[3].lower() == "maj" else "minor"


def pairs(root: Path) -> list[tuple[Path, Path]]:
    """(mid, 같은 이름의 wav). macOS가 만든 __MACOSX · ._ 파일은 뺀다."""
    files = [p for p in _long(root).rglob("*") if "__MACOSX" not in p.parts and not p.name.startswith("._")]
    wavs = {p.stem.lower(): p for p in files if p.suffix.lower() == ".wav"}
    return [(m, wavs[m.stem.lower()]) for m in sorted(files) if m.suffix.lower() == ".mid" and m.stem.lower() in wavs]


def match(ref: list[tuple[float, int]], est: list[tuple[float, int]], tol: float) -> tuple[float, float, float]:
    """(F1, 정밀도, 재현율). 시작점이 tol 안 · 같은 음인 쌍을 한 번씩만 짝짓는다."""
    if not ref or not est:
        return 0.0, 0.0, 0.0
    starts = np.array([e[0] for e in est])
    used = np.zeros(len(est), dtype=bool)
    hit = 0
    for rs, rp in ref:
        near = [j for j in np.nonzero(np.abs(starts - rs) <= tol)[0] if not used[j] and est[j][1] == rp]
        if near:
            used[min(near, key=lambda j: abs(starts[j] - rs))] = True
            hit += 1
    p, r = hit / len(est), hit / len(ref)
    return (2 * p * r / (p + r) if hit else 0.0), p, r


def tempo_verdict(est: float, true: float) -> str:
    for k, tag in ((1, "정확"), (0.5, "반"), (2, "두 배")):
        if abs(est - true * k) <= max(1.0, true * k * 0.02):
            return tag
    return "틀림"


def pct(values: list[float]) -> str:
    q = np.percentile(values, [25, 50, 75])
    return " · ".join(f"{v:.2f}" for v in q)


def main() -> None:
    found = pairs(PRIVATE) if PRIVATE.exists() else []
    if not found:
        print("samples/private에 같은 이름의 wav · mid 짝이 없어 건너뜀")
        return
    mix_rec, mix_rec_pc, stem_f1, stem_f1_oct, stem_p, stem_r = [], [], [], [], [], []
    keys: dict[str, Counter] = {"audio": Counter(), "midi": Counter()}
    tempo = Counter()
    seconds = spent = 0.0
    songs = 0
    for mid, wav in found:
        bpm, want = T.bpm_from_name(wav.stem), key_of_name(wav.stem)
        if not bpm:
            continue
        try:
            midi_project = midi_bytes_to_project(mid.read_bytes(), "m")
        except MidiParseError:
            continue
        ref = sorted((n.start, n.pitch) for t in midi_project.tracks if not t.is_drum for n in t.notes)
        if not ref:
            continue
        songs += 1
        tol = TOL_SECONDS * bpm / 60
        end = max(n.start + n.duration for t in midi_project.tracks for n in t.notes) + tol

        def notes_of(path: Path) -> tuple[list[T.NoteEvent], float]:
            nonlocal seconds, spent
            audio = T.load_audio(path.read_bytes())
            t0 = time.perf_counter()
            events = T.transcribe(audio)
            spent += time.perf_counter() - t0
            seconds += len(audio) / T.SAMPLE_RATE
            return events, len(audio) / T.SAMPLE_RATE

        events, dur = notes_of(wav)
        est = [(e.start * bpm / 60, e.pitch) for e in events]
        mix_rec.append(match(ref, est, tol)[2])
        mix_rec_pc.append(match([(s, p % 12) for s, p in ref], [(s, p % 12) for s, p in est], tol)[2])
        tempo[tempo_verdict(T.estimate_bpm(events, dur), bpm)] += 1
        if want:
            audio_key = analyze(T.events_to_project(events, "a", bpm, [])).key
            midi_key = analyze(midi_project).key
            keys["audio"][key_relation((audio_key.tonic, audio_key.mode), want)] += 1
            keys["midi"][key_relation((midi_key.tonic, midi_key.mode), want)] += 1

        stems = [s for s in mid.parent.glob("*.wav") if not s.name.startswith("._")]
        best = best_oct = (0.0, 0.0, 0.0)
        for s in stems:
            ev, _ = notes_of(s)
            e = [(x.start * bpm / 60, x.pitch) for x in ev if x.start * bpm / 60 <= end]
            best = max(best, match(ref, e, tol))
            for octave in (-24, -12, 0, 12, 24):
                best_oct = max(best_oct, match(ref, [(b, p + octave) for b, p in e], tol))
        if stems:
            stem_f1.append(best[0])
            stem_p.append(best[1])
            stem_r.append(best[2])
            stem_f1_oct.append(best_oct[0])
        print(f"  {songs:3d} {wav.stem[:60]}", file=sys.stderr, flush=True)

    print("## 오디오 → MIDI 평가\n")
    print(f"샘플팩 루프 {songs}곡 (같은 이름의 wav · mid 짝, 이름에 BPM), 시작점 ±{TOL_SECONDS * 1000:.0f}ms · 같은 음이면 맞음. 값은 25% · 중간 · 75%\n")
    print("| 무엇 | 곡 | 값 |")
    print("|---|---|---|")
    print(f"| 완성 루프: 정답 음을 찾은 비율 | {len(mix_rec)} | {pct(mix_rec)} |")
    print(f"| 완성 루프: 음이름만 (옥타브 무시) | {len(mix_rec)} | {pct(mix_rec_pc)} |")
    print(f"| 악기별 stem 중 가장 맞는 것 F1 | {len(stem_f1)} | {pct(stem_f1)} |")
    print(f"| └ 정밀도 · 재현율 (중간) | {len(stem_f1)} | {np.median(stem_p):.2f} · {np.median(stem_r):.2f} |")
    print(f"| 악기별 stem, 옥타브를 옮겨 맞춘 F1 | {len(stem_f1)} | {pct(stem_f1_oct)} |")
    for k, c in keys.items():
        n = sum(c.values())
        score = sum(MIREX[r] * v for r, v in c.items()) / n
        label = "변환한 완성 루프" if k == "audio" else "정답 mid"
        print(f"\n조성 — {label}: 정답 {c['정답']}/{n} · MIREX {score:.2f} · " + " · ".join(f"{r} {v}" for r, v in c.most_common() if r != "정답"))
    print(f"\n템포 짐작 (이름을 가리고): " + " · ".join(f"{r} {tempo[r]}" for r in ("정확", "반", "두 배", "틀림")) + f" / {sum(tempo.values())}")
    print(f"\n속도: 오디오 1초에 {spent / seconds:.3f}초 (모델 + 음표 묶기, 파일 읽기 빼고)")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
