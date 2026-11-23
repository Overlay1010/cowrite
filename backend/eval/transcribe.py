"""오디오 → MIDI 평가 (LLM 없음). `python -m eval.transcribe [모델.onnx]` (backend에서, 모델을 주면 앱 모델 대신 그걸로)

샘플팩(samples/private, 저장소에 없음)에서 같은 이름의 루프 wav와 mid를 짝지어 mid를 정답으로 쓴다.
- 정답 mid가 소리와 안 맞는 곡(파트가 stem에 없음 · 루프의 다른 자리 · 옥타브나 조가 옮겨짐)을 원래 Basic Pitch로 먼저 가려
  (eval/labels.py) stem 지표는 정답이 맞는 곡만을 주 지표로, 모든 곡은 앞선 기록과 비교용으로 함께 낸다.
- 미디는 템포가 120으로 박힌 게 많아 박(틱 ÷ 해상도)으로 비교하고, 오디오는 초 × 이름의 BPM ÷ 60.
- 완성 루프 wav는 여러 악기가 섞이고 mid는 그중 한 악기라서 정답 음을 찾은 비율(재현율)만 본다.
- 같은 폴더의 악기별 stem wav는 하나씩 바꿔 mid와 가장 잘 맞는 stem의 F1 (mid 길이 안 음표만).
  샘플팩 신스는 mid보다 한두 옥타브 위아래로 울리는 게 많아 옥타브를 옮겨 맞춘 F1도 함께.
- 코드: 정답 mid와 변환 결과를 같은 화성 분석(app/harmony)으로 읽어 박마다 근음이 같은 비율 (완성 루프 · 가장 맞는 stem).
- 조성: 변환한 완성 루프 / 정답 mid를 같은 엔진으로 읽어 이름의 조와 비교.
- 템포: 이름의 BPM을 가리고 짐작한 값. 루프가 아닌 녹음 대신 완성 루프의 끝을 잘라(70%) · 앞을 잘라(0.3마디) 다시 바꿔 본다.
- 앱과 같게 이름의 템포로 16분 격자에 맞춘 뒤 잰다(대부분 격자에 붙은 파일만 맞춘다).
- 음표 묶기를 고칠 때는 곡 이름으로 나눈 A/B 두 묶음에서 모두 오르는지 본다 (한쪽에 맞춘 조정 막기).
"""

from __future__ import annotations

import os
import zlib
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

from app.harmony.analyze import SHARP, analyze
from app.models.project import Note, Project, Track
from app.midi_io import transcribe as T
from app.midi_io.convert import MidiParseError, midi_bytes_to_project
from eval.harmony import MIREX, key_relation
from eval.labels import judge_all

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
    return " · ".join(f"{v:.2f}" for v in q) + f" | {np.mean(values):.2f}"


def group(name: str) -> str:
    """곡 이름으로 나눈 A/B 묶음 (고칠 때 한쪽에서 고르고 다른 쪽에서 확인)."""
    return "AB"[zlib.crc32(name.encode()) % 2]


def chord_roots_agree(ref: Project, got: Project, end: float) -> float | None:
    """박마다(박 가운데) 두 분석의 코드 근음이 같은 비율. 정답이 화음 없음(N.C.)인 박은 뺀다."""
    a, b = analyze(ref).chords, analyze(got).chords
    at = lambda cs, t: next((c for c in cs if c.start <= t < c.end), None)  # noqa: E731
    same = n = 0
    for t in np.arange(0.5, end, 1.0):
        rc = at(a, t)
        if rc is None or rc.root is None:
            continue
        n += 1
        gc = at(b, t)
        same += gc is not None and gc.root == rc.root
    return same / n if n else None


def stems_of(mid: Path) -> list[Path]:
    return sorted(s for s in mid.parent.glob("*.wav") if not s.name.startswith("._"))


def songs_of(found: list[tuple[Path, Path]]) -> list[tuple[Path, Path, float, Project, list[Note]]]:
    """(mid, wav, 이름의 BPM, 정답 프로젝트, 드럼 뺀 정답 음표) — 이름에 BPM이 없거나 음이 없는 짝은 뺀다."""
    out = []
    for mid, wav in found:
        bpm = T.bpm_from_name(wav.stem)
        if not bpm:
            continue
        try:
            midi_project = midi_bytes_to_project(mid.read_bytes(), "m")
        except MidiParseError:
            continue
        ref_notes = [n for t in midi_project.tracks if not t.is_drum for n in t.notes]
        if ref_notes:
            out.append((mid, wav, bpm, midi_project, ref_notes))
    return out


def main() -> None:
    found = pairs(PRIVATE) if PRIVATE.exists() else []
    if not found:
        print("samples/private에 같은 이름의 wav · mid 짝이 없어 건너뜀")
        return
    songs_list = songs_of(found)
    verdicts = judge_all(
        [(mid, [(n.start, n.duration, n.pitch) for n in notes], stems_of(mid), bpm, TOL_SECONDS * bpm / 60) for mid, _, bpm, _, notes in songs_list],
        _long(PRIVATE) / ".label_check.json",
    )
    mix_rec, mix_rec_pc, stem_f1, stem_f1_oct, stem_p, stem_r = [], [], [], [], [], []
    chords: dict[str, list[float]] = {"mix": [], "stem": []}
    f1_by_group: dict[str, list[float]] = {"A": [], "B": []}
    clean: dict[str, list[float]] = {"stem": [], "oct": [], "mix": [], "chord": [], "A": [], "B": []}
    keys: dict[str, Counter] = {"audio": Counter(), "midi": Counter()}
    tempo: dict[str, Counter] = {"원본": Counter(), "끝 자름": Counter(), "앞 자름": Counter()}
    seconds = spent = 0.0
    songs = snapped_files = 0
    for mid, wav, bpm, midi_project, ref_notes in songs_list:
        want = key_of_name(wav.stem)
        ref = sorted((n.start, n.pitch) for n in ref_notes)
        ok = verdicts[str(mid)].clean
        songs += 1
        tol = TOL_SECONDS * bpm / 60
        end = max(n.start + n.duration for n in ref_notes) + tol
        # 정답을 이름의 템포로 (mid에 박힌 템포는 120인 게 많다) — 화성 분석이 마디를 템포와 상관없이 박으로 나누지만 맞춰 둔다
        ref_project = Project(id="r", name="r", tempo=bpm, tracks=[Track(id="t1", name="r", program=0, notes=ref_notes)])

        def run(audio: np.ndarray) -> T.Transcription:
            nonlocal seconds, spent
            t0 = time.perf_counter()
            tr = T.transcribe_audio(audio)
            spent += time.perf_counter() - t0
            seconds += tr.seconds
            return tr

        audio = T.load_audio(wav.read_bytes())
        tr = run(audio)
        events, snapped = T.quantize(tr.events, bpm)  # 앱과 같게: 이름의 템포로 16분 격자에 (대부분 격자에 붙은 파일만)
        snapped_files += snapped
        est = [(e.start * bpm / 60, e.pitch) for e in events]
        mix_rec.append(match(ref, est, tol)[2])
        mix_rec_pc.append(match([(s, p % 12) for s, p in ref], [(s, p % 12) for s, p in est], tol)[2])
        tempo["원본"][tempo_verdict(T.estimate_bpm(tr), bpm)] += 1
        tail = audio[: int(len(audio) * 0.7)]
        head = audio[int(0.3 * 240 / bpm * T.SAMPLE_RATE) :]
        tempo["끝 자름"][tempo_verdict(T.estimate_bpm(T.transcribe_audio(tail)), bpm)] += 1
        tempo["앞 자름"][tempo_verdict(T.estimate_bpm(T.transcribe_audio(head)), bpm)] += 1
        converted = T.events_to_project(events, "a", bpm, [])
        if (c := chord_roots_agree(ref_project, converted, end)) is not None:
            chords["mix"].append(c)
        if want:
            audio_key = analyze(converted).key
            midi_key = analyze(midi_project).key
            keys["audio"][key_relation((audio_key.tonic, audio_key.mode), want)] += 1
            keys["midi"][key_relation((midi_key.tonic, midi_key.mode), want)] += 1

        stems = stems_of(mid)
        best = best_oct = (0.0, 0.0, 0.0)
        best_events: list[T.NoteEvent] = []
        for s in stems:
            ev = T.quantize(run(T.load_audio(s.read_bytes())).events, bpm)[0]
            e = [(x.start * bpm / 60, x.pitch) for x in ev if x.start * bpm / 60 <= end]
            m = match(ref, e, tol)
            if m > best:
                best, best_events = m, ev
            for octave in (-24, -12, 0, 12, 24):
                best_oct = max(best_oct, match(ref, [(b, p + octave) for b, p in e], tol))
        if stems:
            stem_f1.append(best[0])
            stem_p.append(best[1])
            stem_r.append(best[2])
            stem_f1_oct.append(best_oct[0])
            f1_by_group[group(wav.stem)].append(best[0])
            sc = chord_roots_agree(ref_project, T.events_to_project(best_events, "s", bpm, []), end) if best_events else None
            if sc is not None:
                chords["stem"].append(sc)
            if ok:
                clean["stem"].append(best[0])
                clean["oct"].append(best_oct[0])
                clean["mix"].append(mix_rec[-1])
                clean[group(wav.stem)].append(best[0])
                if sc is not None:
                    clean["chord"].append(sc)
        print(f"  {songs:3d} {wav.stem[:60]}", file=sys.stderr, flush=True)

    print("## 오디오 → MIDI 평가\n")
    print(f"샘플팩 루프 {songs}곡 (같은 이름의 wav · mid 짝, 이름에 BPM), 시작점 ±{TOL_SECONDS * 1000:.0f}ms · 같은 음이면 맞음. 값은 25% · 중간 · 75% | 평균\n")
    why = Counter(r for v in verdicts.values() for r in v.reasons)
    n_stem = sum("stem 없음" not in v.reasons for v in verdicts.values())
    print(f"정답 확인 (원래 Basic Pitch로, eval/labels.py): stem 있는 {n_stem}곡 중 정답이 소리와 맞는 곡 {len(clean['stem'])} — "
          + " · ".join(f"{r} {why[r]}" for r in ("안 들림", "자리 어긋남", "조 · 옥타브 다름")) + " (한 곡이 여럿에 걸리기도)\n")
    print("**주 지표 — 정답이 소리와 맞는 곡만**\n")
    print("| 무엇 | 곡 | 값 |")
    print("|---|---|---|")
    print(f"| 악기별 stem 중 가장 맞는 것 F1 | {len(clean['stem'])} | {pct(clean['stem'])} |")
    print(f"| └ 묶음 A · B 평균 | {len(clean['A'])} · {len(clean['B'])} | {np.mean(clean['A']):.3f} · {np.mean(clean['B']):.3f} |")
    print(f"| 악기별 stem, 옥타브를 옮겨 맞춘 F1 | {len(clean['oct'])} | {pct(clean['oct'])} |")
    print(f"| 완성 루프: 정답 음을 찾은 비율 | {len(clean['mix'])} | {pct(clean['mix'])} |")
    print(f"| 코드 근음이 정답과 같은 박 (가장 맞는 stem) | {len(clean['chord'])} | {pct(clean['chord'])} |")
    print("\n**모든 곡** (정답이 소리와 안 맞는 곡 포함 — 앞선 기록과 비교용)\n")
    print("| 무엇 | 곡 | 값 |")
    print("|---|---|---|")
    print(f"| 완성 루프: 정답 음을 찾은 비율 | {len(mix_rec)} | {pct(mix_rec)} |")
    print(f"| 완성 루프: 음이름만 (옥타브 무시) | {len(mix_rec)} | {pct(mix_rec_pc)} |")
    print(f"| 악기별 stem 중 가장 맞는 것 F1 | {len(stem_f1)} | {pct(stem_f1)} |")
    print(f"| └ 정밀도 · 재현율 (중간) | {len(stem_f1)} | {np.median(stem_p):.2f} · {np.median(stem_r):.2f} |")
    print(f"| └ 묶음 A · B 평균 | {len(f1_by_group['A'])} · {len(f1_by_group['B'])} | {np.mean(f1_by_group['A']):.3f} · {np.mean(f1_by_group['B']):.3f} |")
    print(f"| 악기별 stem, 옥타브를 옮겨 맞춘 F1 | {len(stem_f1)} | {pct(stem_f1_oct)} |")
    print(f"| 16분 격자에 맞춘 완성 루프 | {songs} | {snapped_files}곡 |")
    print(f"| 코드 근음이 정답과 같은 박 (완성 루프) | {len(chords['mix'])} | {pct(chords['mix'])} |")
    print(f"| 코드 근음이 정답과 같은 박 (가장 맞는 stem) | {len(chords['stem'])} | {pct(chords['stem'])} |")
    for k, c in keys.items():
        n = sum(c.values())
        score = sum(MIREX[r] * v for r, v in c.items()) / n
        label = "변환한 완성 루프" if k == "audio" else "정답 mid"
        print(f"\n조성 — {label}: 정답 {c['정답']}/{n} · MIREX {score:.2f} · " + " · ".join(f"{r} {v}" for r, v in c.most_common() if r != "정답"))
    print()
    for k, c in tempo.items():
        print(f"템포 짐작 (이름을 가리고, {k}): " + " · ".join(f"{r} {c[r]}" for r in ("정확", "반", "두 배", "틀림")) + f" / {sum(c.values())}  ")
    print(f"\n속도: 오디오 1초에 {spent / seconds:.3f}초 (모델 + 음표 묶기, 파일 읽기 빼고)")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    if len(sys.argv) > 1:  # 학습한 체크포인트 재기 (backend/train/transcriber)
        T.MODEL_PATH = Path(sys.argv[1])
        T._session.cache_clear()
    print(f"모델: {T.MODEL_PATH.name}\n")
    main()
