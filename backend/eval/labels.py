"""정답 mid가 실제 소리와 맞는지 곡마다 가린다 (eval.transcribe가 씀).

샘플팩 mid가 늘 오디오와 맞지는 않는다 — 그 파트가 stem에 없거나, 루프의 다른 자리(4~8박 뒤)와 맞거나,
옥타브 · 조가 옮겨져 있다. 이런 곡은 어떤 모델로도 점수가 안 나와 모델 비교를 흐린다.

- 재는 모델은 바꾸지 않은 **원래 Basic Pitch(nmp.onnx)로 고정** — 재는 모델이 바뀌어도 같은 곡이 골라진다.
- 곡마다 stem을 하나씩, 시간을 1/32박씩 ±8박 옮겨 보고(같은 점수면 덜 옮긴 쪽) 다시 반음 ±24로 옮겨 정답과 가장 맞는 것을 찾는다.
- 문제로 치는 것 (옮겨서 F1이 0.05 넘게 오를 때만 — 우연히 걸린 옮김을 문제로 세지 않게):
  - 자리 어긋남: 0.5박 넘게 옮겨야 맞음
  - 조 · 옥타브 다름: 반음을 옮겨야 맞음
  - 안 들림: 옮김을 반영해도 정답 음의 절반 넘게 그 시각에 그 음(어느 옥타브든)의 활성도가 0.2 미만
- 결과는 samples/private/.label_check.json에 (파일 크기 · 수정 시각이 같으면 다시 안 잰다).
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from app.midi_io import transcribe as T

JUDGE = Path(T.__file__).parent / "basic_pitch" / "nmp.onnx"
VERSION = 1
SHIFTS = np.arange(-256, 257) / 32  # 박
TRANSPOSE = range(-24, 25)
GAIN = 0.05  # 옮겨서 이만큼 넘게 올라야 문제로 친다
HEARD_ACT = 0.2
MAX_UNHEARD = 0.5


@dataclass
class Verdict:
    clean: bool
    reasons: list[str] = field(default_factory=list)  # "안 들림" · "자리 어긋남" · "조 · 옥타브 다름", stem이 없으면 "stem 없음"
    stem: str = ""
    shift: float = 0.0  # 박
    transpose: int = 0
    unheard: float = 0.0


def _f1(ref: list[tuple[float, int]], est: list[tuple[float, int]], tol: float) -> float:
    if not ref or not est:
        return 0.0
    starts = np.array([e[0] for e in est])
    pitches = np.array([e[1] for e in est])
    used = np.zeros(len(est), dtype=bool)
    hit = 0
    for rs, rp in ref:
        near = np.nonzero((np.abs(starts - rs) <= tol) & (pitches == rp) & ~used)[0]
        if len(near):
            used[near[np.argmin(np.abs(starts[near] - rs))]] = True
            hit += 1
    return 2 * hit / (len(ref) + len(est))


def reasons_of(f1_plain: float, f1_shift: float, shift: float, f1_trans: float, transpose: int, unheard: float) -> list[str]:
    """옮겨 본 결과 → 문제 목록 (빈 목록이면 정답이 소리와 맞음)."""
    out = []
    if unheard > MAX_UNHEARD:
        out.append("안 들림")
    if abs(shift) > 0.5 and f1_shift - f1_plain > GAIN:
        out.append("자리 어긋남")
    if transpose != 0 and f1_trans - f1_shift > GAIN:
        out.append("조 · 옥타브 다름")
    return out


def _judge_song(ref_notes: list[tuple[float, float, int]], stems: list[Path], bpm: float, tol: float) -> Verdict:
    """ref_notes: (시작 박, 길이 박, 음). tol: 박."""
    if not stems:
        return Verdict(clean=False, reasons=["stem 없음"])
    ref = [(s, p) for s, _, p in ref_notes]
    end = max(s + d for s, d, _ in ref_notes) + tol
    best = None  # (점수, -|옮김|, 옮김, stem, 이벤트, 오디오, 옮기지 않은 F1)
    for stem in stems:
        audio = T.load_audio(stem.read_bytes())
        events = [(e.start * bpm / 60, e.pitch) for e in T.quantize(T.transcribe_audio(audio).events, bpm)[0]]
        plain = _f1(ref, [e for e in events if e[0] <= end], tol)
        for d in SHIFTS:
            moved = [(b + d, p) for b, p in events if -tol <= b + d <= end]
            cand = (_f1(ref, moved, tol), -abs(d), float(d), stem, events, audio, plain)
            if best is None or cand[:2] > best[:2]:
                best = cand
    f1_shift, _, shift, stem, events, audio, plain = best
    moved = [(b + shift, p) for b, p in events if -tol <= b + shift <= end]
    f1_trans, transpose = max(((_f1(ref, [(b, p + k) for b, p in moved], tol), -abs(k), k) for k in TRANSPOSE))[0::2]

    notes_act, _ = T.model_output(audio)
    times = T._frame_times(len(notes_act))
    unheard = 0
    for s, d, p in ref_notes:
        t0 = (s - shift) * 60 / bpm
        t1 = max(t0 + min(d * 60 / bpm, 0.3), t0 + 0.05)
        frames = np.nonzero((times >= t0 - 0.03) & (times <= t1))[0]
        cols = [c for c in range((p - transpose - T.MIDI_OFFSET) % 12, T.N_PITCHES, 12)]
        if not len(frames) or notes_act[np.ix_(frames, cols)].max() < HEARD_ACT:
            unheard += 1
    unheard_share = unheard / len(ref_notes)
    reasons = reasons_of(plain, f1_shift, shift, f1_trans, transpose, unheard_share)
    return Verdict(not reasons, reasons, stem.name, shift, transpose, round(unheard_share, 3))


def _signature(paths: list[Path]) -> list[list]:
    return [[p.name, p.stat().st_size, int(p.stat().st_mtime)] for p in sorted(paths)]


def judge_all(songs: list[tuple[Path, list[tuple[float, float, int]], list[Path], float, float]], cache: Path) -> dict[str, Verdict]:
    """songs: (mid, 정답 음표(박), stem들, bpm, tol 박) → 곡(mid 경로)마다 판정. 원래 모델로 잰 뒤 재는 모델을 되돌린다."""
    try:
        saved = json.loads(cache.read_text(encoding="utf-8")) if cache.exists() else {}
    except (OSError, ValueError):
        saved = {}
    if saved.get("version") != VERSION:
        saved = {"version": VERSION, "songs": {}}
    out: dict[str, Verdict] = {}
    todo = []
    for mid, notes, stems, bpm, tol in songs:
        sig = _signature([mid, *stems])
        hit = saved["songs"].get(str(mid))
        if hit and hit["sig"] == sig:
            out[str(mid)] = Verdict(**hit["verdict"])
        else:
            todo.append((mid, notes, stems, bpm, tol, sig))
    if todo:
        model, T.MODEL_PATH = T.MODEL_PATH, JUDGE
        T._session.cache_clear()
        try:
            for i, (mid, notes, stems, bpm, tol, sig) in enumerate(todo, 1):
                v = _judge_song(notes, stems, bpm, tol)
                out[str(mid)] = v
                saved["songs"][str(mid)] = {"sig": sig, "verdict": asdict(v)}
                print(f"  정답 확인 {i}/{len(todo)} {mid.stem[:50]}: {' · '.join(v.reasons) or '맞음'}", flush=True, file=sys.stderr)
        finally:
            T.MODEL_PATH = model
            T._session.cache_clear()
        cache.write_text(json.dumps(saved, ensure_ascii=False, indent=1), encoding="utf-8")
    return out
