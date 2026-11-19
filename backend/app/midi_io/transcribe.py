"""오디오(WAV 등) → 음표. Spotify Basic Pitch 모델(ONNX)로 음 높이 · 시작점을 읽고 음표로 묶는다.

basic-pitch 패키지는 Windows · Python 3.13에서 tensorflow 2.15 이하를 요구해 설치되지 않아서,
모델 파일만 onnxruntime으로 돌리고 창 나누기 · 음표 묶기는 여기 옮겨 왔다.
아래 _windows ~ _notes는 basic-pitch 0.4.0의 inference.py · note_creation.py를 옮긴 것이다
(Copyright 2022 Spotify AB, Apache License 2.0 — basic_pitch/LICENSE · NOTICE). 피치 벤드는 앱에 담을 곳이 없어 뺐다.

쓰는 모델은 basic_pitch/nmp_cowrite.onnx — 원래 nmp.onnx의 뒤쪽 합성곱 6층을 샘플팩 스타일 합성 루프로 이어서 학습한 것이다
(그래프 · 크기 · 속도는 같고 가중치만 다르다, backend/train/transcriber · basic_pitch/CHANGES.md). 샘플팩 90곡에서 섞인 루프
재현율 0.25 → 0.32, 조성 63 → 69/88 (docs/audio-to-midi.md). nmp.onnx는 학습의 출발점으로 남겨 둔다.
"""

from __future__ import annotations

import io
import math
import re
import uuid
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from app.models.project import Note, Project, Track

MODEL_PATH = Path(__file__).parent / "basic_pitch" / "nmp_cowrite.onnx"

SAMPLE_RATE = 22050
FFT_HOP = 256
FPS = SAMPLE_RATE // FFT_HOP  # 86 (모델 출력 한 칸 ≈ 11.6ms)
WINDOW_SAMPLES = SAMPLE_RATE * 2 - FFT_HOP  # 모델 입력 한 창 = 2초
WINDOW_FRAMES = FPS * 2
OVERLAP_FRAMES = 30
OVERLAP = OVERLAP_FRAMES * FFT_HOP
HOP = WINDOW_SAMPLES - OVERLAP
MIDI_OFFSET = 21  # 출력 0번 칸 = A0
N_PITCHES = 88

# Basic Pitch 기본값. 시작점 문턱 0.6은 악기 하나짜리 stem F1을 올리지만(0.26 · 0.28 → 0.28 · 0.32) 여러 악기가 섞인
# 루프의 재현율 · 코드 근음을 깎아서(중간 0.15 → 0.11 · 0.75 → 0.70) 그대로 둔다. 쪼개진 음 합치기 · 같이 시작한 배음 빼기도
# 0.5에서는 효과가 없어 넣지 않았다 (docs/audio-to-midi.md).
ONSET_THRESHOLD = 0.5
FRAME_THRESHOLD = 0.3
MIN_NOTE_MS = 127.7
ENERGY_TOL = 11

# 격자 맞춤: 시작점의 GRID_NEED 이상이 이미 16분 격자 ±0.25칸 안인 파일만, ±GRID_WINDOW칸 안의 시작 · 끝을 격자로.
# 스윙 · 셋잇단 · 자유로운 연주는 이 검사에 안 걸려 그대로 두고, 템포를 잘못 짐작했을 때도 대개 안 걸린다.
# 샘플팩 90곡 중 34곡을 맞추고 점수(음 · 코드 · 조성)는 그대로다.
GRID_NEED = 0.6
GRID_WINDOW = 0.3

MAX_SECONDS = 10 * 60


class AudioParseError(ValueError):
    pass


@dataclass(frozen=True)
class NoteEvent:
    start: float  # 초
    end: float
    pitch: int
    amplitude: float  # 0~1


@lru_cache(maxsize=1)
def _session():
    import onnxruntime as ort

    return ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])


def warm_up() -> None:
    """onnxruntime · scipy · soundfile을 불러오고 모델을 읽는다. 처음 한 번 16초쯤 걸려서(이후 1분짜리 변환이 2초쯤)
    서버가 켜질 때 뒤에서 미리 해 둔다."""
    import scipy.signal  # noqa: F401
    import soundfile  # noqa: F401

    _session()


def load_audio(data: bytes) -> np.ndarray:
    """오디오 파일 바이트 → 22050Hz 모노 float32."""
    import soundfile as sf
    from scipy.signal import resample_poly

    try:
        audio, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    except Exception as e:  # libsndfile이 못 읽는 형식 · 깨진 파일
        raise AudioParseError(f"오디오 파일을 읽을 수 없습니다: {e}") from e
    if not len(audio):
        raise AudioParseError("소리가 없는 파일입니다.")
    if len(audio) / sr > MAX_SECONDS:
        raise AudioParseError(f"오디오는 {MAX_SECONDS // 60}분 이하만 바꿀 수 있습니다.")
    mono = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        g = math.gcd(int(sr), SAMPLE_RATE)
        mono = resample_poly(mono, SAMPLE_RATE // g, int(sr) // g).astype(np.float32)
    return mono


def _windows(audio: np.ndarray) -> np.ndarray:
    padded = np.concatenate([np.zeros(OVERLAP // 2, dtype=np.float32), audio])
    out = []
    for i in range(0, len(padded), HOP):
        w = padded[i : i + WINDOW_SAMPLES]
        out.append(np.pad(w, (0, WINDOW_SAMPLES - len(w))))
    return np.stack(out)[..., None]


def _unwrap(output: np.ndarray, n_samples: int) -> np.ndarray:
    half = OVERLAP_FRAMES // 2
    output = output[:, half:-half, :]
    n_frames = int(np.floor(n_samples * (FPS / SAMPLE_RATE)))
    return output.reshape(-1, output.shape[2])[:n_frames]


def model_output(audio: np.ndarray, batch: int = 16) -> tuple[np.ndarray, np.ndarray]:
    """(음 활성도, 시작점 활성도), 둘 다 (프레임, 88)."""
    windows = _windows(audio)
    session = _session()
    notes, onsets = [], []
    for i in range(0, len(windows), batch):
        note, onset = session.run(
            ["StatefulPartitionedCall:1", "StatefulPartitionedCall:2"],
            {"serving_default_input_2:0": windows[i : i + batch]},
        )
        notes.append(note)
        onsets.append(onset)
    return _unwrap(np.concatenate(notes), len(audio)), _unwrap(np.concatenate(onsets), len(audio))


def _frame_times(n_frames: int) -> np.ndarray:
    original = np.arange(n_frames) * FFT_HOP / SAMPLE_RATE
    window_numbers = np.floor(np.arange(n_frames) / WINDOW_FRAMES)
    window_offset = (FFT_HOP / SAMPLE_RATE) * (WINDOW_FRAMES - WINDOW_SAMPLES / FFT_HOP) + 0.0018  # Basic Pitch의 보정값
    return original - window_offset * window_numbers


def _inferred_onsets(onsets: np.ndarray, frames: np.ndarray, n_diff: int = 2) -> np.ndarray:
    """음 활성도가 크게 뛴 곳도 시작점으로 (다시 친 같은 음)."""
    diffs = []
    for n in range(1, n_diff + 1):
        shifted = np.concatenate([np.zeros((n, frames.shape[1])), frames])
        diffs.append(shifted[n:, :] - shifted[:-n, :])
    frame_diff = np.min(diffs, axis=0)
    frame_diff[frame_diff < 0] = 0
    frame_diff[:n_diff, :] = 0
    if frame_diff.max() > 0:
        frame_diff = onsets.max() * frame_diff / frame_diff.max()
    return np.maximum(onsets, frame_diff)


def _notes(frames: np.ndarray, onsets: np.ndarray, min_note_len: int) -> list[tuple[int, int, int, float]]:
    """활성도 → (시작 프레임, 끝 프레임, 음 번호, 세기). Basic Pitch output_to_notes_polyphonic (melodia trick 포함)."""
    from scipy.signal import argrelmax

    n_frames = frames.shape[0]
    onsets = _inferred_onsets(onsets, frames)
    peaks = np.zeros(onsets.shape)
    idx = argrelmax(onsets, axis=0)
    peaks[idx] = onsets[idx]
    onset_t, onset_f = np.where(peaks >= ONSET_THRESHOLD)
    remaining = frames.copy()
    events = []

    def clear(i: int, f: int) -> None:
        remaining[i, f] = 0
        if f < N_PITCHES - 1:
            remaining[i, f + 1] = 0
        if f > 0:
            remaining[i, f - 1] = 0

    for start, f in zip(onset_t[::-1], onset_f[::-1]):  # 뒤에서부터
        if start >= n_frames - 1:
            continue
        i, k = start + 1, 0
        while i < n_frames - 1 and k < ENERGY_TOL:
            k = k + 1 if remaining[i, f] < FRAME_THRESHOLD else 0
            i += 1
        i -= k
        if i - start <= min_note_len:
            continue
        remaining[start:i, f] = 0
        if f < N_PITCHES - 1:
            remaining[start:i, f + 1] = 0
        if f > 0:
            remaining[start:i, f - 1] = 0
        events.append((int(start), int(i), int(f) + MIDI_OFFSET, float(np.mean(frames[start:i, f]))))

    # melodia trick: 시작점 없이 이어지는 남은 소리도 음표로
    while remaining.max() > FRAME_THRESHOLD:
        mid, f = np.unravel_index(np.argmax(remaining), remaining.shape)
        remaining[mid, f] = 0
        i, k = mid + 1, 0
        while i < n_frames - 1 and k < ENERGY_TOL:
            k = k + 1 if remaining[i, f] < FRAME_THRESHOLD else 0
            clear(i, f)
            i += 1
        end = i - 1 - k
        i, k = mid - 1, 0
        while i > 0 and k < ENERGY_TOL:
            k = k + 1 if remaining[i, f] < FRAME_THRESHOLD else 0
            clear(i, f)
            i -= 1
        start = i + 1 + k
        if end - start <= min_note_len:
            continue
        events.append((int(start), int(end), int(f) + MIDI_OFFSET, float(np.mean(frames[start:end, f]))))
    return events


@dataclass(frozen=True)
class Transcription:
    events: list[NoteEvent]
    onset_env: np.ndarray  # 모델 칸마다 시작점 세기 (88음 합, ±23ms 최댓값, 평균 1) — 템포 짐작용
    times: np.ndarray  # 칸마다 시각(초)
    seconds: float


def transcribe_audio(audio: np.ndarray) -> Transcription:
    """22050Hz 모노 → 음표 (초 단위, 시작 순) + 템포 짐작에 쓸 시작점 세기."""
    from scipy.ndimage import maximum_filter1d

    frames, onsets = model_output(audio)
    min_len = int(np.round(MIN_NOTE_MS / 1000 * FPS))
    times = _frame_times(frames.shape[0])
    events = [
        NoteEvent(start=max(0.0, float(times[s])), end=float(times[e]), pitch=p, amplitude=a)
        for s, e, p, a in _notes(frames, onsets, min_len)
    ]
    env = maximum_filter1d(onsets.sum(axis=1), size=5)
    env = env / env.mean() if env.mean() > 0 else env
    return Transcription(sorted(events, key=lambda n: (n.start, n.pitch)), env, times, len(audio) / SAMPLE_RATE)


def quantize(events: list[NoteEvent], bpm: float) -> tuple[list[NoteEvent], bool]:
    """16분 격자에 맞춘다 — 시작점이 이미 대부분 격자 가까이 있는 파일만. (음들, 맞췄는지)"""
    step = 15 / bpm
    if not events:
        return events, False
    near = sum(abs(e.start / step - round(e.start / step)) <= 0.25 for e in events)
    if near < GRID_NEED * len(events):
        return events, False

    def snap(t: float) -> float:
        g = round(t / step) * step
        return g if abs(t - g) <= GRID_WINDOW * step else t

    out: dict[tuple[int, int], NoteEvent] = {}
    for e in sorted(events, key=lambda n: (n.start, -n.amplitude)):
        s, en = snap(e.start), snap(e.end)
        if en <= s:
            en = s + step
        out.setdefault((round(s / step * 4), e.pitch), NoteEvent(s, en, e.pitch, e.amplitude))  # 같은 자리 · 같은 음은 하나로
    return sorted(out.values(), key=lambda n: (n.start, n.pitch)), True


def transcribe(audio: np.ndarray) -> list[NoteEvent]:
    return transcribe_audio(audio).events


# 이름에 템포가 없을 때 짐작하기 (샘플팩 루프 90곡, 정답 = 이름의 BPM): 정확 77 · 반/두 배 8 · 틀림 5 (예전 규칙 51 · 9 · 30).
# 후보는 (1) 루프 길이 = N마디인 템포 — 샘플팩 루프는 90곡 모두 정확히 정수 마디다(4 · 8 · 16만이 아니라 12 · 24 · 34마디도) —
# 와 (2) 60~190 BPM을 0.5 간격으로, 첫 박 위치도 박의 1/8씩 옮겨 본 자유 템포(마디로 잘리지 않은 녹음).
# 첫 박부터 박 · 뒷박 · 16분 자리에 시작점이 몰리는 정도가 가장 높은 것을 고른다. 이 점수는 반/두 배까지는 잘 맞히지만
# 어느 쪽인지는 못 가려서(느린 쪽을 좋아한다) 90~180 BPM으로 접는다 — 90곡 중 81곡이 이 창에 든다.
TEMPO_LOW, TEMPO_HIGH = 60, 190
TEMPO_FOLD = 90
GRID_WEIGHTS = (1.0, 0.6, 0.3)  # 박 · 뒷박(8분) · 그 사이 16분


def _grid_score(tr: Transcription, bpm: float, phase: float = 0.0) -> float:
    """phase초를 첫 박으로 놓은 bpm 격자에서 박 · 뒷박 · 16분 자리의 시작점 세기 평균 (전체 평균 = 1)."""
    t = np.arange(phase, tr.times[-1], 15 / bpm)
    if not len(t):
        return 0.0
    v = tr.onset_env[np.clip(np.searchsorted(tr.times, t), 0, len(tr.times) - 1)]
    k = np.arange(len(t)) % 4
    parts = (v[k == 0], v[k == 2], v[k % 2 == 1])
    return float(sum(w * p.mean() for w, p in zip(GRID_WEIGHTS, parts) if len(p)) / sum(GRID_WEIGHTS))


def _fold(bpm: float) -> float:
    while bpm < TEMPO_FOLD:
        bpm *= 2
    while bpm >= TEMPO_FOLD * 2:
        bpm /= 2
    return bpm


def estimate_tempo(tr: Transcription) -> tuple[float, float]:
    """(템포, 첫 박 위치 초). 루프면 첫 박 = 파일 처음(0)."""
    if not tr.events or len(tr.times) < 2 or tr.seconds <= 0:
        return 120.0, 0.0
    loops = [n * 240 / tr.seconds for n in range(1, int(TEMPO_HIGH * tr.seconds / 240) + 1)]
    cands = [(_grid_score(tr, bpm), bpm, 0.0) for bpm in loops if bpm >= TEMPO_LOW]
    cands += [
        (_grid_score(tr, bpm, phase), bpm, phase)
        for bpm in np.arange(TEMPO_LOW, TEMPO_HIGH, 0.5)
        for phase in np.arange(8) * 7.5 / bpm  # 첫 박 위치: 박을 8로 나눠 옮겨 본다
    ]
    _, bpm, phase = max(cands)
    bpm = round(_fold(float(bpm)), 2)
    return bpm, float(phase) % (60 / bpm)


def estimate_bpm(tr: Transcription) -> float:
    return estimate_tempo(tr)[0]


def align_to_beat(events: list[NoteEvent], bpm: float, phase: float) -> list[NoteEvent]:
    """첫 박 위치(phase초)가 박 위에 오도록 음을 뒤로 민다 (0 ≤ 미는 양 < 한 박). 앞부분은 못갖춘마디가 된다."""
    lead = (-phase) % (60 / bpm)
    if lead < 1e-6:
        return events
    return [NoteEvent(e.start + lead, e.end + lead, e.pitch, e.amplitude) for e in events]


def events_to_project(events: list[NoteEvent], name: str, bpm: float, warnings: list[str]) -> Project:
    """초 단위 음표 → 박 단위 프로젝트. 템포가 틀려도 박 × 60 ÷ 템포 = 원래 초라서 들리는 박자는 그대로다."""
    per_beat = bpm / 60
    notes = [
        Note(
            pitch=e.pitch,
            start=round(e.start * per_beat, 4),
            duration=max(round((e.end - e.start) * per_beat, 4), 0.0001),
            velocity=max(1, min(127, int(round(127 * e.amplitude)))),
        )
        for e in events
    ]
    if not notes:
        raise AudioParseError("음 높이가 있는 소리를 찾지 못했습니다. 드럼만 있는 소리는 바꿀 수 없어요.")
    return Project(
        id=uuid.uuid4().hex[:12],
        name=name,
        tempo=round(bpm, 2),
        tracks=[Track(id="t1", name="오디오에서 바꾼 음", program=0, notes=notes)],
        warnings=warnings,
    )


def audio_bytes_to_project(data: bytes, name: str, bpm: float | None = None) -> Project:
    tr = transcribe_audio(load_audio(data))
    events = tr.events
    warnings = ["오디오에서 음을 읽어 만든 MIDI예요. 여러 악기가 섞인 소리는 음이 빠지거나 더 생길 수 있어요."]
    if bpm is None:
        bpm = bpm_from_name(name)
        if bpm is None:
            bpm, phase = estimate_tempo(tr)
            events = align_to_beat(events, bpm, phase)
            warnings.append(
                f"템포를 소리에서 짐작했어요({bpm:g} BPM). 들리는 박자는 그대로지만, 틀렸다면 마디 줄이 어긋나 보여요. "
                "다시 열 때 템포를 넣거나 파일 이름에 '120 BPM'처럼 넣으면 그 템포로 읽어요."
            )
    events, snapped = quantize(events, bpm)
    if snapped:
        warnings[0] += " 음 시작 · 끝은 16분 격자에 맞췄어요."
    return events_to_project(events, name, bpm, warnings)


BPM_IN_NAME = re.compile(r"(?<!\d)(\d{2,3}(?:\.\d+)?)\s*bpm", re.IGNORECASE)


def bpm_from_name(name: str) -> float | None:
    """'Cymatics - Mango - 98 BPM F Min' → 98. 샘플팩 루프는 대부분 이름에 템포가 있다."""
    m = BPM_IN_NAME.search(name)
    if m and 40 <= float(m.group(1)) <= 300:
        return float(m.group(1))
    return None
