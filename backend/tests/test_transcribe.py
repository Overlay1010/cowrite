"""오디오 → 음표 (app/midi_io/transcribe.py). 소리는 코드로 합성한다 (배음이 있는 음을 정해진 박에)."""

from __future__ import annotations

import io

import numpy as np
import pytest
import soundfile as sf

from app.midi_io import transcribe as T

SR = 44100


def tone(pitch: int, seconds: float, sr: int = SR) -> np.ndarray:
    """배음 다섯 개 · 빠른 어택 · 서서히 줄어드는 음 (건반 비슷하게)."""
    t = np.arange(int(seconds * sr)) / sr
    f = 440 * 2 ** ((pitch - 69) / 12)
    wave = sum(np.sin(2 * np.pi * f * k * t) / k for k in range(1, 6))
    env = np.minimum(1, t / 0.005) * np.exp(-t * 2.5) * np.minimum(1, (seconds - t) / 0.02)
    return (wave * env).astype(np.float32)


def render(notes: list[tuple[int, float, float]], bpm: float, beats: float, channels: int = 1) -> bytes:
    """(음, 시작 박, 길이 박) → WAV 바이트."""
    spb = 60 / bpm
    out = np.zeros(int(beats * spb * SR), dtype=np.float32)
    for pitch, start, length in notes:
        s = int(start * spb * SR)
        x = tone(pitch, length * spb)[: len(out) - s]
        out[s : s + len(x)] += x
    out = 0.3 * out / np.abs(out).max()
    if channels == 2:
        out = np.stack([out, out], axis=1)
    buf = io.BytesIO()
    sf.write(buf, out, SR, format="WAV", subtype="PCM_16")
    return buf.getvalue()


# C – Am – F – G를 한 마디씩, 4분음표로 펼쳐서 (120 BPM, 4마디)
ARP = [(p, bar * 4 + i, 1.0) for bar, ch in enumerate([(60, 64, 67, 72), (57, 60, 64, 69), (53, 57, 60, 65), (55, 59, 62, 67)]) for i, p in enumerate(ch)]


def test_transcribes_pitches_on_their_beats():
    events = T.transcribe(T.load_audio(render(ARP, 120, 16)))
    project = T.events_to_project(events, "x", 120, [])
    got = {(n.pitch, round(n.start * 4) / 4) for n in project.tracks[0].notes}
    want = {(p, float(b)) for p, b, _ in ARP}
    assert len(want & got) >= len(want) - 1  # 16음 중 15음 이상을 같은 음 · 같은 박에
    near = [n for n in project.tracks[0].notes if (n.pitch, round(n.start)) in {(p, b) for p, b, _ in ARP}]
    assert max(abs(n.start - round(n.start)) for n in near) < 0.05  # 박에서 0.05박(25ms) 안


def test_stereo_and_other_rates_are_resampled():
    buf = io.BytesIO()
    x = np.stack([tone(69, 1.0, 48000)] * 2, axis=1)
    sf.write(buf, x, 48000, format="FLAC")
    audio = T.load_audio(buf.getvalue())
    assert audio.ndim == 1 and abs(len(audio) - T.SAMPLE_RATE) < 2  # 1초
    assert 69 in {e.pitch for e in T.transcribe(audio)}


def test_bpm_from_name():
    assert T.bpm_from_name("Cymatics - Mango - 98 BPM F Min") == 98
    assert T.bpm_from_name("loop_140bpm") == 140
    assert T.bpm_from_name("Piano Take 3") is None
    assert T.bpm_from_name("1000 BPM") is None


# 100 BPM, 마디마다 4분 · 8분 · 16분이 섞인 리듬 (2마디 반복)
RHYTHM = [0, 1, 1.5, 2, 2.75, 3, 3.5, 4, 5, 5.5, 6, 6.25, 6.5, 7]
MELODY = [(60 + (i * 5) % 12, b + rep * 8, 0.25) for rep in range(2) for i, b in enumerate(RHYTHM)]


def test_estimate_bpm_from_loop_length():
    # 4마디 = 9.6초 루프: 길이가 N마디인 템포 중 박 · 8분 · 16분 자리에 시작점이 몰리는 것
    tr = T.transcribe_audio(T.load_audio(render(MELODY, 100, 16)))
    assert abs(tr.seconds - 9.6) < 0.01
    assert T.estimate_bpm(tr) == 100


def test_estimate_bpm_for_recording_not_cut_to_bars():
    # 앞에 0.37초 쉬고 시작해 4마디 뒤 조금 남는 녹음 → 길이가 마디로 떨어지지 않으니 자유 템포 + 첫 박 위치
    wav = render([(p, b + 0.37 * 100 / 60, d) for p, b, d in MELODY], 100, 17.3)
    tr = T.transcribe_audio(T.load_audio(wav))
    bpm, phase = T.estimate_tempo(tr)
    assert abs(bpm - 100) <= 1
    # 첫 박 위치만큼 밀면 음이 16분 격자 위에 온다
    project = T.events_to_project(T.align_to_beat(tr.events, bpm, phase), "x", bpm, [])
    off = [abs(n.start * 4 - round(n.start * 4)) for n in project.tracks[0].notes]
    assert np.median(off) < 0.15


def N(start: float, end: float, pitch: int, amp: float) -> T.NoteEvent:
    return T.NoteEvent(start=start, end=end, pitch=pitch, amplitude=amp)


def test_quantize_only_when_mostly_on_the_grid():
    step = 15 / 120  # 120 BPM 16분
    loose = [N(i * step + d, i * step + step * 0.9, 60, 0.5) for i, d in enumerate([0.01, -0.02, 0.015, 0, -0.01, 0.02, 0.03, -0.012])]
    got, snapped = T.quantize(loose, 120)
    assert snapped and all(abs(e.start / step - round(e.start / step)) < 1e-9 for e in got)
    triplets = [N(i * 0.5 / 3, i * 0.5 / 3 + 0.1, 60, 0.5) for i in range(12)]  # 8분 셋잇단: 16분 격자에 안 붙음
    got, snapped = T.quantize(triplets, 120)
    assert not snapped and got == triplets


def test_upload_wav_opens_as_project(client):
    data = render(ARP, 120, 16, channels=2)
    r = client.post("/api/v1/projects", files={"file": ("Arp - 120 BPM.wav", data, "audio/wav")})
    assert r.status_code == 200, r.text
    p = r.json()
    assert p["tempo"] == 120 and p["name"] == "Arp - 120 BPM"
    assert len(p["tracks"]) == 1 and len(p["tracks"][0]["notes"]) >= 15
    assert "오디오에서" in p["warnings"][0] and len(p["warnings"]) == 1
    assert "격자" in p["warnings"][0]  # 4분음표 펼침화음이라 16분 격자에 맞춤
    played = {(pitch, float(b)) for pitch, b, _ in ARP}
    hits = [n for n in p["tracks"][0]["notes"] if (n["pitch"], round(n["start"])) in played]
    assert len(hits) >= 15 and all(n["start"] == round(n["start"]) for n in hits)  # 친 음은 박 위에 정확히


def test_upload_wav_without_bpm_guesses_and_says_so(client):
    r = client.post("/api/v1/projects", files={"file": ("arp.wav", render(ARP, 120, 16), "audio/wav")})
    p = r.json()
    assert p["tempo"] == 120  # 4마디 = 8초 → 120 BPM
    assert any("짐작" in w for w in p["warnings"])


def test_upload_given_bpm_wins(client):
    r = client.post("/api/v1/projects", files={"file": ("Arp - 90 BPM.wav", render(ARP, 120, 16), "audio/wav")}, data={"bpm": "120"})
    assert r.json()["tempo"] == 120


def test_silence_has_no_notes(client):
    buf = io.BytesIO()
    sf.write(buf, np.zeros(SR, dtype=np.float32), SR, format="WAV")
    r = client.post("/api/v1/projects", files={"file": ("quiet.wav", buf.getvalue(), "audio/wav")})
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "audio_parse_failed"


@pytest.mark.parametrize("name", ["._loop.wav", "broken.mp3"])
def test_broken_audio(client, name):
    r = client.post("/api/v1/projects", files={"file": (name, b"\x00\x05\x16\x07 not audio", "audio/wav")})
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "audio_parse_failed"


def test_monophonic_keeps_the_louder_of_overlapping_notes():
    got = T.monophonic([N(0.0, 1.0, 60, 0.8), N(0.5, 0.7, 61, 0.3), N(1.0, 2.0, 62, 0.5), N(1.5, 2.5, 74, 0.9)])
    assert [(e.start, e.pitch) for e in got] == [(0.0, 60), (1.5, 74)]  # 겹치는 약한 음(61 · 62)은 빠진다


def test_upload_humming_reads_one_note_at_a_time(client):
    wav = render([(60, 0, 4), (64, 0, 4), (67, 0, 4)], 120, 4)  # 화음 — 보통은 세 음, 허밍으로 읽으면 겹치지 않게
    chord = client.post("/api/v1/projects", files={"file": ("c - 120 BPM.wav", wav, "audio/wav")}).json()
    hum = client.post("/api/v1/projects", files={"file": ("c - 120 BPM.wav", wav, "audio/wav")}, data={"mono": "true"}).json()
    assert len(chord["tracks"][0]["notes"]) >= 3 and "허밍" not in chord["warnings"][0]
    notes = sorted(hum["tracks"][0]["notes"], key=lambda n: n["start"])
    assert notes and all(a["start"] + a["duration"] <= b["start"] + 1e-6 for a, b in zip(notes, notes[1:]))
    assert "허밍" in hum["warnings"][0]
