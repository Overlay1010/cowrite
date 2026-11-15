"""멜로디에 코드 붙이기 (app/harmony/harmonize.py)."""

from app.harmony.harmonize import harmonize, is_line, needs_harmony, song_chords
from app.models.project import Note, Project, Track
from app.tools import run_tool


def line(*specs, name="Lead", program=0) -> Track:
    return Track(id=name, name=name, program=program, notes=[Note(pitch=p, start=s, duration=d, velocity=90) for p, s, d in specs])


def song(*tracks: Track) -> Project:
    return Project(id="h", tracks=list(tracks))


def names(p: Project) -> list[str]:
    return [c.name for c in harmonize(p).chords]


def test_simple_melody_gets_primary_chords():
    """C E G | F A C | G B D | C — 마디마다 코드 음을 짚는 멜로디는 I IV V I."""
    tune = line(*[(p, b * 4 + i * 1.0, 1.0) for b, ps in enumerate([(60, 64, 67, 64), (65, 69, 72, 69), (67, 71, 74, 71), (72, 67, 64, 60)])
                  for i, p in enumerate(ps)])
    h = harmonize(song(tune))
    assert h.key.name == "C major" and [c.name for c in h.chords] == ["C", "F", "G", "C"]
    assert [c.roman for c in h.chords] == ["I", "IV", "V", "I"]


def test_minor_riff_stays_in_key():
    """G 단조 멜로디에 조성 밖 코드(E 장조 같은 것)를 붙이지 않는다 — 예전엔 Cymatics Sphinx에 E/G가 붙었다."""
    riff = [(67, 0, 1), (70, 1, 1), (74, 2, 1), (70, 3, 1), (63, 4, 1), (67, 5, 1), (70, 6, 2),
            (72, 8, 1), (75, 9, 1), (79, 10, 1), (75, 11, 1), (74, 12, 1), (67, 13, 1), (70, 14, 2)]
    h = harmonize(song(line(*riff)))
    allowed = {"Gm", "Bb", "Cm", "Dm", "D", "Eb", "F"}
    assert h.key.name == "G minor" and {c.name for c in h.chords} <= allowed
    assert h.chords[0].name == "Gm"


def test_line_detection_ignores_legato_overlap():
    """꾸밈음 · 레가토로 살짝씩 겹치는 멜로디도 한 줄. 같은 순간에 여러 음을 누르는 트랙은 화음."""
    legato = line((67, 0, 0.6), (68, 0.5, 0.3), (67, 0.75, 0.5), (70, 1.0, 1.2), (72, 2.0, 1.0))
    chords = line(*[(p, b * 4.0, 4.0) for b in range(2) for p in (60, 64, 67)], name="Pad", program=89)
    assert is_line(legato) and not is_line(chords)
    assert needs_harmony(song(legato)) and not needs_harmony(song(legato, chords))


def test_same_chords_after_our_bass_is_added():
    """편곡은 베이스를 먼저 깔고 화음을 깐다 — 베이스가 생긴 곡에 다시 붙여도 같은 코드여야 파트끼리 어긋나지 않는다
    (예전엔 베이스가 생긴 뒤 다시 읽어 G 단조 곡 2마디에 G 장조가 깔렸다)."""
    riff = [(67, 0, 1), (70, 1, 1), (74, 2, 1), (70, 3, 1), (63, 4, 1), (67, 5, 1), (70, 6, 2),
            (72, 8, 1), (75, 9, 1), (79, 10, 1), (75, 11, 1), (74, 12, 1), (67, 13, 1), (70, 14, 2)]
    p = song(line(*riff))
    with_bass = run_tool(p, "generate_bassline", {}).project
    assert [(c.start, c.name) for c in song_chords(with_bass).chords] == [(c.start, c.name) for c in harmonize(p).chords]


def test_short_rest_keeps_chord_long_rest_is_no_chord():
    """한 마디보다 짧게 쉬면 앞 코드를 이어 끌고(반주가 반 마디마다 끊기지 않게), 그보다 길면 N.C. (반주 쪽이 앞뒤 진행으로 채움)."""
    tune = line((60, 0, 2), (64, 4, 2), (67, 6, 2), (60, 20, 4))  # 2~4박 쉼(반 마디) · 8~20박 쉼(3마디)
    chords = harmonize(song(tune)).chords
    assert chords[0].start == 0 and chords[0].end >= 4
    assert any(c.root is None and c.start <= 8 and c.end >= 20 for c in chords)


def test_songs_with_harmony_are_read_not_harmonized():
    pad = line(*[(p, b * 4.0, 4.0) for b, ch in enumerate([(60, 64, 67), (57, 60, 64)]) for p in ch], name="Pad", program=89)
    assert [c.name for c in song_chords(song(pad)).chords] == ["C", "Am"]


# 4마디 G 단조 리프를 5~8마디에서 옥타브만 내려 되풀이하는 곡 (Cymatics Sphinx와 같은 꼴, 음은 직접 만든 것)
G, Ab, Bb, D, Eb, F = 67, 68, 70, 74, 75, 65
RIFF = [[G, G, D, Eb, Ab, Bb, G, Ab], [G, F, G, Ab, G, Bb, D, G], [G, Ab, G, G, Bb, Ab, G, G], [G, F, G, Ab, G, Ab, G, G]]


def riff_song(legato: bool = False) -> Project:
    """8분음표 리프. legato면 음마다 다음 음과 살짝 겹치게."""
    notes = [Note(pitch=p - 12 * half, start=(half * 4 + b) * 4 + i * 0.5, duration=0.6 if legato else 0.5, velocity=90)
             for half in (0, 1) for b, bar in enumerate(RIFF) for i, p in enumerate(bar)]
    return song(Track(id="t1", name="Lead", notes=notes))


def test_repeated_phrase_gets_same_chords():
    """옥타브만 바꿔 되풀이하는 4마디에는 같은 코드. 예전엔 이어짐 점수 때문에 Gm Gm Gm Eb / Cm Gm Eb Gm처럼 다르게 붙었다."""
    from app.harmony.harmonize import repeats

    p = riff_song()
    assert repeats(p.tracks[0].notes, 4, 8) == [(4, [0, 4])]
    chords = harmonize(p).chords
    per_bar = [next(c.name for c in chords if c.start <= b * 4 < c.end) for b in range(8)]
    assert per_bar[:4] == per_bar[4:]


def test_legato_melody_counts_as_melody():
    """살짝씩 겹치는 레가토 멜로디도 멜로디 — 예전엔 겹침 비율로 화음 트랙처럼 봐서 반주가 멜로디를 피하지 못했다."""
    from app.harmony.analyze import melody_ids

    p = riff_song(legato=True)
    bass = Track(id="t2", name="Bass", program=33, notes=[Note(pitch=43, start=b * 4.0, duration=4.0, velocity=90) for b in range(8)])
    assert melody_ids(p.model_copy(update={"tracks": [*p.tracks, bass]})) == {"t1"}
