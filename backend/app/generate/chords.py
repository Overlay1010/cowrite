"""코드 반주 만들기 (generate_chords). 무늬 셋: 패드(길게 누르기) · 끊어 치기(리듬) · 아르페지오(한 음씩 펼치기).

코드는 v0.2 분석 엔진이 읽은 곡의 코드, 또는 사용자가 말한 진행("C – Am – F – G")을 따라간다.
음 고르기는 무늬와 상관없이 같다 (아래). 무늬는 고른 4음을 어떤 리듬으로 치느냐만 다르다.
- 음역: 베이스 위쪽과 멜로디 아래쪽 사이 빈 곳 — 다른 악기와 같은 높이에서 겹치지 않게. 4마디 묶음마다 따로 정하고,
  멜로디가 낮아 아래가 탁한 높이뿐이면 멜로디 위로
- 보이싱: 4음. 첫 코드는 음역 가운데, 다음 코드부터는 앞 코드에서 가장 덜 움직이게 (패드가 코드마다 뛰지 않게).
  그동안 같이 울리는 멜로디 · 다른 트랙 음과 반음(단2 · 단9)으로 부딪히는 음은 피하고, 3음이라도 많이 부딪히면 뺀다
- 소리: 곡 편성에 맞춰 — 스트링 · 관악기가 있으면 스트링 앙상블, 신스 리드 · 신스 베이스 · 일렉트로닉 킷이 있으면
  폴리신스 패드, 그 밖(건반 · 로파이 · 피아노)은 따뜻한 패드. 이미 같은 소리가 있으면 다른 패드.
  템포로는 고르지 않는다 (150 BPM 피아노 왈츠에 신스 패드가 깔렸다)
- 장르(style)를 말하면: 끊어 치기는 그 장르의 두 마디 리듬(소울 당겨 치기 · 펑크 16분 …), 소울 · 재즈 · 로파이 · 붐뱁 · 펑크는
  7 · 9음을 더한다 (멜로디와 반음으로 부딪히면 덜어 냄). 사용자가 말한 진행은 말한 그대로
- 세기: 원래 곡보다 조금 작게 (뒤에 깔리게). 끊어 치기는 자리마다 조금씩 흔든다
- 다른 버전: version마다 첫 코드를 다른 모양(촘촘하게 · 넓게 · 높게 · 4도로 벌려)으로 짜고, 거기서부터 이어 간다
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field, replace
from itertools import product
from random import Random
from statistics import mean, median

from app.generate.check import playable
from app.harmony import ChordSpan
from app.harmony.harmonize import song_chords
from app.harmony.analyze import melody_ids
from app.harmony.reharm import INTERVALS, TENSION
from app.harmony.voicing import revoice
from app.models.project import Note, Project, Track
from app.tools.instrument import heard_program

VOICES = 4
LOW, HIGH = 45, 79  # 패드가 머무를 바깥 한계 (A2 ~ G5)
GAP = 2  # 반음. 베이스 · 멜로디와 이만큼은 떨어진다
NARROW = 10  # 반음. 빈 곳이 이보다 좁으면 멜로디 아래쪽에 붙여 놓는다
BASS_BELOW = 48  # 음 높이 가운데값이 이보다 낮은 화음 트랙은 베이스로 본다 (C3)
CLEAR = 50  # D3. 화음을 이보다 낮게 촘촘히 쌓으면 탁하다 — 멜로디 아래 빈 곳이 여기서부터 NARROW보다 좁으면 멜로디 위로
UP = 12  # 반음. 멜로디 위로 올릴 때 음역 폭
DRIFT = 5  # 반음. 보이싱 가운데가 음역 가운데에서 이만큼 넘게 끌려가면 다시 가운데에서 짠다
SOFTER = 0.75  # 원래 곡 세기에 곱한다
CLUSTER = 6  # 반음 붙은 두 음의 벌점 (움직임 반음 수로)
HEIGHT_SHIFT = {"low": -7, "mid": 0, "high": 7}
# 다른 트랙과 반음(단2 · 단9)으로 부딪히는 음 피하기 — 편곡 점검(app/arrange/review.py)과 같은 잣대.
# 점검에 가장 많이 걸린 게 멜로디가 C를 끄는 동안 패드가 B를 누르는 것이었다 (Am-F-C-G 진행 팝 반주 12.5박)
OVERLAP = 0.5  # 박. 이보다 짧게 겹치면 지나가는 음으로 본다 (점검과 같은 값)
RUB = 16  # 반음 움직임 몇 개만큼의 벌점: 다른 트랙 음과 반음으로 1박 같이 울릴 때
HEAVY = 1.0  # 박. 꼭 넣을 음(3음 · 7음)이라도 어느 높이에 두든 이만큼 부딪히면 뺀다 (멜로디가 4음을 끌면 3음 대신 4음 = sus4)
SUS = {"": "sus4", "7": "7sus4"}  # 장3음 대신 4음을 넣으면 바뀌는 코드 (검사 · 결과에 바뀐 이름으로)
SHAPES = [(-7, -3, 2, 6), (-10, -3, 4, 9), (-4, 0, 5, 9), (-12, -5, 0, 7)]  # 버전마다 첫 코드를 짤 모양 (음역 가운데에서 반음)

WARM_PAD, POLYSYNTH_PAD, STRINGS = 89, 90, 48
OTHER_PADS = [91, 88, 95]  # 콰이어 · 뉴에이지 · 스윕 패드 (이미 같은 소리가 있을 때)
ELECTRONIC_KITS = {24, 25, 26}  # 일렉트로닉 · TR-808 · 하우스


@dataclass
class PadPlan:
    program: int
    low: int  # 실제로 쓴 가장 낮은 · 높은 음
    high: int
    chords: list[ChordSpan] = field(default_factory=list)  # 패드를 놓은 코드 (검사에도 쓴다)
    feel: str = ""  # 쓴 리듬 (끊어 치기 · 아르페지오)


def pad_program(project: Project) -> int:
    heard = [heard_program(t) for t in project.tracks if not t.is_drum]
    kits = {t.program for t in project.tracks if t.is_drum}
    if any(40 <= p <= 79 for p in heard):  # 스트링 · 앙상블 · 금관 · 목관 — 어쿠스틱 편성에 신스 패드는 튄다
        want = STRINGS
    elif any(80 <= p <= 87 or p in (38, 39) for p in heard) or kits & ELECTRONIC_KITS:
        want = POLYSYNTH_PAD
    else:
        want = WARM_PAD
    from app.arrange.suggest import choose_program  # 곡 읽기 (늦게 — 도구 목록과 돌고 도는 import)

    return choose_program(project, "pad", [want, WARM_PAD, POLYSYNTH_PAD, *OTHER_PADS])


def _percentile(values: list[int], q: float) -> int:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def pad_window(project: Project, start: float, end: float) -> tuple[int, int]:
    """패드가 머무를 [가장 낮은, 가장 높은] 음. 베이스(따로 있을 때) 위쪽 · 멜로디 아래쪽 사이 빈 곳."""
    pitched = [t for t in project.tracks if not t.is_drum and t.notes]
    melody = melody_ids(project)
    harmonic = [t for t in pitched if t.id not in melody]

    def heard(t: Track) -> list[int]:
        return [n.pitch for n in t.notes if n.start < end and n.start + n.duration > start]

    low, high = LOW, HIGH
    # 베이스 트랙: 베이스 악기이거나 실제로 낮게 치는 화음 트랙. 그냥 가장 낮은 트랙으로 고르면 이미 깐 패드를 베이스로 보고
    # 그 위에 깔아서 "낮은 패드"가 원래 패드보다 높게 나왔다 (Alarma: 패드 G3~G4 위에 C4~G5)
    basses = [t for t in harmonic if 32 <= heard_program(t) <= 39 or median(n.pitch for n in t.notes) < BASS_BELOW]
    # 베이스가 여럿이면(원래 베이스 + 새로 만든 베이스) 모두의 위로 — 가장 낮은 것 하나만 보면 원래 베이스 높이에 깔렸다 (lofi 52 → 43)
    if len(harmonic) > 1:
        for bass in basses:
            if pitches := heard(bass):
                low = max(low, _percentile(pitches, 0.9) + GAP)
    tune = [p for t in pitched if t.id in melody for p in heard(t)]
    if tune:
        below = _percentile(tune, 0.1) - GAP
        above = max(tune) + GAP  # 위로 올릴 땐 가장 높은 음 위로 (짧게 찍는 꼭대기 음 D4 · Eb4와도 겹쳤다)
        # 멜로디가 낮아 그 아래로는 탁한 높이밖에 안 남으면 멜로디 위로 (Sphinx 후반 4마디: 멜로디 F3~Eb4 아래에 깔 자리가 없어
        # 반주가 멜로디와 같은 G3~G4에 깔렸다)
        if below - max(low, CLEAR) < NARROW and above + NARROW <= HIGH:
            return above, min(HIGH, above + UP)
        high = min(high, below)
        if basses and high - CLEAR >= NARROW:  # 베이스가 낮은 곳을 받치면 화음은 탁하지 않은 높이부터
            low = max(low, CLEAR)
    if high - low < NARROW:  # 좁으면 멜로디 바로 아래 한 옥타브 (멜로디를 가리지 않게, 베이스와는 조금 겹쳐도)
        high = max(LOW + 12, min(HIGH, high))
        low = high - 12
    return low, high


def _must(chord: ChordSpan) -> set[int]:
    """꼭 넣을 음: 코드 색을 정하는 3음 · 7음(6음) · 9음 · 변한 5음. 7음이 없는 3화음이면 세 음 모두."""
    ivs = INTERVALS.get(chord.quality, (0, 4, 7))
    color = {i for i in ivs if i in (2, 3, 4, 6, 8, 9, 10, 11)}  # 9음(2)도 — Cm9라고 해 놓고 9음 없이 Eb를 겹쳐 쌓았다
    picked = color if any(i in (9, 10, 11) for i in ivs) else set(ivs)
    return {(chord.root + i) % 12 for i in picked}


def heard_with(project: Project, chords: list[ChordSpan], start: float, end: float) -> list[dict[int, float]]:
    """코드마다 그동안 같이 울리는 다른 트랙(드럼 말고) 음: {음 높이: 겹친 박}. 같은 높이를 다 합쳐 OVERLAP보다 짧으면 뺀다
    (지나가는 음). 음 하나씩 따지면 0.1~0.25박짜리 Ab을 마디마다 여러 번 찍는 멜로디(Sphinx)와 G가 계속 부딪혀도 몰랐다."""
    notes = sorted((n for t in project.tracks if not t.is_drum for n in t.notes), key=lambda n: n.start)
    out = []
    for c in chords:
        s, e = max(c.start, start), min(c.end, end)
        heard: dict[int, float] = {}
        for n in notes:
            if n.start >= e:
                break
            ov = min(e, n.start + n.duration) - max(s, n.start)
            if ov > 0:
                heard[n.pitch] = heard.get(n.pitch, 0.0) + ov
        out.append({m: beats for m, beats in heard.items() if beats >= OVERLAP})
    return out


def rub(pitch: int, heard: dict[int, float]) -> float:
    """pitch가 heard의 음들과 반음(단2 · 단9 — 위 음이 아래 음보다 반음 + 옥타브들)으로 같이 울리는 박."""
    return sum(beats for m, beats in heard.items() if abs(m - pitch) % 12 == 1)


def lead(prev: list[int], chord: ChordSpan, low: int, high: int, heard: dict[int, float] | None = None) -> list[int]:
    """prev(4음)에서 가장 덜 움직여 chord로 가는 4음. 목소리마다 코드 음 중 가까운 후보를 두고, 겹치지 않고 음역 안이며
    꼭 넣을 음을 다 넣는 조합 중 움직임 합이 가장 작은 것 (후보가 적어 다 따져 본다).
    v0.2 보이싱 규칙(3음부터 겹치기)을 그대로 쓰면 Am에서 C를 겹치느라 한 음이 음역 밖으로 뛰고, 안으로 들이면 같은 음이 둘이 됐다.
    heard(같이 울리는 다른 트랙 음)가 있으면 그 음들과 반음으로 부딪히는 음에 벌점을 주고, 어느 높이에 둬도 많이 부딪히는
    꼭 넣을 음은 뺀다 (멜로디가 C를 끄는 G 코드면 B를 빼 G · D만)."""
    ivs = INTERVALS.get(chord.quality, (0, 4, 7))
    pcs = {(chord.root + i) % 12 for i in ivs}
    must = _must(chord)
    heard = heard or {}
    if heard:  # 근음 · 5음은 베이스가 받치니 조금만 부딪혀도 빼도 된다 (B 코드 위 멜로디 C에 패드 B), 3음 · 7음은 많이 부딪힐 때만
        color = {(chord.root + i) % 12 for i in ivs if i not in (0, 7)}
        dropped = {pc for pc in must
                   if min((rub(q, heard) for q in range(low - GAP, high + GAP + 1) if q % 12 == pc), default=0.0)
                   >= (HEAVY if pc in color else OVERLAP)}
        must -= dropped
        # 장3음을 뺐으면 그 위에서 울리던 4음(멜로디)을 넣을 수 있게 — G 코드 위 멜로디 C면 B 대신 C로 Gsus4.
        # 단3음은 반음 위가 장3음이라 넣으면 장조가 돼서 빼기만 한다
        fourth = (chord.root + 5) % 12
        if chord.quality in SUS and (chord.root + 4) % 12 in dropped and any(m % 12 == fourth for m in heard):
            pcs.add(fourth)
    options = []
    for p in prev:
        near = {p - (p - pc) % 12 for pc in pcs} | {p - (p - pc) % 12 + 12 for pc in pcs}
        options.append(sorted((q for q in near if low - GAP <= q <= high + GAP), key=lambda q: abs(q - p))[:4])
    rubs = {q: rub(q, heard) for opts in options for q in opts} if heard else {}
    best = None
    for combo in product(*options):
        if len(set(combo)) < len(combo) or not must <= {q % 12 for q in combo}:
            continue
        cost = sum(abs(a - b) for a, b in zip(combo, prev)) + RUB * sum(rubs.get(q, 0.0) for q in combo)
        voiced = sorted(combo)  # 이웃한 두 음이 반음이면 탁하다 (Dm9의 E F를 붙여 쌓던 것) — 반음 6개 움직임만큼 피한다
        cost += CLUSTER * sum(1 for x, y in zip(voiced, voiced[1:]) if y - x == 1)
        if best is None or cost < best[0]:
            best = (cost, combo)
    return sorted(best[1]) if best else sorted(revoice(prev, chord))


def shape(version: int) -> list[int]:
    """version번째 첫 코드 모양. 모양을 다 쓰면 조금씩 흔들어 새 모양을 만든다."""
    base = SHAPES[(version - 1) % len(SHAPES)]
    if version <= len(SHAPES):
        return list(base)
    rng = Random(version)
    return [x + rng.choice((-2, -1, 1, 2)) for x in base]


def voicings(chords: list[ChordSpan], windows: list[tuple[int, int]], version: int = 1,
             heard: list[dict[int, float]] | None = None) -> list[list[int]]:
    """코드마다 4음. 앞 코드에서 가장 덜 움직이게 잇고, 가운데에서 너무 멀어지면 다시 가운데에서. 코드마다 음역 windows 안
    (음역이 바뀌면 — 멜로디가 옥타브를 바꾼 묶음 — 새 음역 가운데에서 다시 짠다).
    heard(코드마다 같이 울리는 다른 트랙 음, heard_with)가 있으면 그 음들과 반음으로 부딪히지 않게 고른다."""
    out: list[list[int]] = []
    prev: list[int] = []
    for i, (c, (low, high)) in enumerate(zip(chords, windows)):
        center = (low + high) // 2
        seed = [center + x for x in shape(version)]
        h = heard[i] if heard else None
        v = lead(prev if prev and windows[i - 1] == (low, high) else seed, c, low, high, h)
        if abs(mean(v) - center) > DRIFT:
            v = lead(seed, c, low, high, h)
        out.append(v)
        prev = v
    return out


# 끊어 치기 리듬: 한 마디(4박) 안의 (시작 박, 길이). 버전마다 다음 리듬. 3/4처럼 짧은 마디는 마디 안에 드는 것만
STAB_RHYTHMS = [
    ((0, 0.5), (1, 0.5), (2, 0.5), (3, 0.5)),  # 박마다
    ((0.5, 0.5), (1.5, 0.5), (2.5, 0.5), (3.5, 0.5)),  # 뒷박 (하우스 · 레게)
    ((0, 1), (1.5, 0.5), (2.5, 1)),  # 당겨서
    ((0, 0.75), (1.5, 0.75), (3, 0.5)),  # 3-3-2
]
ARP_STYLES = [("up", 0.5), ("updown", 0.5), ("up", 0.25), ("updown", 0.25)]  # 아르페지오: (순서, 한 음 길이 박)
# 결과 요약에 적을 리듬 (LLM이 짐작해 말하지 않게)
STAB_NAMES = ["박마다", "뒷박", "당겨서", "3-3-2"]
ARP_NAMES = ["8분음표 올라가기", "8분음표 오르내리기", "16분음표 올라가기", "16분음표 오르내리기"]
# 장르별 반주 리듬: 두 마디(4/4 여덟 박) 안의 (시작 박, 길이). 장르를 말하면 위 리듬 대신 (버전마다 다음 것).
# "소울 느낌으로 피아노 반주"에 박마다 8분음표로 뚝뚝 끊어 쳐서 "절대 소울 느낌이 아닌데"를 들었다 (2026-11-15).
# 다음 코드 바로 앞(반 박 안)에서 시작해 코드가 바뀌는 자리를 넘어 끄는 음은 다음 코드를 미리 친다 (당겨 치기)
GENRE_COMP: dict[str, list[tuple[tuple[float, float], ...]]] = {
    "soul": [((0, 1.5), (1.5, 0.5), (2.5, 1.0), (4, 1.0), (5.5, 1.0), (7.5, 1.5)),
             ((0, 0.75), (0.75, 0.75), (2, 0.5), (2.5, 1.5), (4, 1.5), (5.5, 0.5), (6.5, 1.0), (7.5, 1.0))],
    "jazz": [((0, 1.0), (1.5, 0.5), (4.5, 0.5), (6, 1.0), (7.5, 1.0))],  # 찰스턴 · 당겨 치기
    "funk": [((0, 0.25), (0.75, 0.25), (1.5, 0.5), (2.5, 0.25), (3.25, 0.25), (3.75, 0.25),
              (4, 0.25), (4.75, 0.25), (5.5, 0.25), (6.5, 0.5), (7.25, 0.25))],  # 16분음표로 짧게
    "lofi": [((0, 2.5), (2.5, 1.5), (4, 3.5), (7.5, 1.0))],  # 길게 늘어지게
    "boombap": [((0, 1.5), (1.5, 2.0), (4, 2.5), (6.5, 1.5))],
    "pop": [((0, 1.5), (1.5, 1.0), (2.5, 1.5), (4, 1.5), (5.5, 1.0), (6.5, 1.5))],
    "rock": [tuple((i * 0.5, 0.5) for i in range(16))],  # 8분음표로 고르게
    "house": [tuple((i + 0.5, 0.5) for i in range(8))],  # 뒷박
    "trap": [((0, 2.0), (2.5, 1.5), (4, 4.0))],
    "latin": [((0, 0.5), (1.5, 0.5), (3, 0.5), (5, 0.5), (6, 0.5), (7.5, 1.0))],  # 클라베 자리
}
GENRE_COMP_NAMES = {
    "soul": "소울 당겨 치기", "jazz": "재즈 컴핑(찰스턴)", "funk": "펑크 16분 끊어 치기", "lofi": "로파이 길게 늘어지게",
    "boombap": "붐뱁 길게", "pop": "팝 3-3-2", "rock": "록 8분음표", "house": "하우스 뒷박", "trap": "트랩 길게", "latin": "라틴 클라베",
}
PUSH = 0.5  # 박. 코드가 바뀌기 이만큼 안쪽에서 시작해 넘어가는 음 = 다음 코드를 당겨 친 것
# 7 · 9음을 넣는 장르. 음계 화음마다 넣을 텐션(reharm.TENSION)부터, 멜로디와 반음으로 부딪히면 한 단계씩 덜어 낸다
COLORFUL = {"soul", "jazz", "lofi", "boombap", "funk"}
LIGHTER = {"maj9": ["maj9", "maj7", "add9", "6"], "m9": ["m9", "m7", "madd9"], "9": ["9", "7"], "7": ["7"], "m7": ["m7"],
           "m7b5": ["m7b5"]}
HUMAN = 4  # 끊어 치기 세기를 치는 자리마다 ±이만큼 흔든다 (기계처럼 같은 세기로 들리지 않게)
PUSH_ACCENT = 4  # 당겨 친 음은 조금 세게
ACCENT = 8  # 끊어 치기에서 마디 첫 박을 이만큼 세게
STAB_SOFTER = 0.85  # 끊어 치기는 패드보다 덜 줄인다 (짧게 치는 소리라 묻힌다)

PIANO, E_PIANO, VIBES, HARP, SAW_LEAD = 0, 4, 11, 46, 81


def comp_program(project: Project, pattern: str) -> int:
    """무늬별 소리. 패드는 pad_program. 끊어 치기 · 아르페지오는 신스 리드 · 일렉트로닉 킷이 있으면 폴리신스 · 톱니파 리드,
    스트링 · 관악기 편성이면 피아노 · 하프, 그 밖(건반 · 로파이)은 전자 피아노 · 비브라폰. 이미 같은 소리가 있으면 다음 후보.
    신스 베이스만으로는 신스 편성으로 보지 않는다 (로파이 곡 아르페지오가 톱니파 리드로 나왔다)."""
    if pattern == "pad":
        return pad_program(project)
    heard = [heard_program(t) for t in project.tracks if not t.is_drum]
    kits = {t.program for t in project.tracks if t.is_drum}
    electronic = any(80 <= p <= 87 for p in heard) or bool(kits & ELECTRONIC_KITS)
    acoustic = any(40 <= p <= 79 for p in heard)
    if pattern == "stab":
        want = [POLYSYNTH_PAD, E_PIANO] if electronic else [PIANO, E_PIANO] if acoustic else [E_PIANO, PIANO]
    else:
        want = [SAW_LEAD, E_PIANO] if electronic else [HARP, PIANO] if acoustic else [E_PIANO, VIBES, HARP]
    from app.arrange.suggest import choose_program

    # 이미 쓴 소리 · 같은 높이의 같은 계열(로즈 옆 피아노 반주)은 피해 같은 역할의 후보 묶음에서 다음 것 (app/arrange/suggest.py)
    flavor = "electronic" if electronic else "acoustic" if acoustic else "neutral"
    return choose_program(project, "keys" if pattern == "stab" else "pluck", [*want, PIANO, VIBES], flavor)


# ---------- 직접 말한 진행 ----------

_CHORD = re.compile(r"^([A-G])([#b]?)(.*?)(?:/([A-G])([#b]?))?$")
QUALITY_ALIASES = {
    "maj": "", "M": "", "min": "m", "mi": "m", "-": "m", "M7": "maj7", "Maj7": "maj7", "ma7": "maj7", "Δ": "maj7", "Δ7": "maj7",
    "-7": "m7", "min7": "m7", "mi7": "m7", "ø": "m7b5", "ø7": "m7b5", "m7-5": "m7b5", "°": "dim", "o": "dim", "°7": "dim7",
    "o7": "dim7", "+": "aug", "sus": "sus4", "add2": "add9", "M9": "maj9", "Maj9": "maj9", "69": "6/9", "dom7": "7",
}
_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
_ACC = {"#": 1, "b": -1}


def parse_chord(text: str) -> tuple[int, str, int | None]:
    """"Am7" · "F#m7b5" · "C/E" · "Bbmaj7" → (근음, 종류, 베이스 또는 None). 모르는 이름이면 ValueError."""
    m = _CHORD.match(text.strip().replace("(", "").replace(")", "").replace("♯", "#").replace("♭", "b"))
    quality = QUALITY_ALIASES.get(m[3], m[3]) if m else None
    if not m or quality not in INTERVALS:
        raise ValueError(text)
    root = (_PC[m[1]] + _ACC.get(m[2], 0)) % 12
    bass = (_PC[m[4]] + _ACC.get(m[5], 0)) % 12 if m[4] else None
    return root, quality, bass


def place(progression: list[str], start: float, end: float, beats: float) -> list[ChordSpan]:
    """진행을 [start, end) 박에 한 코드 beats박씩 되풀이해 놓는다 (끝에서 자른다)."""
    parsed = [parse_chord(c) for c in progression]
    out = []
    t, i = start, 0
    while t < end - 1e-6:
        root, quality, bass = parsed[i % len(parsed)]
        name = progression[i % len(progression)].strip()
        out.append(ChordSpan(t, min(t + beats, end), root, root if bass is None else bass, name, quality, "", "", 1.0))
        t, i = t + beats, i + 1
    return out


FILL_MAX = 8  # 마디. 곡 중간에 이보다 길게 아무 음도 없는 곳은 일부러 비운 구간(브레이크)일 수 있어 채우지 않는다
PHRASE, BLOCK = 4, 8  # 마디. 루프 음악은 4마디 진행을 8마디 묶음 안에서 되풀이하는 일이 많다


def fill_rests(chords: list[ChordSpan], bpb: float) -> tuple[list[ChordSpan], list[tuple[int, int, int | None]]]:
    """곡 중간에 아무 음도 없어 코드를 못 읽은 마디(N.C. — 멜로디만 있는 곡에서 멜로디가 쉬는 곳)를 채운다.
    마디마다 같은 8마디 묶음의 다른 4마디(뒤 먼저, 그다음 앞) → 8마디 앞 · 뒤의 같은 자리 진행을 옮겨 오고, 그것도 없으면
    앞 코드를 이어 끈다. 곡 앞뒤의 빈 곳(멜로디가 아직 안 나옴 · 끝난 뒤)과 FILL_MAX마디보다 긴 빈 곳은 그대로.
    lofi_groove 멜로디만 남기면 9~12마디가 비는데, 원래 곡은 9~12마디가 13~16마디와 같은 진행(Dm9 · G13 · Cmaj9)이었다.
    반환: (N.C.를 뺀 코드들, 채운 구간 [(시작 마디, 끝 마디, 옮겨 온 거리(마디) 또는 None — 이어 끔 · 섞임)])."""
    known = [c for c in chords if c.root is not None]
    if not known:
        return [], []

    def whole(bar: int) -> bool:  # 그 마디(0부터)가 읽은 코드로 꽉 덮였나
        s, e = bar * bpb, (bar + 1) * bpb
        return sum(max(0.0, min(e, c.end) - max(s, c.start)) for c in known) >= bpb - 1e-6

    out = list(known)
    filled: list[tuple[int, int, int | None]] = []
    for gap in (c for c in chords if c.root is None and c.start > known[0].start and c.end < known[-1].end + 1e-6):
        a, b = math.ceil(gap.start / bpb - 1e-6), math.floor(gap.end / bpb + 1e-6)  # 빈 곳에 통째로 든 마디 [a, b)
        if not 1 <= b - a <= FILL_MAX:
            continue
        moves = set()
        for bar in range(a, b):
            near = [bar + PHRASE, bar - PHRASE] if (bar % BLOCK) < PHRASE else [bar - PHRASE, bar + PHRASE]
            near = [s for s in near if s // BLOCK == bar // BLOCK] + [bar - BLOCK, bar + BLOCK]
            src = next((s for s in near if s >= 0 and whole(s)), None)
            if src is None:  # 옮겨 올 진행이 없으면 앞 코드를 이어 끈다
                prev = max((c for c in out if c.start < bar * bpb - 1e-6), key=lambda c: c.start)
                out.append(replace(prev, start=bar * bpb, end=(bar + 1) * bpb))
                moves.add(None)
                continue
            shift = (bar - src) * bpb
            for c in known:
                s, e = max(c.start, src * bpb), min(c.end, (src + 1) * bpb)
                if e - s > 1e-6:
                    out.append(replace(c, start=s + shift, end=e + shift))
            moves.add(src - bar)
        filled.append((a + 1, b, moves.pop() if len(moves) == 1 else None))
    return sorted(out, key=lambda c: c.start), filled


def filled_text(filled: list[tuple[int, int, int | None]], start: float, end: float, bpb: float) -> str:
    """결과 요약에 적을 말: 멜로디가 쉬어 코드를 못 읽어 채운 마디 (그 범위에 걸친 것만)."""
    words = []
    for a, b, move in filled:
        if (b * bpb) <= start + 1e-6 or (a - 1) * bpb >= end - 1e-6:
            continue
        how = (f"{a + move}~{b + move}마디 진행을 되풀이해" if move is not None else "앞뒤 진행으로")
        words.append(f"{a}~{b}마디는 멜로디가 쉬어 코드를 못 읽어 {how} 채웠어요")
    return " · ".join(words)


def follow(
    project: Project, start: float, end: float | None, progression: list[str] | None, bars_per_chord: float = 1.0,
) -> tuple[list[ChordSpan], float, str]:
    """따라갈 코드와 끝 박, 채운 마디 말. progression(직접 말한 진행)이 있으면 곡 길이만큼 되풀이하고(곡이 한 바퀴보다 짧으면 한 바퀴),
    없으면 곡의 코드(분석)에서 [start, end)에 걸친 것 — 곡 중간에 멜로디가 쉬어 못 읽은 마디는 fill_rests로 채운다.
    반주 · 베이스 생성기가 같이 쓴다."""
    if progression:
        beats = bars_per_chord * project.beats_per_bar
        if end is None:
            end = max(song_end(project), start + len(progression) * beats)
        return place(progression, start, end, beats), end, ""
    a = song_chords(project)  # 화음 트랙이 없는(멜로디 · 베이스만 있는) 곡은 멜로디에 코드를 붙인다 (app/harmony/harmonize.py)
    bpb = project.beats_per_bar
    chords, filled = fill_rests(a.chords, bpb)
    end = a.chords[-1].end if end is None and a.chords else (end or 0.0)
    return ([c for c in chords if c.end > start + 1e-6 and c.start < end - 1e-6], end,
            filled_text(filled, start, end, bpb))


def song_end(project: Project) -> float:
    """곡 끝 박 (마디 단위로 올림). 음이 없으면 0."""
    last = max((n.start + n.duration for t in project.tracks for n in t.notes), default=0.0)
    bpb = project.beats_per_bar
    return math.ceil(last / bpb - 1e-9) * bpb


# ---------- 무늬 ----------

def _stab(chords: list[ChordSpan], voiced: list[list[int]], start: float, end: float, bpb: float, version: int,
          velocity: int, style: str | None = None) -> tuple[list[Note], list[ChordSpan]]:
    """리듬으로 화음을 끊어 친다. 장르가 있으면(4/4) 그 장르의 두 마디 리듬, 없으면 마디마다 같은 리듬(STAB_RHYTHMS).
    코드가 바뀌기 바로 앞에서 시작해 넘어가는 음은 다음 코드를 당겨 치고, 그 코드의 첫 박은 이어 끄니 다시 치지 않는다.
    리듬 자리가 없는 짧은 코드(반 마디 · 뒷박 리듬)도 첫 박에 한 번은 친다. 같은 높이가 겹치지 않게 다음 칠 때까지만 끈다.
    (음, 검사용 코드 — 당겨 친 자리부터 다음 코드로 본다)."""
    patterns = GENRE_COMP.get(style or "") if abs(bpb - 4) < 1e-6 else None
    if patterns:
        rhythm, cycle = patterns[(version - 1) % len(patterns)], 2 * bpb
    else:
        rhythm, cycle = tuple((o, d) for o, d in STAB_RHYTHMS[(version - 1) % len(STAB_RHYTHMS)] if o < bpb - 1e-6), bpb
    grid = sorted(k * cycle + o for k in range(int(start // cycle), math.ceil(end / cycle)) for o, _ in rhythm)
    lengths = {round(k * cycle + o, 4): d for k in range(int(start // cycle), math.ceil(end / cycle)) for o, d in rhythm}
    hits: list[tuple[float, int, bool]] = []  # (시작, 코드 번호, 당겨 침)
    pushed: set[int] = set()
    for h in grid:
        if not start - 1e-6 <= h < end - 1e-6:
            continue
        i = next((k for k, c in enumerate(chords) if c.start - 1e-6 <= h < c.end - 1e-6), None)
        if i is None:
            continue
        nxt = i + 1 if i + 1 < len(chords) and abs(chords[i + 1].start - chords[i].end) < 1e-6 else None
        if (nxt is not None and chords[i].end - h <= PUSH + 1e-6 and h + lengths[round(h, 4)] > chords[i].end + 1e-6
                and chords[i].end < end - 1e-6):
            hits.append((h, nxt, True))
            pushed.add(nxt)
        elif not (i in pushed and abs(h - chords[i].start) < 1e-6):
            hits.append((h, i, False))
    for k, c in enumerate(chords):  # 리듬 자리가 하나도 없는 코드
        if k not in pushed and not any(i == k for _, i, _ in hits) and c.start < end - 1e-6 and c.end > start + 1e-6:
            hits.append((max(c.start, start), k, False))
    hits.sort()
    rng = Random(version)
    notes: list[Note] = []
    spans = [replace(c) for c in chords]
    for n, (h, i, push) in enumerate(hits):
        stop = min(chords[i].end, end, hits[n + 1][0] if n + 1 < len(hits) else end)
        length = min(lengths.get(round(h, 4), 1.0), stop - h)
        if push:
            spans[i - 1].end, spans[i].start = h, h
        vel = velocity + (ACCENT if abs(h % bpb) < 1e-6 else 0) + (PUSH_ACCENT if push else 0) + rng.randint(-HUMAN, HUMAN)
        vel = max(1, min(127, vel))
        notes += [Note(pitch=p, start=round(h, 4), duration=round(length, 4), velocity=vel) for p in voiced[i]]
    return notes, spans


def _arpeggio(chords: list[ChordSpan], voiced: list[list[int]], start: float, end: float, version: int,
              velocity: int) -> list[Note]:
    """코드마다 고른 음을 아래부터 한 음씩 (버전마다 올라가기 · 오르내리기, 8분 · 16분음표). 코드가 바뀌면 처음부터."""
    order, rate = ARP_STYLES[(version - 1) % len(ARP_STYLES)]
    notes = []
    for c, v in zip(chords, voiced):
        seq = v + v[-2:0:-1] if order == "updown" else v
        s, e = max(c.start, start), min(c.end, end)
        k, t = 0, s
        while t < e - 1e-6:
            notes.append(Note(pitch=seq[k % len(seq)], start=round(t, 4), duration=round(min(rate, e - t), 4), velocity=velocity))
            k, t = k + 1, t + rate
    return notes


def _with_root(voicing: list[int], chord: ChordSpan, floor: int) -> list[int]:
    """맨 아래에 근음을 받친다 (곡에 음이 하나도 없어 베이스가 없을 때). 악기가 못 내는 높이면 그대로."""
    low = min(voicing)
    root = low - 1 - (low - 1 - chord.root) % 12
    return [root, *voicing] if root >= floor else voicing


def _bounds(window: tuple[int, int], height: str | None, floor: int, ceil: int) -> tuple[int, int]:
    """음역에 높이 말(low · high)을 더하고, 고른 악기로 낼 수 있는 높이 안으로 (보이싱이 GAP만큼 넘칠 수 있어 그만큼 안쪽)."""
    shift = HEIGHT_SHIFT.get(height or "mid", 0)
    low, high = max(LOW - 7, window[0] + shift), min(HIGH + 7, window[1] + shift)
    low, high = max(low, floor + GAP), min(high, ceil - GAP)
    if high - low < 12:
        low, high = max(floor + GAP, min(low, ceil - GAP - 12)), min(ceil - GAP, max(high, floor + GAP + 12))
    return low, high


def _windows(project: Project, chords: list[ChordSpan], start: float, end: float, height: str | None,
             floor: int, ceil: int) -> tuple[list[ChordSpan], list[tuple[int, int]]]:
    """코드마다 음역. PHRASE마디 묶음마다 그 안의 멜로디로 정한다 — 곡 전체로 한 번만 정하면 멜로디가 옥타브를 내려
    되풀이하는 곡에서 낮은 쪽에 맞춰져 높은 쪽 멜로디와 겹쳤다. 묶음에 멜로디가 없으면 곡 전체로.
    묶음 경계를 넘는 코드는 음역이 바뀌면 경계에서 나눈다 (4~5마디에 걸친 Gm이 4마디 음역 그대로 낮은 멜로디와 겹쳤다).
    (나눈 코드들, 코드마다 음역)."""
    whole = pad_window(project, start, end)
    melody = melody_ids(project)
    span = PHRASE * project.beats_per_bar
    cache: dict[int, tuple[int, int]] = {}

    def window(k: int) -> tuple[int, int]:
        if k not in cache:
            s, e = max(start, k * span), min(end, (k + 1) * span)
            sings = any(n.start < e and n.start + n.duration > s for t in project.tracks if t.id in melody for n in t.notes)
            cache[k] = _bounds(pad_window(project, s, e) if sings else whole, height, floor, ceil)
        return cache[k]

    out_chords, out = [], []
    for c in chords:
        piece, e = c, min(c.end, end)
        while True:
            k = int((max(piece.start, start) + 1e-6) // span)
            last = k
            while (last + 1) * span < e - 1e-6 and window(last + 1) == window(k):
                last += 1
            cut = (last + 1) * span
            if cut >= e - 1e-6:
                out_chords.append(piece)
                out.append(window(k))
                break
            out_chords.append(replace(piece, end=cut))  # 다음 묶음에서 음역이 바뀐다 — 나머지는 새 음역으로
            out.append(window(k))
            piece = replace(piece, start=cut)
    return out_chords, out


def chord_notes(
    project: Project, start: float = 0.0, end: float | None = None, height: str | None = None,
    program: int | None = None, version: int = 1, pattern: str = "pad", progression: list[str] | None = None,
    bars_per_chord: float = 1.0, style: str | None = None,
) -> tuple[list[Note], PadPlan]:
    """[start, end) 박에 코드대로 화음 음. 코드는 progression(직접 말한 진행)이 있으면 그것을 곡 길이만큼 되풀이하고
    (곡이 진행 한 바퀴보다 짧으면 한 바퀴), 없으면 곡의 코드(분석) — 코드가 없는 구간(N.C.)은 비운다.
    program이 없으면 곡 편성에 맞춰 고른다. 곡에 음이 하나도 없으면(드럼만) 맨 아래에 근음을 받친다.
    progression의 모르는 코드 이름은 ValueError."""
    program = comp_program(project, pattern) if program is None else program
    bpb = project.beats_per_bar
    chords, end, filled = follow(project, start, end, progression, bars_per_chord)
    if not chords:
        return [], PadPlan(program, 0, 0)
    # 장르 색(7 · 9음)은 곡에서 읽거나 붙인 코드에만 — 사용자가 말한 진행은 말한 그대로
    chords, colors = _colored(project, chords, start, end) if style in COLORFUL and not progression else (chords, "")
    floor, ceil = playable(program)
    chords, windows = _windows(project, chords, start, end, height, floor, ceil)
    played = [n.velocity for t in project.tracks if not t.is_drum for n in t.notes]
    # 장르 리듬은 길게 끄는 음이 많아 패드만큼 줄인다 (짧게 끊어 치는 기본 리듬만 덜 줄임)
    softer = STAB_SOFTER if pattern == "stab" and style not in GENRE_COMP else SOFTER
    velocity = max(35, min(90, round(median(played) * softer))) if played else 64
    voiced = voicings(chords, windows, version, heard_with(project, chords, start, end))
    if not any(n.start < end and n.start + n.duration > start for t in project.tracks if not t.is_drum for n in t.notes):
        voiced = [_with_root(v, c, floor) for c, v in zip(chords, voiced)]
    chords, eased = _eased(chords, voiced)
    if pattern == "stab":
        notes, chords = _stab(chords, voiced, start, end, bpb, version, velocity, style)
    elif pattern == "arpeggio":
        notes = _arpeggio(chords, voiced, start, end, version, velocity)
    else:
        notes = []
        for c, voicing in zip(chords, voiced):
            s, e = max(c.start, start), min(c.end, end)
            notes += [Note(pitch=p, start=round(s, 4), duration=round(e - s, 4), velocity=velocity) for p in voicing]
    pitches = [n.pitch for n in notes]
    names = {"stab": STAB_NAMES, "arpeggio": ARP_NAMES}.get(pattern)
    rhythm = names[(version - 1) % len(names)] if names else ""
    if pattern == "stab" and style in GENRE_COMP and abs(bpb - 4) < 1e-6:
        rhythm = GENRE_COMP_NAMES[style]
    feel = " · ".join(x for x in (rhythm, colors, eased, filled) if x)
    return notes, PadPlan(program, min(pitches), max(pitches), chords, feel)


def _colored(project: Project, chords: list[ChordSpan], start: float, end: float) -> tuple[list[ChordSpan], str]:
    """음계 화음(3화음)에 7 · 9음을 더한다 (소울 · 재즈 · 로파이처럼 COLORFUL 장르). 더한 음이 그동안 울리는 다른 트랙 음과
    반음(위아래 어느 쪽이든)으로 OVERLAP박 넘게 같이 울리면 한 단계 덜어 낸다 (Gm9 → Gm7). 짧은 음도 합쳐 센다 — Sphinx는
    0.1~0.25박짜리 Ab을 마디마다 여러 번 찍어, 하나씩 보면 지나가는 음이지만 Gm9의 A와 계속 부딪힌다. (코드들, 결과에 적을 말)."""
    key = song_chords(project).key
    notes = [n for t in project.tracks if not t.is_drum for n in t.notes]
    out, names = [], []
    for c in chords:
        rich = TENSION[key.mode].get((c.root - key.tonic) % 12) if c.root is not None and c.quality in ("", "m") else None
        if rich is None or (rich.startswith("m") and not rich.startswith("maj")) != (c.quality == "m"):
            out.append(c)
            continue
        s, e = max(c.start, start), min(c.end, end)
        heard: dict[int, float] = {}
        for n in notes:
            ov = min(e, n.start + n.duration) - max(s, n.start)
            if ov > 0:
                heard[n.pitch % 12] = heard.get(n.pitch % 12, 0.0) + ov
        base = {(c.root + i) % 12 for i in INTERVALS[c.quality]}
        for q in LIGHTER.get(rich, [rich]):
            added = {(c.root + i) % 12 for i in INTERVALS[q]} - base
            if all(heard.get((pc + 1) % 12, 0.0) < OVERLAP and heard.get((pc - 1) % 12, 0.0) < OVERLAP for pc in added):
                head, *slash = c.name.split("/")
                head = head[: len(head) - len(c.quality)] if c.quality else head
                c = replace(c, quality=q, name="/".join([head + q, *slash]))
                names.append(c.name)
                break
        out.append(c)
    return out, ("7 · 9음 " + " · ".join(dict.fromkeys(names))) if names else ""


def _eased(chords: list[ChordSpan], voiced: list[list[int]]) -> tuple[list[ChordSpan], str]:
    """부딪힘을 피하느라 바뀐 코드: 3음 대신 4음을 넣은 코드는 sus4 이름으로 바꾸고(검사가 4음을 코드 밖으로 보지 않게),
    3음 · 7음을 뺀 코드와 함께 결과에 적을 말 (LLM이 "코드대로 깔았다"고만 하지 않게)."""
    out, sus, thin = [], 0, 0
    for c, v in zip(chords, voiced):
        have = {p % 12 for p in v}
        if c.quality in SUS and (c.root + 4) % 12 not in have and (c.root + 5) % 12 in have:
            base, *slash = c.name.split("/")
            c = replace(c, quality=SUS[c.quality], name="/".join([base + "sus4", *slash]))
            sus += 1
        elif {pc for pc in _must(c) if (pc - c.root) % 12 not in (0, 7)} - have:
            thin += 1
        out.append(c)
    words = ([f"멜로디와 부딪히지 않게 sus4로 바꾼 코드 {sus}개"] if sus else []) + ([f"3음 · 7음을 뺀 코드 {thin}개"] if thin else [])
    return out, " · ".join(words)


def pad_notes(
    project: Project, start: float = 0.0, end: float | None = None, height: str | None = None,
    program: int | None = None, version: int = 1,
) -> tuple[list[Note], PadPlan]:
    """곡의 코드대로 패드 (chord_notes의 pattern pad)."""
    return chord_notes(project, start, end, height, program, version)
