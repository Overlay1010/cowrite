import * as Tone from 'tone'

/**
 * 믹스 크기 (재생 · 음원 내보내기가 같이 쓴다).
 * 악기는 FluidR3 크기에 맞춰 두었는데(levels.ts · hqset.ts) 드럼 샘플은 맞춘 적이 없어서, 재생과 같은 규칙으로 트랙마다 크기(LUFS)를
 * 재 보니 드럼이 멜로디보다 18~21dB 컸다 — "드럼 소리만 너무 크고 다른 피아노 코드는 너무 작게 들린다" (2026-11-17).
 * 드럼을 DRUM_DB만큼 줄이고(소울 · 팝 · 록 · 펑크 · 재즈 · 로파이에서 드럼 전체가 멜로디 +1~5dB), 줄어든 만큼 마스터를 올려
 * 곡 전체 크기는 예전과 비슷하게(-22 ~ -19 LUFS) 두었다. 트랙이 많아 넘치는 곳은 리미터가 막는다.
 */
export const MASTER_DB = 6
export const DRUM_DB = -17
const LIMIT_DB = -1

/** 이 오디오 공간(지금 context — 음원 내보내기면 오프라인)에 마스터: 볼륨 → 리미터 → 스피커 */
export function createMaster(): Tone.Volume {
  const limiter = new Tone.Limiter(LIMIT_DB).toDestination()
  return new Tone.Volume(MASTER_DB).connect(limiter)
}
