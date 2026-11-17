/**
 * 귀에 들리는 크기 (ITU-R BS.1770 K-가중 · 400ms 창 = "순간 크기"). 한 번 친 소리는 가장 큰 창의 값.
 * 최대 크기(피크)로 맞추면 짧게 "틱" 하는 햇이 길게 울리는 킥만큼 크게 들려서, 내 드럼 샘플을 이걸로 맞춘다 (2026-11-17).
 * Tone.js를 쓰지 않는 계산만 — 노드에서 바로 돌려 재 볼 수 있게.
 */

type Biquad = { b: [number, number, number]; a: [number, number, number] }

/** K-가중 두 단(고음 선반 + 저음 차단) 계수. 어떤 샘플 레이트에서도 48kHz 표준 계수와 같은 모양 (libebur128 방식) */
function kWeight(rate: number): [Biquad, Biquad] {
  let K = Math.tan((Math.PI * 1681.974450955533) / rate)
  let Q = 0.7071752369554196
  const Vh = Math.pow(10, 3.999843853973347 / 20)
  const Vb = Math.pow(Vh, 0.4996667741545416)
  let a0 = 1 + K / Q + K * K
  const shelf: Biquad = {
    b: [(Vh + (Vb * K) / Q + K * K) / a0, (2 * (K * K - Vh)) / a0, (Vh - (Vb * K) / Q + K * K) / a0],
    a: [1, (2 * (K * K - 1)) / a0, (1 - K / Q + K * K) / a0],
  }
  K = Math.tan((Math.PI * 38.13547087602444) / rate)
  Q = 0.5003270373238773
  a0 = 1 + K / Q + K * K
  const highpass: Biquad = { b: [1, -2, 1], a: [1, (2 * (K * K - 1)) / a0, (1 - K / Q + K * K) / a0] }
  return [shelf, highpass]
}

function filter(x: Float32Array, { b, a }: Biquad): Float32Array {
  const y = new Float32Array(x.length)
  let x1 = 0, x2 = 0, y1 = 0, y2 = 0
  for (let i = 0; i < x.length; i++) {
    const v = b[0] * x[i] + b[1] * x1 + b[2] * x2 - a[1] * y1 - a[2] * y2
    x2 = x1; x1 = x[i]; y2 = y1; y1 = v
    y[i] = v
  }
  return y
}

const WINDOW_S = 0.4
const HOP_S = 0.02
const MAX_S = 3 // 긴 샘플(원샷으로 넣은 루프 등)은 앞 3초만

const toLufs = (meanSquare: number) => -0.691 + 10 * Math.log10(meanSquare)

/**
 * 채널마다 K-가중을 건 소리의 제곱을 채널 수로 나눠 더한 누적합. 앞 n개 + 끝에 tail개만큼 조용한 꼬리.
 * 채널마다 따로 재서 평균(BS.1770은 채널 합 — 모노 파일과 같은 잣대가 되게 채널 수로 나눔). 파형을 먼저 평균 내면
 * 좌우가 다른 스테레오는 작게, ffmpeg -ac 1처럼 합치면 좌우가 같은 스테레오는 3dB 크게 재진다
 */
function weightedSums(channels: Float32Array[], rate: number, n: number, tail: number): Float64Array {
  const [shelf, highpass] = kWeight(rate)
  const sq = new Float64Array(n + tail)
  for (const ch of channels) {
    const x = new Float32Array(n + tail)
    x.set(ch.subarray(0, n))
    const y = filter(filter(x, shelf), highpass)
    for (let i = 0; i < y.length; i++) sq[i] += (y[i] * y[i]) / channels.length
  }
  const sums = new Float64Array(sq.length + 1)
  for (let i = 0; i < sq.length; i++) sums[i + 1] = sums[i] + sq[i]
  return sums
}

/** 400ms 창(hop 간격)마다 평균 제곱 */
function blocks(sums: Float64Array, win: number, hop: number): number[] {
  const out: number[] = []
  for (let i = 0; i + win < sums.length; i += hop) out.push((sums[i + win] - sums[i]) / win)
  return out
}

const channelsOf = (buffer: AudioBuffer) =>
  Array.from({ length: buffer.numberOfChannels }, (_, c) => buffer.getChannelData(c))

/** 한 번 친 소리: 채널들 → 가장 큰 400ms 창의 크기 (LUFS). 소리가 없으면 -Infinity */
export function peakLoudness(channels: Float32Array[], rate: number): number {
  const n = Math.min(channels[0]?.length ?? 0, Math.round(MAX_S * rate))
  const win = Math.round(WINDOW_S * rate)
  // 끝에 창 하나만큼 조용한 꼬리를 붙여 짧은 소리도 창 하나를 채운다
  const best = Math.max(0, ...blocks(weightedSums(channels, rate, n, win), win, Math.round(HOP_S * rate)))
  return best > 0 ? toLufs(best) : -Infinity
}

/** 루프 · 곡: 채널들 → 통합 크기 (LUFS, BS.1770 — 400ms 창 75% 겹침, -70 LUFS · 평균 -10dB 아래 창은 빼고 평균) */
export function integratedLoudness(channels: Float32Array[], rate: number): number {
  const n = channels[0]?.length ?? 0
  const win = Math.round(WINDOW_S * rate)
  const all = blocks(weightedSums(channels, rate, n, 0), win, Math.round(win / 4))
  const loud = all.filter((m) => m > 0 && toLufs(m) > -70)
  if (!loud.length) return -Infinity
  const gate = toLufs(loud.reduce((a, b) => a + b, 0) / loud.length) - 10
  const kept = loud.filter((m) => toLufs(m) > gate)
  return toLufs(kept.reduce((a, b) => a + b, 0) / kept.length)
}

/** AudioBuffer(브라우저가 읽은 샘플) — 한 번 친 소리의 크기 */
export const bufferLoudness = (buffer: AudioBuffer) => peakLoudness(channelsOf(buffer), buffer.sampleRate)
/** AudioBuffer — 루프 전체의 크기 */
export const bufferIntegratedLoudness = (buffer: AudioBuffer) => integratedLoudness(channelsOf(buffer), buffer.sampleRate)
