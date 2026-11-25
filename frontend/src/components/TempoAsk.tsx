import { useState } from 'react'
import { useProject } from '../store/project'

/** 이름에 템포가 없는 오디오를 열 때: 템포를 알면 넣고, 모르면 소리로 짐작 */
export function TempoAsk() {
  const { askTempo, upload, cancelAskTempo } = useProject()
  const [text, setText] = useState('')
  const [mono, setMono] = useState(false)
  if (!askTempo) return null
  const bpm = Number(text)
  const valid = text.trim() !== '' && bpm >= 40 && bpm <= 300
  const close = () => {
    setText('')
    setMono(false)
    cancelAskTempo()
  }
  const open = (value: number | null) => {
    setText('')
    setMono(false)
    void upload(askTempo, value, mono)
  }
  return (
    <>
      <button className="scrim" aria-label="템포 묻기 닫기" onClick={close} />
      <form
        className="tempo-ask"
        role="dialog"
        aria-label="오디오 템포"
        onSubmit={(e) => {
          e.preventDefault()
          if (valid) open(bpm)
        }}
        onKeyDown={(e) => e.key === 'Escape' && close()}
      >
        <div>
          <h2>이 오디오의 템포를 아세요?</h2>
          <p className="hint">
            <strong>{askTempo.name}</strong> — 이름에 템포가 없어요. 알면 넣어 주세요, 마디 줄이 정확히 맞아요. 모르면 소리로 짐작해요
            (샘플팩 루프 90곡 중 77곡을 맞혔고, 틀려도 들리는 박자는 그대로예요).
          </p>
        </div>
        <label className="tempo-ask-field">
          <input
            type="number"
            inputMode="decimal"
            min={40}
            max={300}
            step="any"
            placeholder="예: 120"
            value={text}
            onChange={(e) => setText(e.target.value)}
            autoFocus
          />
          <span>BPM</span>
        </label>
        <label className="tempo-ask-mono">
          <input type="checkbox" checked={mono} onChange={(e) => setMono(e.target.checked)} />
          <span>허밍 · 노래예요 (한 번에 한 음씩 읽기)</span>
        </label>
        <div className="tempo-ask-actions">
          <button type="button" onClick={close}>
            취소
          </button>
          <button type="button" onClick={() => open(null)}>
            모르겠어요 · 짐작하기
          </button>
          <button type="submit" className="primary" disabled={!valid}>
            이 템포로 열기
          </button>
        </div>
      </form>
    </>
  )
}
