"""Basic Pitch(nmp.onnx, Apache 2.0)를 합성 샘플팩 루프(synth_data.py)로 이어서 학습.
CQT 필터는 고정하고 뒤쪽 합성곱 6층(파라미터 16,700개)만 학습한다. 체크포인트마다 학습한 가중치를 원래 nmp.onnx 그래프에
끼운 .onnx를 함께 저장한다(patch_onnx.py — PyTorch로 다시 내보내면 같은 모델이 7배 느려진다).

앱에 넣은 nmp_cowrite.onnx (2026-11-20):
  python train.py --onnx nmp.onnx --out run2 --steps 1000 --lr 1e-4 --batch 16 --workers 5 --banks <fetch_samples.py 폴더> --save_every 100
  의 800걸음 (CPU 8코어 약 40분). 더 오래 하면 합성 소리에 맞춰져 실제 루프 점수가 떨어진다 — 체크포인트마다 재서 고른다.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset

sys.path.insert(0, str(Path(__file__).parent))
import synth_data as G  # noqa: E402

SR = 22050
HOP = 256
WIN = SR * 2 - HOP  # 43844
FRAMES = 172
N_PITCH = 88
BINS = 264  # 반음당 3칸
MIDI_OFFSET = 21


def labels_for(notes, start_s: float):
    """창(시작 start_s초)의 정답: 음 활성(172×88) · 시작점(172×88) · 윤곽(172×264, 가운데 칸 1 · 옆 칸 0.5)."""
    note = np.zeros((FRAMES, N_PITCH), np.float32)
    onset = np.zeros((FRAMES, N_PITCH), np.float32)
    contour = np.zeros((FRAMES, BINS), np.float32)
    fps = SR / HOP
    for s, e, p in notes:
        k = p - MIDI_OFFSET
        if not 0 <= k < N_PITCH:
            continue
        a = int(round((s - start_s) * fps))
        b = int(round((e - start_s) * fps))
        if b < 0 or a >= FRAMES:
            continue
        lo, hi = max(a, 0), min(max(b, a + 1), FRAMES)
        note[lo:hi, k] = 1
        c = 3 * k + 1
        contour[lo:hi, c] = 1
        contour[lo:hi, c - 1] = np.maximum(contour[lo:hi, c - 1], 0.5)
        if c + 1 < BINS:
            contour[lo:hi, c + 1] = np.maximum(contour[lo:hi, c + 1], 0.5)
        if 0 <= a < FRAMES:
            onset[a, k] = 1
    return note, onset, contour


class Loops(IterableDataset):
    def __init__(self, seed: int, banks: str | None, share: float = 0.5):
        self.seed, self.banks, self.share = seed, banks, share

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        wid = info.id if info else 0
        rng = np.random.default_rng(self.seed * 1000 + wid + int(time.time()) % 100000)
        G.SAMPLED_SHARE = self.share  # 작업자는 새 프로세스라 여기서 정한다
        if self.banks and not G._BANKS:
            G.load_banks(Path(self.banks))
        while True:
            audio, notes = G.make_loop(rng)
            # 루프 하나에서 창 2개
            for _ in range(2):
                if len(audio) <= WIN:
                    x = np.pad(audio, (0, WIN - len(audio)))
                    st = 0
                else:
                    st = int(rng.integers(0, len(audio) - WIN))
                    x = audio[st : st + WIN]
                n, o, c = labels_for(notes, st / SR)
                yield torch.from_numpy(x[:, None].copy()), torch.from_numpy(n), torch.from_numpy(o), torch.from_numpy(c)


def bce(pred, target, pos_weight=None):
    pred = pred.clamp(1e-6, 1 - 1e-6)
    if pos_weight is None:
        return -(target * torch.log(pred) + (1 - target) * torch.log(1 - pred)).mean()
    return -(pos_weight * target * torch.log(pred) + (1 - pos_weight) * (1 - target) * torch.log(1 - pred)).mean()


def freeze_frontend(model):
    """CQT · 다운샘플 필터(고정된 신호 처리)는 학습하지 않는다."""
    for n, p in model.named_parameters():
        if "cq_t2010" in n:
            p.requires_grad = False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--banks", default=None)
    ap.add_argument("--save_every", type=int, default=500)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--train_only", default="", help="이 이름 조각이 든 파라미터만 학습 (쉼표로)")
    ap.add_argument("--sampled_share", type=float, default=0.5)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(max(1, os.cpu_count() - args.workers))

    import bp_torch
    import patch_onnx

    model = bp_torch.load(args.onnx)
    if args.resume:
        model.load_state_dict(torch.load(args.resume))
    model.train()
    freeze_frontend(model)
    if args.train_only:
        keys = [k for k in args.train_only.split(",") if k]
        for n, p in model.named_parameters():
            if not any(k in n for k in keys):
                p.requires_grad = False
    params = [p for p in model.parameters() if p.requires_grad]
    print("학습할 파라미터", sum(p.numel() for p in params), flush=True)
    opt = torch.optim.Adam(params, lr=args.lr)
    dl = DataLoader(Loops(1, args.banks, args.sampled_share), batch_size=args.batch, num_workers=args.workers, persistent_workers=True)
    t0 = time.time()
    run = {"note": 0.0, "onset": 0.0, "contour": 0.0}
    for step, (x, n, o, c) in enumerate(dl, start=1):
        outs = model(x)
        onset, note, contour = outs  # 그래프 출력 순서: :2 시작점 · :1 음 · :0 윤곽
        ln, lo_, lc = bce(note, n), bce(onset, o, torch.tensor(0.95)), bce(contour, c)
        loss = ln + lo_ + lc
        opt.zero_grad()
        loss.backward()
        opt.step()
        for k, v in (("note", ln), ("onset", lo_), ("contour", lc)):
            run[k] = 0.98 * run[k] + 0.02 * float(v) if step > 1 else float(v)
        if step % 50 == 0:
            print(f"{step:6d} {time.time() - t0:7.0f}s 음 {run['note']:.4f} 시작점 {run['onset']:.4f} 윤곽 {run['contour']:.4f}", flush=True)
        if step % args.save_every == 0 or step == args.steps:
            torch.save(model.state_dict(), out / f"step{step}.pt")
            patch_onnx.write(model, args.onnx, out / f"step{step}.onnx")
        if step >= args.steps:
            break


if __name__ == "__main__":
    main()
