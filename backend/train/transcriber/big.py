"""큰 오디오 → 음표 모델 시험 (2026-11-22).

Basic Pitch는 학습되는 숫자가 1만 7천 개뿐이라(이어서 학습할 때 거의 다 학습했다) 더 키우지 않으면 정확도가 안 오른다.
여기서는 그 위에 넓은 합성곱 + 시간 축 트랜스포머를 얹는다.

- 입력: 고정한 Basic Pitch(nmp_cowrite.onnx)의 화음 쌓은 CQT(8배음 × 264칸)와 그 모델의 음 · 시작점 · 윤곽 출력
- 출력: 음 · 시작점 활성도 (172칸 × 88음). 처음에는 Basic Pitch 출력과 똑같고(고치는 층을 0에서 시작) 고칠 것만 배운다
- 앱과 같은 입출력 이름의 ONNX로 내보내 `python -m eval.transcribe <onnx>`로 바로 잰다

  python big.py --base ../../app/midi_io/basic_pitch/nmp_cowrite.onnx --out runbig --steps 3000 --banks banks

GPU (RunPod, runpod/README.md): --slakh <slakh_data.py 폴더> --surge <surge_render.py 폴더> — 구매한 샘플팩은 쓰지 않는다.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, IterableDataset

sys.path.insert(0, str(Path(__file__).parent))
import synth_data as G  # noqa: E402
from train import FRAMES, SR, WIN, Loops, bce, labels_for  # noqa: E402


class Mixed(IterableDataset):
    """합성 루프 + 샘플팩(cymatics_data): real 비율만큼 A 묶음 실제 stem, midi 비율만큼 MIDI 팩 곡을 신스로 렌더."""

    def __init__(self, seed: int, banks: str | None, cym: str, real: float, midi: float, share: float = 0.5):
        self.seed, self.banks, self.cym, self.real, self.midi, self.share = seed, banks, cym, real, midi, share

    def __iter__(self):
        import cymatics_data as C

        info = torch.utils.data.get_worker_info()
        wid = info.id if info else 0
        rng = np.random.default_rng(self.seed * 1000 + wid + int(time.time()) % 100000)
        G.SAMPLED_SHARE = self.share
        if self.banks and not G._BANKS:
            G.load_banks(Path(self.banks))
        C.load(Path(self.cym))
        while True:
            r = rng.random()
            if r < self.real:
                audio, notes = C.real_loop(rng, SR)
            elif r < self.real + self.midi:
                audio, notes = G.make_loop(rng, C.midi_music)
            else:
                audio, notes = G.make_loop(rng)
            for _ in range(2):
                if len(audio) <= WIN:
                    x, st = np.pad(audio, (0, WIN - len(audio))), 0
                else:
                    st = int(rng.integers(0, len(audio) - WIN))
                    x = audio[st : st + WIN]
                n, o, c = labels_for(notes, st / SR)
                yield torch.from_numpy(x[:, None].copy()), torch.from_numpy(n), torch.from_numpy(o), torch.from_numpy(c)

class Blend(IterableDataset):
    """GPU 학습용 (구매한 샘플팩 없이): slakh_share만큼 Slakh2100, surge_share만큼 Surge 렌더 루프, 나머지는 합성 루프."""

    def __init__(self, seed: int, banks: str | None, share: float, slakh: str | None, slakh_share: float,
                 surge: str | None, surge_share: float):
        self.seed, self.banks, self.share = seed, banks, share
        self.slakh, self.slakh_share, self.surge, self.surge_share = slakh, slakh_share, surge, surge_share

    def __iter__(self):
        import slakh_data as S

        info = torch.utils.data.get_worker_info()
        wid = info.id if info else 0
        rng = np.random.default_rng(self.seed * 1000 + wid + int(time.time()) % 100000)
        G.SAMPLED_SHARE = self.share
        if self.banks and not G._BANKS:
            G.load_banks(Path(self.banks))
        slakh = S.Slakh(Path(self.slakh)) if self.slakh else None
        surge = sorted(Path(self.surge).glob("*.npz")) if self.surge else []
        while True:
            r = rng.random()
            if slakh and r < self.slakh_share:
                audio, notes = slakh.excerpt(rng)
            elif surge and r < self.slakh_share + self.surge_share:
                z = np.load(surge[int(rng.integers(0, len(surge)))])
                audio, notes = z["audio"].astype(np.float32), [(float(s), float(e), int(p)) for s, e, p in z["labels"]]
            else:
                audio, notes = G.make_loop(rng)
            for _ in range(2):
                if len(audio) <= WIN:
                    x, st = np.pad(audio, (0, WIN - len(audio))), 0
                else:
                    st = int(rng.integers(0, len(audio) - WIN))
                    x = audio[st : st + WIN]
                n, o, c = labels_for(notes, st / SR)
                yield torch.from_numpy(x[:, None].copy()), torch.from_numpy(n), torch.from_numpy(o), torch.from_numpy(c)


HCQT_NODE = "model_1_harmonic_stacking_diff_strided_slice_7"


def frozen_base(path: str) -> torch.fx.GraphModule:
    """Basic Pitch를 읽어 출력에 화음 쌓은 CQT를 더한다: (시작점, 음, 윤곽, hcqt[B,172,264,8])."""
    import bp_torch

    m = bp_torch.load(path)
    nodes = [n for n in m.graph.nodes if n.name.startswith(HCQT_NODE)]
    assert len(nodes) == 1, nodes
    out = next(n for n in m.graph.nodes if n.op == "output")
    out.args = ((*out.args[0], nodes[0]),)
    m.recompile()
    m.eval()
    for p in m.parameters():
        p.requires_grad = False
    return m


def logit(p: torch.Tensor) -> torch.Tensor:
    p = p.clamp(1e-4, 1 - 1e-4)
    return torch.log(p) - torch.log1p(-p)


class ConvBlock(nn.Sequential):
    def __init__(self, cin, cout, k, stride=(1, 1), dilation=(1, 1)):
        pad = ((k[0] - 1) // 2 * dilation[0], 0 if stride[1] > 1 else (k[1] - 1) // 2 * dilation[1])
        super().__init__(nn.Conv2d(cin, cout, k, stride, pad, dilation), nn.GroupNorm(8, cout), nn.GELU())


class Big(nn.Module):
    """hcqt(8) + 윤곽(1) → 합성곱(264칸 → 88음) → 음마다 특징 + 프레임 전체를 보는 트랜스포머 → 음 · 시작점 고침값."""

    def __init__(self, ch: int = 48, d: int = 256, layers: int = 4, heads: int = 4):
        super().__init__()
        # 264칸에서 도는 층이 학습 시간 대부분이라 얇게 두고 곧바로 88음으로 줄인다
        self.conv = nn.Sequential(
            ConvBlock(9, 16, (5, 5)),
            ConvBlock(16, ch, (3, 3), stride=(1, 3)),  # 264 → 88 (반음 3칸 → 1음)
            ConvBlock(ch, ch, (3, 3), dilation=(1, 12)),  # 옥타브 위아래를 같이 본다
            ConvBlock(ch, ch, (5, 3)),
        )
        c = ch + 2  # + Basic Pitch 음 · 시작점 로짓
        self.squeeze = nn.Conv2d(c, 16, 1)
        self.proj = nn.Linear(16 * 88, d)
        self.pos = nn.Parameter(torch.zeros(1, FRAMES, d))
        layer = nn.TransformerEncoderLayer(d, heads, 2 * d, dropout=0.1, batch_first=True, norm_first=True, activation="gelu")
        self.tf = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.global_head = nn.Linear(d, 2 * 88)  # 음 · 시작점
        self.local_head = nn.Conv2d(c + 2, 2, (3, 1), padding=(1, 0))  # 음마다 (지역 특징 + 전체에서 받은 값)
        self.back = nn.Linear(d, 2 * 88)
        for m in (self.global_head, self.local_head):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, hcqt, contour, note_p, onset_p):
        b = hcqt.shape[0]
        x = torch.cat([hcqt.permute(0, 3, 1, 2), contour.unsqueeze(1)], 1)  # B,9,T,264
        x = self.conv(x)  # B,2ch,T,88
        # bf16에서는 1 - 1e-4가 1로 반올림돼 log(0) → nan. 로짓과 마지막 합은 fp32로
        nl, ol = logit(note_p.float()), logit(onset_p.float())
        x = torch.cat([x, nl.unsqueeze(1), ol.unsqueeze(1)], 1)  # B,c,T,88
        s = self.squeeze(x).permute(0, 2, 1, 3).reshape(b, FRAMES, -1)
        h = self.tf(self.proj(s) + self.pos)  # B,T,d
        g = self.global_head(h).view(b, FRAMES, 2, 88)
        back = self.back(h).view(b, FRAMES, 2, 88).permute(0, 2, 1, 3)  # B,2,T,88
        loc = self.local_head(torch.cat([x, back], 1))  # B,2,T,88
        with torch.autocast(hcqt.device.type, enabled=False):
            note = nl + g[:, :, 0].float() + loc[:, 0].float()
            onset = ol + g[:, :, 1].float() + loc[:, 1].float()
            return torch.sigmoid(onset), torch.sigmoid(note)


class Full(nn.Module):
    """앱이 부르는 모양: 오디오(B,43844,1) → (시작점, 음, 윤곽)."""

    def __init__(self, base, big):
        super().__init__()
        self.base, self.big = base, big

    def forward(self, audio):
        # 고정된 Basic Pitch(CQT 포함)는 bf16이면 정밀도가 떨어져 fp32로 돌린다 (학습하지 않아 비용이 작다)
        with torch.no_grad(), torch.autocast(audio.device.type, enabled=False):
            onset_p, note_p, contour, hcqt = self.base(audio.float())
        onset, note = self.big(hcqt, contour, note_p, onset_p)
        return onset, note, contour


def export(full: Full, path: Path) -> None:
    device = next(full.big.parameters()).device
    full.cpu().eval()
    torch.onnx.export(
        full,
        (torch.zeros(1, 43844, 1),),
        str(path),
        input_names=["serving_default_input_2:0"],
        output_names=["StatefulPartitionedCall:2", "StatefulPartitionedCall:1", "StatefulPartitionedCall:0"],
        dynamic_axes={"serving_default_input_2:0": {0: "b"}},
        opset_version=17,
        dynamo=False,
    )
    full.to(device)
    full.big.train()


def cpu_count() -> int:
    """컨테이너에 실제로 주어진 CPU 수. RunPod은 os.cpu_count()가 서버 전체(48)를 보여 주고 cgroup으로 6.8개만 준다."""
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()  # cgroup v2
        if quota != "max":
            return max(1, int(int(quota) / int(period)))
    except (OSError, ValueError):
        pass
    try:  # cgroup v1 (서버에 따라 이쪽 — 그대로 두면 128개로 본다)
        quota = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text())
        period = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text())
        if quota > 0:
            return max(1, quota // period)
    except (OSError, ValueError):
        pass
    return os.cpu_count() or 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--banks", default=None)
    ap.add_argument("--save_every", type=int, default=500)
    ap.add_argument("--sampled_share", type=float, default=0.5)
    ap.add_argument("--ch", type=int, default=48)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--cym", default=None, help="cymatics_data.py가 만든 목록 (없으면 합성만)")
    ap.add_argument("--real", type=float, default=0.3)
    ap.add_argument("--midi", type=float, default=0.3)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--slakh", default=None, help="slakh_data.py prepare가 만든 폴더 (GPU 학습)")
    ap.add_argument("--slakh_share", type=float, default=0.5)
    ap.add_argument("--surge", default=None, help="surge_render.py가 렌더한 폴더")
    ap.add_argument("--surge_share", type=float, default=0.3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max_minutes", type=float, default=0, help="이 시간이 지나면 저장하고 멈춤 (0이면 없음)")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(max(1, cpu_count() - args.workers))

    device = torch.device(args.device)
    full = Full(frozen_base(args.base), Big(args.ch, args.d, args.layers))
    if args.resume:
        full.big.load_state_dict(torch.load(args.resume, map_location="cpu"))
    full.to(device)
    amp = device.type == "cuda"
    print("장치", device, torch.cuda.get_device_name(0) if amp else "", flush=True)
    params = list(full.big.parameters())
    print("학습할 파라미터", sum(p.numel() for p in params), flush=True)
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    warm = 200
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s, args.steps) / args.steps))
    )
    if args.cym:
        data = Mixed(2, args.banks, args.cym, args.real, args.midi, args.sampled_share)
    elif args.slakh or args.surge:
        data = Blend(2, args.banks, args.sampled_share, args.slakh, args.slakh_share, args.surge, args.surge_share)
    else:
        data = Loops(2, args.banks, args.sampled_share)
    dl = DataLoader(data, batch_size=args.batch, num_workers=args.workers, persistent_workers=True, pin_memory=amp, prefetch_factor=4)
    t0 = time.time()
    run = {"note": 0.0, "onset": 0.0}
    full.big.train()
    pos = torch.tensor(0.95, device=device)
    for step, (x, n, o, _c) in enumerate(dl, start=1):
        x, n, o = x.to(device, non_blocking=True), n.to(device, non_blocking=True), o.to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            onset, note, _ = full(x)
        onset, note = onset.float(), note.float()
        ln, lo_ = bce(note, n), bce(onset, o, pos)
        loss = ln + lo_
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        for k, v in (("note", ln), ("onset", lo_)):
            run[k] = 0.98 * run[k] + 0.02 * float(v) if step > 1 else float(v)
        if step % 50 == 0:
            print(f"{step:6d} {time.time() - t0:7.0f}s 음 {run['note']:.4f} 시작점 {run['onset']:.4f}", flush=True)
        late = args.max_minutes and time.time() - t0 > args.max_minutes * 60
        if step % args.save_every == 0 or step == args.steps or late:
            torch.save(full.big.state_dict(), out / f"step{step}.pt")
            export(full, out / f"step{step}.onnx")
        if step >= args.steps or late:
            break


if __name__ == "__main__":
    main()
