"""Basic Pitch nmp.onnx → PyTorch 모듈 (onnx2torch). 반사 패딩이 가운데 축(오디오 길이)에 걸려 있어
onnx2torch가 못 하는 것을 축을 맨 뒤로 옮겨 패딩하고 되돌리게 고쳐 끼운다."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from onnx2torch.node_converters import pad as _pad


def _general_pad(x: torch.Tensor, onnx_pads: list[int], mode: str, value=0.0) -> torch.Tensor:
    n = x.dim()
    begin, end = onnx_pads[:n], onnx_pads[n:]
    if mode == "constant":
        return F.pad(x, _pad._onnx_padding_to_torch(onnx_pads), mode="constant", value=value or 0.0)
    for d in range(n):
        if begin[d] == 0 and end[d] == 0:
            continue
        x = x.transpose(d, n - 1)
        shape = x.shape
        y = F.pad(x.reshape(-1, 1, shape[-1]), (begin[d], end[d]), mode=mode)
        x = y.reshape(*shape[:-1], y.shape[-1]).transpose(d, n - 1)
    return x


def _dyn_forward(self, input_tensor, pads, constant_value=0.0):
    return _general_pad(input_tensor, [int(v) for v in pads.tolist()], self.mode, constant_value)


_pad.OnnxPadDynamic.forward = _dyn_forward


def load(path: str) -> torch.nn.Module:
    import onnx
    from onnx2torch import convert

    return convert(onnx.load(path))  # 경로를 주면 Windows에서 임시 파일을 다시 못 열어 실패한다
