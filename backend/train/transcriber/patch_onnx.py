"""학습한 가중치를 원래 nmp.onnx 그래프의 initializer에 그대로 끼운다 (그래프 · 속도는 원래 그대로).
python patch_onnx.py <원래 nmp.onnx> <학습한 .pt> <저장할 .onnx>"""
import sys
from pathlib import Path

import numpy as np
import onnx
import torch
from onnx import numpy_helper

sys.path.insert(0, str(Path(__file__).parent))
import bp_torch  # noqa: E402


def main(src, pt, dst):
    model = bp_torch.load(src)
    model.load_state_dict(torch.load(pt))
    write(model, src, dst)


def write(model, src, dst):
    proto = onnx.load(str(src))
    inits = {i.name: i for i in proto.graph.initializer}
    nodes = {n.name: n for n in proto.graph.node}
    changed = 0
    for name, p in model.named_parameters():
        if "cq_t2010" in name:
            continue
        node_name, kind = name.rsplit(".", 1)
        node = nodes.get(node_name)
        if node is None:
            # onnx2torch가 이름의 '/' 등을 바꿨을 수 있어 찾아본다
            cands = [n for n in proto.graph.node if n.name.replace("/", "_") == node_name or n.name.endswith(node_name)]
            node = cands[0] if cands else None
        if node is None:
            raise SystemExit(f"노드를 못 찾음: {name}")
        idx = 1 if kind == "weight" else 2
        init_name = node.input[idx]
        if init_name not in inits:
            raise SystemExit(f"initializer가 아님: {name} → {init_name}")
        old = numpy_helper.to_array(inits[init_name])
        new = p.detach().numpy().astype(old.dtype)
        if new.shape != old.shape:
            raise SystemExit(f"모양이 다름: {name} {new.shape} vs {old.shape}")
        if not np.array_equal(new, old):
            inits[init_name].CopyFrom(numpy_helper.from_array(new, init_name))
            changed += 1
    onnx.save(proto, str(dst))
    print("바꾼 가중치", changed, flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:4])
