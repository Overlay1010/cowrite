"""이 PC에서 돌리는 감시: Pod의 체크포인트 · 로그를 몇 분마다 받아 오고, 잔액이 바닥에 가까우면 다 받은 뒤 Pod를 지운다.

  python runpod/watch.py <pod-id> --ssh "root@IP -p PORT" --floor 2

- 결과는 이 PC(runs_gpu/, git 무시)에 쌓이므로 Pod가 어떻게 되든 학습한 만큼은 남는다.
- 받는 것: /workspace/runs 아래 *.onnx · *.pt(새 것만) · *.log(매번 덮어씀). Slakh 변환본은 다시 만들 수 있어서 안 받는다.
- 잔액이 --floor 달러 아래면: 마지막으로 다 받고 → Pod 삭제(Terminate). 잔액이 0이 되면 RunPod이 볼륨을 지울 수 있어서, 그 전에 끝낸다.
- 표준 라이브러리만 쓴다. runpodctl(키는 runpodctl doctor로 저장) · ssh · scp가 PATH에 있어야 한다.
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REMOTE = "/workspace/runs"


def log(msg: str) -> None:
    print(f"{datetime.now():%m-%d %H:%M:%S} {msg}", flush=True)


def run(cmd: list[str], timeout: int = 600) -> str:
    return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout, check=True).stdout


def balance(ctl: str) -> float | None:
    """runpodctl me 출력에서 잔액을 찾는다 (필드 이름이 버전마다 달라 'balance'가 들어간 숫자 키를 쓴다)."""
    try:
        data = json.loads(run([ctl, "me", "-o", "json"], timeout=60))
    except Exception as e:  # 네트워크가 잠깐 끊겨도 감시는 계속
        log(f"잔액 조회 실패: {type(e).__name__} {e}")
        return None
    stack = [data]
    while stack:
        d = stack.pop()
        if isinstance(d, dict):
            for k, v in d.items():
                if "balance" in k.lower() and isinstance(v, (int, float)):
                    return float(v)
                stack.append(v)
        elif isinstance(d, list):
            stack.extend(d)
    log("잔액 필드를 못 찾음: " + json.dumps(data, ensure_ascii=False)[:300])
    return None


def ssh_parts(ssh: str, key: str | None) -> tuple[list[str], list[str], str]:
    """'root@IP -p PORT' → ssh 옵션, scp 옵션(-P), 대상."""
    parts = shlex.split(ssh)
    target = next(p for p in parts if "@" in p)
    port = parts[parts.index("-p") + 1] if "-p" in parts else "22"
    common = ["-o", "StrictHostKeyChecking=accept-new", "-o", "ConnectTimeout=20", "-o", "BatchMode=yes"]
    if key:
        common += ["-i", key]
    return common + ["-p", port], common + ["-P", port], target


def pull(ssh: str, key: str | None, out: Path) -> int:
    """새 체크포인트와 로그를 받는다. 받은 파일 수를 돌려준다."""
    sopt, copt, target = ssh_parts(ssh, key)
    listing = run(["ssh", *sopt, target, f"cd {REMOTE} 2>/dev/null || exit 0; find . -type f \\( -name '*.onnx' -o -name '*.pt' -o -name '*.log' \\) -printf '%s %p\\n'"], timeout=120)
    got = 0
    for line in listing.splitlines():
        size, rel = line.split(" ", 1)
        rel = rel.removeprefix("./")
        dst = out / rel
        fresh = rel.endswith(".log") or not dst.exists() or dst.stat().st_size != int(size)
        if not fresh:
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_name(dst.name + ".part")  # 받다 끊겨도 반쪽 파일을 체크포인트로 착각하지 않게
        run(["scp", "-q", *copt, f"{target}:{REMOTE}/{rel}", str(tmp)], timeout=1800)
        tmp.replace(dst)
        got += 0 if rel.endswith(".log") else 1
    return got


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pod")
    ap.add_argument("--ssh", required=True, help='예: "root@1.2.3.4 -p 12345"')
    ap.add_argument("--key", default=str(Path.home() / ".runpod" / "ssh" / "runpodctl-ssh-key"))
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "runs_gpu"))
    ap.add_argument("--every", type=float, default=5, help="분")
    ap.add_argument("--floor", type=float, default=2.0, help="잔액이 이 달러 아래면 다 받고 Pod 삭제")
    ap.add_argument("--ctl", default="runpodctl")
    ap.add_argument("--no_delete", action="store_true", help="잔액이 바닥이어도 지우지 않고 받기만")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    key = a.key if Path(a.key).exists() else None
    log(f"감시 시작: Pod {a.pod} → {out}, {a.every}분마다, 잔액 ${a.floor} 아래면 {'받기만' if a.no_delete else '다 받고 삭제'}")
    fails = 0
    while True:
        try:
            n = pull(a.ssh, key, out)
            fails = 0
            if n:
                log(f"새 파일 {n}개 받음")
        except Exception as e:
            fails += 1
            log(f"받기 실패 {fails}번째: {type(e).__name__} {str(e)[:200]}")
        b = balance(a.ctl)
        if b is not None:
            log(f"잔액 ${b:.2f}")
            if b < a.floor:
                log("잔액이 바닥에 가까움 → 마지막으로 받기")
                try:
                    pull(a.ssh, key, out)
                except Exception as e:
                    log(f"마지막 받기 실패: {e} — 그래도 지금까지 받은 건 {out}에 있음")
                if a.no_delete:
                    log("--no_delete라 Pod는 그대로 둠. 직접 지워 주세요")
                else:
                    log(run([a.ctl, "pod", "delete", a.pod], timeout=120).strip()[:300])
                    log("Pod 삭제함")
                return
        time.sleep(a.every * 60)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
