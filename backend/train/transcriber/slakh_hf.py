"""Slakh2100을 Hugging Face 사본(WhiRik/slakh_flac, 원래 redux와 같은 폴더 구조)에서 곡 단위로 받아 바로 변환한다.

  HF_TOKEN=... python slakh_hf.py <출력 폴더> --tracks 600 --procs 6

- Zenodo 원본(tar.gz 104GB)은 곡 파일이 뒤섞여 있어 앞부분만 받으면 온전한 곡이 없고, 받다 끊기면 이어 받을 수 없다.
  이 사본은 곡마다 파일이 따로라, 필요한 곡만 받고 끊겨도 받은 곡은 남는다.
- 데이터셋 페이지에서 접근 동의(자동 승인)가 필요하다. 토큰은 HF_TOKEN 또는 huggingface_hub 기본 위치.
- train · validation에서 고정 시드로 곡을 고른다(test는 남기고 omitted는 뺀다 — slakh_data.py와 같음).
- 곡마다: 받기 → slakh_data의 변환(전체 믹스 + 작은 묶음) → 원본 지우기. 이미 변환한 곡은 건너뛴다.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

import slakh_data as S

REPO = "WhiRik/slakh_flac"


def list_tracks() -> dict[str, list[str]]:
    """곡 폴더 → 받을 파일 목록 (all_src.mid는 안 씀)."""
    from huggingface_hub import HfApi

    tracks: dict[str, list[str]] = {}
    for f in HfApi().list_repo_files(REPO, repo_type="dataset"):
        parts = f.split("/")
        if len(parts) < 3 or parts[0] not in S.SPLITS or parts[-1] == "all_src.mid":
            continue
        tracks.setdefault("/".join(parts[:2]), []).append(f)
    return tracks


def one(job: tuple[str, list[str], str, str, int]) -> str:
    from huggingface_hub import hf_hub_download

    track, files, raw, out, seed = job
    name = track.split("/")[1]
    if (Path(out) / f"{name}.notes.npy").exists() or (Path(out) / f"{name}.skip").exists():
        return f"{name} 있음"
    try:
        for f in files:
            hf_hub_download(REPO, f, repo_type="dataset", local_dir=raw)
        msg = S.prepare_track((str(Path(raw) / track), out, seed))
        if not (Path(out) / f"{name}.notes.npy").exists():
            (Path(out) / f"{name}.skip").write_text(msg, encoding="utf-8")  # 악기 없음 등: 다시 받지 않게
        return msg
    except Exception as e:  # 네트워크 오류는 다음 실행에서 다시 받는다
        return f"{name} 실패: {type(e).__name__} {str(e)[:150]}"
    finally:
        shutil.rmtree(Path(raw) / track, ignore_errors=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--tracks", type=int, default=600)
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--raw", default=None, help="받는 중 임시 폴더 (기본 <out>_raw)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    raw = Path(a.raw or f"{out}_raw")
    raw.mkdir(parents=True, exist_ok=True)
    tracks = list_tracks()
    names = sorted(tracks)
    pick = [names[i] for i in np.random.default_rng(a.seed).permutation(len(names))[: a.tracks]]
    print(f"사본의 곡 {len(names)}개 중 {len(pick)}개", flush=True)
    jobs = [(t, tracks[t], str(raw), str(out), i) for i, t in enumerate(pick)]
    fails = 0
    with Pool(a.procs) as pool:
        for i, msg in enumerate(pool.imap_unordered(one, jobs), 1):
            fails += "실패" in msg
            if i % 25 == 0 or i == len(jobs) or "실패" in msg:
                print(f"{i}/{len(jobs)} {msg}", flush=True)
    done = len(list(out.glob("*.notes.npy")))
    print(f"변환된 파일 {done}개 (곡 + 작은 묶음), 실패 {fails}곡", flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
