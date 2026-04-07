#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create an aligned paired-audio directory where each dysarthria/normal pair "
            "has equal length by slowing down the shorter one to match the longer duration."
        )
    )
    parser.add_argument(
        "--pairs-json",
        type=Path,
        default=Path("/home/cmy/MeanFlowSE/pairs.json"),
        help="Path to pairs.json containing filtered pairs.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/home/cmy/MeanFlowSE/tmp_audio_equal_len"),
        help="Output directory for equal-length paired audio.",
    )
    parser.add_argument(
        "--pair-key",
        type=str,
        default="used",
        choices=["used", "excluded"],
        help="Which list in pairs.json to process.",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=0,
        help="Optional limit for processing first N pairs (0 = all).",
    )
    parser.add_argument(
        "--target-sr",
        type=int,
        default=16000,
        help="Target sample rate for output files.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, os.cpu_count() or 4),
        help="Number of parallel worker processes.",
    )
    return parser.parse_args()


def to_2d(wav: np.ndarray) -> np.ndarray:
    if wav.ndim == 1:
        return wav[:, None]
    return wav


def resample_to_target_sr(wav: np.ndarray, src_sr: int, target_sr: int) -> np.ndarray:
    if src_sr == target_sr:
        return wav

    # torchaudio expects [channels, time], while numpy read shape is [time, channels].
    wav_tensor = torch.from_numpy(wav.T)
    resampled = torchaudio.functional.resample(wav_tensor, src_sr, target_sr)
    return resampled.T.numpy()


def speed_stretch(wav: np.ndarray, current_len: int, target_len: int, target_sr: int) -> np.ndarray:
    """Slow down audio so it becomes target_len samples.

    Uses linear interpolation along the time axis — very fast and sufficient
    for small duration adjustments (a few seconds).
    """
    if current_len == target_len:
        return wav

    src_indices = np.linspace(0, current_len - 1, target_len)
    orig_indices = np.arange(current_len)

    channels = wav.shape[1]
    result = np.empty((target_len, channels), dtype=wav.dtype)
    for ch in range(channels):
        result[:, ch] = np.interp(src_indices, orig_indices, wav[:, ch])

    return result


def make_output_path(output_root: Path, src_rel_path: str, domain: str) -> Path:
    src = Path(src_rel_path)
    expected_prefix = Path("tmp_audio") / domain

    if src.parts[: len(expected_prefix.parts)] == expected_prefix.parts:
        sub_path = Path(*src.parts[len(expected_prefix.parts) :])
    else:
        sub_path = src.name

    return output_root / domain / sub_path


def process_one_pair(task: tuple) -> dict | None:
    """Worker function for a single pair. Returns metadata dict or None on skip."""
    pair, project_root, output_root, target_sr = task
    audio_id = pair["id"]

    dys_path = Path(project_root) / pair["dysarthria"]
    normal_path = Path(project_root) / pair["normal"]

    if not dys_path.exists() or not normal_path.exists():
        return None

    dys_wav, dys_sr = sf.read(dys_path, dtype="float32")
    normal_wav, normal_sr = sf.read(normal_path, dtype="float32")

    dys_wav = to_2d(dys_wav)
    normal_wav = to_2d(normal_wav)

    dys_wav = resample_to_target_sr(dys_wav, dys_sr, target_sr)
    normal_wav = resample_to_target_sr(normal_wav, normal_sr, target_sr)

    if dys_wav.shape[1] != normal_wav.shape[1]:
        return None

    target_len = max(dys_wav.shape[0], normal_wav.shape[0])

    if dys_wav.shape[0] < target_len:
        dys_wav = speed_stretch(dys_wav, dys_wav.shape[0], target_len, target_sr)

    if normal_wav.shape[0] < target_len:
        normal_wav = speed_stretch(normal_wav, normal_wav.shape[0], target_len, target_sr)

    out_dys = make_output_path(Path(output_root), pair["dysarthria"], domain="dysarthria")
    out_normal = make_output_path(Path(output_root), pair["normal"], domain="normal")

    out_dys.parent.mkdir(parents=True, exist_ok=True)
    out_normal.parent.mkdir(parents=True, exist_ok=True)

    sf.write(out_dys, dys_wav, target_sr)
    sf.write(out_normal, normal_wav, target_sr)

    return {
        "id": audio_id,
        "dysarthria": str(out_dys.relative_to(Path(project_root))),
        "normal": str(out_normal.relative_to(Path(project_root))),
        "dys_input_sample_rate": dys_sr,
        "normal_input_sample_rate": normal_sr,
        "sample_rate": target_sr,
        "num_samples": target_len,
        "duration_sec": round(target_len / target_sr, 6),
    }


def main() -> None:
    args = parse_args()
    project_root = str(args.pairs_json.parent)

    if not args.pairs_json.exists():
        raise FileNotFoundError(f"pairs.json not found: {args.pairs_json}")

    with args.pairs_json.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if args.pair_key not in data:
        raise KeyError(f"Missing key '{args.pair_key}' in {args.pairs_json}")

    pairs: list[dict] = data[args.pair_key]
    if args.max_pairs > 0:
        pairs = pairs[: args.max_pairs]

    args.output_root.mkdir(parents=True, exist_ok=True)

    tasks = [
        (pair, project_root, str(args.output_root), args.target_sr)
        for pair in pairs
    ]

    aligned_entries: list[dict] = []
    skipped = 0
    done = 0

    print(f"Processing {len(tasks)} pairs with {args.workers} workers...")

    with Pool(processes=args.workers) as pool:
        for result in pool.imap_unordered(process_one_pair, tasks, chunksize=32):
            done += 1
            if result is None:
                skipped += 1
            else:
                aligned_entries.append(result)
            if done % 500 == 0:
                print(f"Processed {done}/{len(tasks)} pairs")

    # Sort by id for deterministic output
    aligned_entries.sort(key=lambda e: e["id"])

    out_meta = args.output_root / f"aligned_pairs_{args.pair_key}.json"
    with out_meta.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "source_pairs_json": str(args.pairs_json),
                "pair_key": args.pair_key,
                "processed_pairs": len(aligned_entries),
                "skipped_pairs": skipped,
                "output_root": str(args.output_root),
                "pairs": aligned_entries,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("\n--- Done ---")
    print(f"Input pairs requested: {len(pairs)}")
    print(f"Aligned pairs written: {len(aligned_entries)}")
    print(f"Skipped pairs:         {skipped}")
    print(f"Workers used:          {args.workers}")
    print(f"Output root:           {args.output_root}")
    print(f"Metadata JSON:         {out_meta}")


if __name__ == "__main__":
    main()
