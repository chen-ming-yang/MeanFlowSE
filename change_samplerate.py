#!/usr/bin/env python3
"""
Convert all WAV files in a directory to 16kHz sample rate.
Usage: python change.py <input_dir> [--recursive] [--workers N]
"""

import argparse
import os
import sys
from pathlib import Path
from multiprocessing import Pool

import soxr
import soundfile as sf
from tqdm import tqdm


def convert_to_16k(wav_path: Path, out_path: Path) -> str:
    audio, sr = sf.read(str(wav_path), always_2d=True)  # (samples, channels)
    if sr == 16000 and out_path == wav_path:
        return f"[skip] {wav_path}"
    if sr != 16000:
        audio = soxr.resample(audio, sr, 16000, quality="HQ")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), audio, 16000)
    return f"[converted] {wav_path}  {sr}Hz -> 16000Hz"


def _worker(args):
    wav_path, out_path = args
    try:
        return convert_to_16k(wav_path, out_path)
    except Exception as e:
        return f"[error] {wav_path}: {e}"


def main():
    parser = argparse.ArgumentParser(description="Convert WAV files to 16kHz.")
    parser.add_argument("input_dir", help="Directory containing WAV files.")
    parser.add_argument("--recursive", "-r", action="store_true",
                        help="Recursively process subdirectories.")
    parser.add_argument("--output-dir", "-o", default=None,
                        help="Directory to write converted files. Defaults to in-place overwrite.")
    parser.add_argument("--no-overwrite", action="store_true",
                        help="Save with '_16k' suffix instead of overwriting.")
    parser.add_argument("--workers", "-j", type=int, default=os.cpu_count(),
                        help="Number of parallel worker processes (default: all CPU cores).")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        print(f"Error: '{input_dir}' is not a valid directory.", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else None
    pattern = "**/*.wav" if args.recursive else "*.wav"
    wav_files = sorted(input_dir.glob(pattern))

    if not wav_files:
        print("No WAV files found.")
        return

    print(f"Found {len(wav_files)} WAV file(s). Using {args.workers} workers.\n")

    tasks = []
    for wav_path in wav_files:
        if output_dir is not None:
            out_path = output_dir / wav_path.relative_to(input_dir)
        elif args.no_overwrite:
            out_path = wav_path.with_stem(wav_path.stem + "_16k")
        else:
            out_path = wav_path
        tasks.append((wav_path, out_path))

    errors = []
    with Pool(processes=args.workers) as pool:
        for msg in tqdm(pool.imap_unordered(_worker, tasks, chunksize=16),
                        total=len(tasks), unit="file"):
            if msg.startswith("[error]"):
                errors.append(msg)

    if errors:
        print(f"\n{len(errors)} error(s):")
        for e in errors:
            print(e, file=sys.stderr)

    print("\nDone.")


if __name__ == "__main__":
    main()
