"""
Prepare DNS Challenge audio by splitting clean/noise wav files into fixed-length chunks.

Outputs two folders under `out_root`:
  - clean_10s/
  - noise_10s/

Clean discovery supports both layouts:
  1) dns_root/datasets.clean.*/**/*.wav
  2) dns_root/clean/**/*.wav
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch
import torchaudio
from tqdm import tqdm


@dataclass
class SplitStats:
    files_seen: int = 0
    chunks_written: int = 0
    short_discarded: int = 0


def discover_clean_files(dns_root: Path) -> list[Path]:
    corpus_dirs = sorted(dns_root.glob("datasets.clean.*"))
    if corpus_dirs:
        files: list[Path] = []
        for directory in corpus_dirs:
            files.extend(directory.glob("**/*.wav"))
            files.extend(directory.glob("*.wav"))
        return sorted(set(files))

    clean_dir = dns_root / "clean"
    if clean_dir.is_dir():
        return sorted(clean_dir.glob("**/*.wav"))

    return sorted(dns_root.glob("**/*.wav"))


def load_mono_resample(path: Path, target_sr: int) -> torch.Tensor:
    wav, sr = torchaudio.load(str(path))
    wav = wav.float().mean(dim=0)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav.unsqueeze(0), sr, target_sr).squeeze(0)
    return wav


def write_segments(
    files: list[Path],
    src_root: Path,
    out_root: Path,
    segment_samples: int,
    sample_rate: int,
    min_tail_samples: int,
    pad_tail: bool,
) -> SplitStats:
    stats = SplitStats()
    out_root.mkdir(parents=True, exist_ok=True)

    for file_path in tqdm(files, desc=f"Splitting {src_root.name}", unit="file"):
        stats.files_seen += 1
        audio = load_mono_resample(file_path, sample_rate)
        total = int(audio.shape[0])
        if total == 0:
            continue

        rel = file_path.relative_to(src_root)
        dst_dir = out_root / rel.parent
        dst_dir.mkdir(parents=True, exist_ok=True)

        n_full = total // segment_samples
        for index in range(n_full):
            start = index * segment_samples
            end = start + segment_samples
            chunk = audio[start:end]
            dst = dst_dir / f"{rel.stem}_seg{index:04d}.wav"
            torchaudio.save(str(dst), chunk.unsqueeze(0), sample_rate)
            stats.chunks_written += 1

        tail = total - n_full * segment_samples
        if tail == 0:
            continue
        if tail < min_tail_samples:
            stats.short_discarded += 1
            continue

        if pad_tail:
            padded = torch.zeros(segment_samples, dtype=audio.dtype)
            padded[:tail] = audio[n_full * segment_samples :]
            dst = dst_dir / f"{rel.stem}_seg{n_full:04d}.wav"
            torchaudio.save(str(dst), padded.unsqueeze(0), sample_rate)
            stats.chunks_written += 1

    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split DNS clean/noise wav files into fixed 10s chunks")
    parser.add_argument("--dns_root", type=str, required=True, help="DNS root with datasets.clean.* or clean/")
    parser.add_argument("--noise_root", type=str, required=True, help="Noise directory (recursive wav search)")
    parser.add_argument("--out_root", type=str, required=True, help="Output root directory")
    parser.add_argument("--segment_seconds", type=float, default=10.0, help="Chunk length in seconds")
    parser.add_argument("--sample_rate", type=int, default=16000, help="Target sample rate")
    parser.add_argument("--min_tail_seconds", type=float, default=1.0, help="Discard tails shorter than this")
    parser.add_argument(
        "--no_pad_tail",
        action="store_true",
        help="Do not write the final partial chunk (only full-length chunks)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dns_root = Path(args.dns_root)
    noise_root = Path(args.noise_root)
    out_root = Path(args.out_root)

    if not dns_root.is_dir():
        raise FileNotFoundError(f"dns_root not found: {dns_root}")
    if not noise_root.is_dir():
        raise FileNotFoundError(f"noise_root not found: {noise_root}")

    clean_files = discover_clean_files(dns_root)
    noise_files = sorted(noise_root.glob("**/*.wav"))

    if not clean_files:
        raise RuntimeError(f"No clean wav files found under: {dns_root}")
    if not noise_files:
        raise RuntimeError(f"No noise wav files found under: {noise_root}")

    seg_samples = int(args.segment_seconds * args.sample_rate)
    min_tail_samples = int(args.min_tail_seconds * args.sample_rate)
    if seg_samples <= 0:
        raise ValueError("segment_seconds must be > 0")

    clean_out = out_root / "clean_10s"
    noise_out = out_root / "noise_10s"

    print(f"Clean files found: {len(clean_files)}")
    print(f"Noise files found: {len(noise_files)}")
    print(f"Segment length: {args.segment_seconds:.2f}s ({seg_samples} samples @ {args.sample_rate} Hz)")

    clean_stats = write_segments(
        files=clean_files,
        src_root=dns_root,
        out_root=clean_out,
        segment_samples=seg_samples,
        sample_rate=args.sample_rate,
        min_tail_samples=min_tail_samples,
        pad_tail=not args.no_pad_tail,
    )
    noise_stats = write_segments(
        files=noise_files,
        src_root=noise_root,
        out_root=noise_out,
        segment_samples=seg_samples,
        sample_rate=args.sample_rate,
        min_tail_samples=min_tail_samples,
        pad_tail=not args.no_pad_tail,
    )

    print("\nDone.")
    print(
        f"Clean  -> files: {clean_stats.files_seen}, chunks: {clean_stats.chunks_written}, "
        f"short tails discarded: {clean_stats.short_discarded}"
    )
    print(
        f"Noise  -> files: {noise_stats.files_seen}, chunks: {noise_stats.chunks_written}, "
        f"short tails discarded: {noise_stats.short_discarded}"
    )
    print(f"\nPrepared clean dir: {clean_out}")
    print(f"Prepared noise dir: {noise_out}")


if __name__ == "__main__":
    main()
