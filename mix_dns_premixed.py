"""
Pre-mix DNS clean and noise audio at fixed SNR levels.

Generates paired (noisy, clean) files under output directory:
  output_dir/
    ├── noisy/  (pre-mixed noisy audio)
    └── clean/  (reference clean audio, symlinked or copied)
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torchaudio
import torch
from tqdm import tqdm


def load_mono_resample(path: Path, target_sr: int) -> torch.Tensor:
    """Load wav, convert to mono, resample if needed."""
    wav, sr = torchaudio.load(str(path))
    wav = wav.float().mean(dim=0)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav.unsqueeze(0), sr, target_sr).squeeze(0)
    return wav


def mix_at_snr(clean: torch.Tensor, noise: torch.Tensor, snr_db: float) -> torch.Tensor:
    """Mix clean audio with noise at target SNR."""
    clean_rms = clean.pow(2).mean().sqrt().clamp(min=1e-9)
    noise_rms = noise.pow(2).mean().sqrt().clamp(min=1e-9)
    snr_linear = 10 ** (snr_db / 20.0)
    noise_scaled = noise * (clean_rms / (noise_rms * snr_linear))
    return (clean + noise_scaled).clamp(-1.0, 1.0)


def discover_files(root: Path) -> list[Path]:
    """Discover clean files from datasets.clean.* or clean/ subdirs."""
    corpus_dirs = sorted(root.glob("datasets.clean.*"))
    if corpus_dirs:
        files: list[Path] = []
        for directory in corpus_dirs:
            files.extend(directory.glob("**/*.wav"))
            files.extend(directory.glob("*.wav"))
        return sorted(set(files))

    clean_dir = root / "clean"
    if clean_dir.is_dir():
        return sorted(clean_dir.glob("**/*.wav"))

    return sorted(root.glob("**/*.wav"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-mix DNS clean and noise audio at fixed SNRs")
    parser.add_argument("--clean_root", type=str, required=True, help="Clean audio root (datasets.clean.* or clean/)")
    parser.add_argument("--noise_dir", type=str, required=True, help="Noise audio directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--snr", type=float, default=10.0, help="SNR in dB for mixing")
    parser.add_argument("--sample_rate", type=int, default=16000, help="Target sample rate")
    parser.add_argument("--max_noise_samples", type=int, default=None, 
                       help="Max noise duration (samples); if set, pad/trim all to this length")
    args = parser.parse_args()

    clean_root = Path(args.clean_root)
    noise_dir = Path(args.noise_dir)
    output_dir = Path(args.output_dir)

    if not clean_root.is_dir():
        raise FileNotFoundError(f"clean_root not found: {clean_root}")
    if not noise_dir.is_dir():
        raise FileNotFoundError(f"noise_dir not found: {noise_dir}")

    clean_files = discover_files(clean_root)
    noise_files = sorted(noise_dir.glob("**/*.wav"))

    if not clean_files:
        raise RuntimeError(f"No clean files found under: {clean_root}")
    if not noise_files:
        raise RuntimeError(f"No noise files found under: {noise_dir}")

    # Create output structure
    noisy_out = output_dir / "noisy"
    clean_out = output_dir / "clean"
    noisy_out.mkdir(parents=True, exist_ok=True)
    clean_out.mkdir(parents=True, exist_ok=True)

    print(f"Clean files: {len(clean_files)}")
    print(f"Noise files: {len(noise_files)}")
    print(f"SNR: {args.snr:.1f} dB")
    print(f"Output: {output_dir}")

    noise_pool = [load_mono_resample(p, args.sample_rate) for p in tqdm(noise_files, desc="Loading noise", unit="file")]

    for clean_path in tqdm(clean_files, desc="Mixing", unit="file"):
        clean = load_mono_resample(clean_path, args.sample_rate)
        
        if clean.shape[0] == 0:
            continue

        # Pick a random noise chunk and match length
        noise = random.choice(noise_pool)
        if noise.shape[0] < clean.shape[0]:
            noise = torch.nn.functional.pad(noise, (0, clean.shape[0] - noise.shape[0]))
        elif noise.shape[0] > clean.shape[0]:
            start = random.randint(0, noise.shape[0] - clean.shape[0])
            noise = noise[start : start + clean.shape[0]]

        # Mix
        noisy = mix_at_snr(clean, noise, args.snr)

        # Preserve structure: datasets.clean.* -> noisy/*, or clean/* -> noisy/*
        rel = clean_path.relative_to(clean_root)
        dst_noisy = noisy_out / rel
        dst_clean = clean_out / rel
        dst_noisy.parent.mkdir(parents=True, exist_ok=True)
        dst_clean.parent.mkdir(parents=True, exist_ok=True)

        # Write both
        torchaudio.save(str(dst_noisy), noisy.unsqueeze(0), args.sample_rate)
        torchaudio.save(str(dst_clean), clean.unsqueeze(0), args.sample_rate)

    print(f"\nPre-mixed audio written to {output_dir}")
    print(f"  Noisy: {noisy_out}")
    print(f"  Clean: {clean_out}")
    print(f"\nTo train, use:")
    print(f"  python train.py --data_root {output_dir} --no_mix_on_the_fly")


if __name__ == "__main__":
    main()
