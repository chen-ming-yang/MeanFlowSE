"""
Pretrain WaveVAE (Flow-VAE from KALL-E) on clean speech data.

The pretrained encoder/decoder are then frozen inside MeanFlowSE.

Usage:
    python train_wave_vae.py --data_dir /path/to/clean_wavs --save_dir wave_vae_ckpts

Training losses:
    1. L1 mel-spectrogram reconstruction
    2. KL divergence (weighted)
    3. Multi-scale STFT loss
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm

from wave_vae import WaveVAE


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

class MelSpectrogramLoss(nn.Module):
    """L1 loss on mel spectrogram."""

    def __init__(self, sample_rate=16000, n_fft=1024, hop_length=256,
                 n_mels=80, f_min=0.0, f_max=8000.0):
        super().__init__()
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
            power=1.0,
        )

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # y_hat, y: (B, T)
        mel_hat = self.mel_spec(y_hat)
        mel_target = self.mel_spec(y)
        return F.l1_loss(mel_hat, mel_target)


class MultiScaleSTFTLoss(nn.Module):
    """Multi-scale spectral convergence + L1 magnitude loss."""

    def __init__(self, fft_sizes=(512, 1024, 2048), hop_sizes=(128, 256, 512),
                 win_sizes=(512, 1024, 2048)):
        super().__init__()
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_sizes = win_sizes

    def _stft_loss(self, y_hat, y, fft_size, hop_size, win_size):
        window = torch.hann_window(win_size, device=y.device)
        Y_hat = torch.stft(y_hat, fft_size, hop_size, win_size, window,
                           return_complex=True)
        Y = torch.stft(y, fft_size, hop_size, win_size, window,
                        return_complex=True)
        mag_hat = Y_hat.abs()
        mag = Y.abs()

        sc_loss = torch.norm(mag - mag_hat, p="fro") / (torch.norm(mag, p="fro") + 1e-8)
        mag_loss = F.l1_loss(torch.log(mag_hat + 1e-7), torch.log(mag + 1e-7))
        return sc_loss + mag_loss

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        loss = 0.0
        for fs, hs, ws in zip(self.fft_sizes, self.hop_sizes, self.win_sizes):
            loss = loss + self._stft_loss(y_hat, y, fs, hs, ws)
        return loss / len(self.fft_sizes)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CleanSpeechDataset(Dataset):
    """Load clean wav files, crop to fixed length."""

    def __init__(self, data_dir: str, sample_rate: int = 16000,
                 segment_len: float = 2.0):
        self.files = sorted(Path(data_dir).rglob("*.wav"))
        if not self.files:
            raise RuntimeError(f"No .wav files found in {data_dir}")
        self.sample_rate = sample_rate
        self.segment_samples = int(segment_len * sample_rate)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        wav, sr = torchaudio.load(self.files[idx])
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
        wav = wav.squeeze(0)  # (T,)

        # Random crop or pad
        if wav.shape[0] >= self.segment_samples:
            start = torch.randint(0, wav.shape[0] - self.segment_samples + 1, (1,)).item()
            wav = wav[start:start + self.segment_samples]
        else:
            wav = F.pad(wav, (0, self.segment_samples - wav.shape[0]))

        return wav


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Logger
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("WaveVAE")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(save_dir / "train_wave_vae.log", mode="a")
        fh.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
        logger.addHandler(fh)

    # Model
    model = WaveVAE(
        latent_dim=args.latent_dim,
        downsample_rates=tuple(args.downsample_rates),
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"WaveVAE parameters: {n_params:.1f} M")
    print(f"Hop length: {model.hop_length} → {args.sample_rate / model.hop_length:.1f} Hz")

    # Dataset
    dataset = CleanSpeechDataset(
        data_dir=args.data_dir,
        sample_rate=args.sample_rate,
        segment_len=args.segment_len,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    print(f"Dataset: {len(dataset)} files, {len(loader)} batches/epoch")

    # Losses
    mel_loss_fn = MelSpectrogramLoss(sample_rate=args.sample_rate).to(device)
    msstft_loss_fn = MultiScaleSTFTLoss().to(device)

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ExponentialLR(optimizer, gamma=args.lr_decay)

    # Resume
    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0) + 1
        print(f"Resumed from epoch {start_epoch}")

    # Training
    for epoch in tqdm(range(start_epoch, args.epochs), desc="Epochs"):
        model.train()
        epoch_loss = 0.0

        for batch_idx, wav in enumerate(loader):
            wav = wav.to(device)  # (B, T)

            recon, mu, logvar = model(wav)

            # Crop to same length
            min_len = min(recon.shape[-1], wav.shape[-1])
            recon = recon[..., :min_len]
            wav_crop = wav[..., :min_len]

            # Losses
            l_mel = mel_loss_fn(recon, wav_crop)
            l_msstft = msstft_loss_fn(recon, wav_crop)
            l_kl = WaveVAE.kl_loss(mu, logvar)

            loss = l_mel + l_msstft + args.kl_weight * l_kl

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            epoch_loss += loss.item()

            if (batch_idx + 1) % args.log_every == 0:
                msg = (
                    f"Epoch {epoch} [{batch_idx+1}/{len(loader)}] "
                    f"loss={loss.item():.4f} mel={l_mel.item():.4f} "
                    f"msstft={l_msstft.item():.4f} kl={l_kl.item():.4f}"
                )
                logger.info(msg)
                if args.verbose:
                    print(msg)

        scheduler.step()
        avg_loss = epoch_loss / len(loader)
        print(f"Epoch {epoch}: avg_loss={avg_loss:.4f}, lr={scheduler.get_last_lr()[0]:.2e}")
        logger.info(f"Epoch {epoch}: avg_loss={avg_loss:.4f}")

        # Save checkpoint
        if (epoch + 1) % args.save_every == 0 or epoch == args.epochs - 1:
            ckpt_path = save_dir / f"wave_vae_epoch{epoch:03d}.pt"
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": {
                    "latent_dim": args.latent_dim,
                    "downsample_rates": args.downsample_rates,
                    "hop_length": model.hop_length,
                },
            }, ckpt_path)
            print(f"Saved → {ckpt_path}")


def main():
    p = argparse.ArgumentParser(description="Pretrain WaveVAE for MeanFlowSE")
    # Data
    p.add_argument("--data_dir", type=str, required=True,
                   help="Directory of clean speech .wav files")
    p.add_argument("--sample_rate", type=int, default=16000)
    p.add_argument("--segment_len", type=float, default=2.0,
                   help="Training segment length in seconds")
    # Model
    p.add_argument("--latent_dim", type=int, default=256,
                   help="Latent dimension (paper: 256)")
    p.add_argument("--downsample_rates", type=int, nargs="+", default=[8, 5, 4, 4],
                   help="Downsample factors (product = hop_length)")
    # Training
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lr_decay", type=float, default=0.999,
                   help="Exponential LR decay per epoch")
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--kl_weight", type=float, default=0.1,
                   help="Weight for KL divergence loss")
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--save_every", type=int, default=10)
    p.add_argument("--save_dir", type=str, default="wave_vae_ckpts")
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--verbose", action="store_true")

    train(p.parse_args())


if __name__ == "__main__":
    main()
