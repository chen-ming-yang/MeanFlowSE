"""
Training script for MeanFlowSE on the DNS Challenge dataset.

Expected dataset layout (DNS Challenge):
    <data_root>/
        clean/          *.wav   clean speech utterances
        noise/          *.wav   noise segments
        noisy/          *.wav   pre-mixed noisy files (optional; used if mix_on_the_fly=False)

If mix_on_the_fly=True (default) the DataLoader mixes clean + noise at runtime
with random SNR, which gives effectively infinite augmentation variety.
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

from tqdm import tqdm

import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from mean_flow import MeanFlowSE, SSLEncoder, VAEEncoder, VAEDecoder
from dit import DiTBackbone
from config import Config, default_config


# ---------------------------------------------------------------------------
# Audio augmentation helpers
# ---------------------------------------------------------------------------

class AudioAugment:
    """Lightweight waveform-level augmentations applied to the noisy mixture."""

    def __init__(self, sample_rate: int = 16000, rir_dir: str | None = None):
        self.sample_rate = sample_rate
        self.rir_files: list[Path] = []
        if rir_dir is not None:
            self.rir_files = sorted(Path(rir_dir).glob("**/*.wav"))
            if self.rir_files:
                print(f"AudioAugment: loaded {len(self.rir_files)} RIR files from {rir_dir}")

    def _random_gain(self, wav: torch.Tensor, low: float = 0.7, high: float = 1.0) -> torch.Tensor:
        return wav * random.uniform(low, high)

    def _random_speed(self, wav: torch.Tensor) -> torch.Tensor:
        """Sox-style speed perturbation ±5 % via resampling."""
        factor = random.uniform(0.95, 1.05)
        orig_len = wav.shape[-1]
        resampled = torchaudio.functional.resample(
            wav,
            orig_freq=self.sample_rate,
            new_freq=int(self.sample_rate * factor),
        )
        # Restore length by trimming or zero-padding
        if resampled.shape[-1] > orig_len:
            resampled = resampled[..., :orig_len]
        else:
            resampled = F.pad(resampled, (0, orig_len - resampled.shape[-1]))
        return resampled

    def _random_reverb(self, wav: torch.Tensor) -> torch.Tensor:
        """Convolve wav with a room impulse response.

        If RIR files are available, randomly chooses between a real RIR file
        (50 % chance) and a simulated exponential IR (50 % chance).
        Falls back to simulated when no RIR files are provided.
        """
        use_real = self.rir_files and random.random() < 0.5
        if use_real:
            rir_path = random.choice(self.rir_files)
            ir, sr = torchaudio.load(str(rir_path))
            ir = ir.mean(dim=0)  # mono
            if sr != self.sample_rate:
                ir = torchaudio.functional.resample(ir, sr, self.sample_rate)
            ir = ir.to(wav.device)
        else:
            ir_len = random.randint(800, 4000)  # ~50–250 ms at 16 kHz
            decay = random.uniform(5.0, 15.0)
            t = torch.arange(ir_len, dtype=torch.float32, device=wav.device)
            ir = torch.exp(-decay * t / ir_len)

        ir = ir / ir.norm().clamp(min=1e-9)
        ir_len = ir.shape[-1]
        # Conv1d expects (batch, channel, time); flip IR to perform correlation
        wav_c = wav.unsqueeze(0).unsqueeze(0)
        ir_c = ir.flip(0).unsqueeze(0).unsqueeze(0)
        out = F.conv1d(wav_c, ir_c, padding=ir_len - 1)
        out = out[..., :wav.shape[-1]]
        return out.squeeze(0).squeeze(0)

    def __call__(self, noisy: torch.Tensor) -> torch.Tensor:
        if random.random() < 0.5:
            noisy = self._random_gain(noisy)
        if random.random() < 0.15:
            noisy = self._random_speed(noisy)
        if random.random() < 0.15:
            noisy = self._random_reverb(noisy)
        return noisy


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class DNSDataset(Dataset):
    """
    DNS Challenge dataset loader.

    Clean files are discovered recursively under `datasets.clean.*` subdirectories
    of `dns_root`:

        dns_root/datasets.clean.<name>/**/*.wav         (multi-corpus layout)

    Noise files are loaded separately from `noise_dir/**/*.wav`.

    mix_on_the_fly=True (default):
        Randomly picks a clean + noise file each step and mixes them at a
        random SNR in [snr_low, snr_high] dB – effectively infinite augmentation.
    mix_on_the_fly=False:
        Reads pre-mixed files from `dns_root/noisy/` paired with the clean files
        discovered above (files must correspond 1-to-1 after sorting).
    """

    def __init__(
        self,
        dns_root: str,
        noise_dir: str,
        sample_rate: int = 16000,
        clip_len: float = 4.0,          # seconds
        mix_on_the_fly: bool = True,
        snr_low: float = -5.0,
        snr_high: float = 20.0,
        augment: bool = True,
        rir_dir: str | None = None,
    ):
        self.dns_root = Path(dns_root)
        self.noise_dir = Path(noise_dir)
        self.sample_rate = sample_rate
        self.clip_samples = int(clip_len * sample_rate)
        self.mix_on_the_fly = mix_on_the_fly
        self.snr_low = snr_low
        self.snr_high = snr_high
        self.augment = augment
        self.aug = AudioAugment(sample_rate, rir_dir=rir_dir)

        # Discover clean files from datasets.clean.*/ subdirectories only.
        if not self.dns_root.is_dir():
            raise FileNotFoundError(
                f"dns_root does not exist: {self.dns_root.resolve()}\n"
                "  Did you forget a leading '/' in the path?"
            )
        clean_subdirs = sorted(self.dns_root.glob("datasets.clean.*"))
        self.clean_files = sorted(self.dns_root.glob("datasets.clean.*/**/*.wav"))
        # Also include wav files directly inside datasets.clean.*/ (no subdirs)
        for d in clean_subdirs:
            self.clean_files = sorted(
                set(self.clean_files) | set(d.glob("*.wav"))
            )
        self.clean_files = sorted(self.clean_files)
        assert len(self.clean_files) > 0, (
            f"No .wav files found under {self.dns_root.resolve()}/datasets.clean.*/\n"
            f"  Found subdirs: {[d.name for d in clean_subdirs] or 'none'}"
        )
        print(f"DNSDataset: loaded {len(self.clean_files)} clean files from {self.dns_root}/datasets.clean.*/")
        self.noise_files = sorted(self.noise_dir.glob("**/*.wav"))
        assert len(self.noise_files) > 0, (
            f"No .wav files found recursively under {self.noise_dir}"
        )
        print(f"DNSDataset: loaded {len(self.noise_files)} noise files from {self.noise_dir}/")

        if not mix_on_the_fly:
            self.noisy_files = sorted((self.dns_root / "noisy").glob("**/*.wav"))
            assert len(self.noisy_files) == len(self.clean_files), (
                f"Noisy ({len(self.noisy_files)}) and clean ({len(self.clean_files)}) "
                "counts must match when mix_on_the_fly=False"
            )

    def __len__(self) -> int:
        return len(self.clean_files)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_mono(self, path: Path) -> torch.Tensor:
        """Load a wav file, convert to mono, resample if needed → (samples,)."""
        wav, sr = torchaudio.load(str(path))
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
        return wav.squeeze(0)

    def _random_crop(self, wav: torch.Tensor) -> torch.Tensor:
        if wav.shape[-1] >= self.clip_samples:
            start = random.randint(0, wav.shape[-1] - self.clip_samples)
            return wav[start: start + self.clip_samples]
        return F.pad(wav, (0, self.clip_samples - wav.shape[-1]))

    def _mix_at_snr(self, clean: torch.Tensor, noise: torch.Tensor, snr_db: float) -> torch.Tensor:
        clean_rms = clean.pow(2).mean().sqrt().clamp(min=1e-9)
        noise_rms = noise.pow(2).mean().sqrt().clamp(min=1e-9)
        snr_linear = 10 ** (snr_db / 20.0)
        noise_scaled = noise * (clean_rms / (noise_rms * snr_linear))
        return (clean + noise_scaled).clamp(-1.0, 1.0)

    # ------------------------------------------------------------------

    def __getitem__(self, idx: int):
        try:
            return self._getitem_inner(idx)
        except Exception:
            # Skip corrupted/unreadable file; return a neighbouring sample
            return self._getitem_inner((idx + 1) % len(self.clean_files))

    def _getitem_inner(self, idx: int):
        clean = self._load_mono(self.clean_files[idx])
        clean = self._random_crop(clean)

        if self.mix_on_the_fly:
            noise_path = random.choice(self.noise_files)
            noise = self._load_mono(noise_path)
            # Loop-pad noise to match clean length
            if noise.shape[-1] < self.clip_samples:
                repeats = math.ceil(self.clip_samples / noise.shape[-1])
                noise = noise.repeat(repeats)[: self.clip_samples]
            else:
                noise = self._random_crop(noise)
            snr = random.uniform(self.snr_low, self.snr_high)
            noisy = self._mix_at_snr(clean, noise, snr)
        else:
            noisy = self._load_mono(self.noisy_files[idx])
            noisy = self._random_crop(noisy)

        if self.augment:
            noisy = self.aug(noisy)
            noisy = noisy.clamp(-1.0, 1.0)

        return noisy, clean


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_model(cfg: argparse.Namespace) -> MeanFlowSE:
    ssl_encoder = SSLEncoder(model_name=cfg.ssl_model, num_layers=cfg.ssl_layers)
    vae_encoder = VAEEncoder(latent_dim=cfg.latent_dim)
    vae_decoder = VAEDecoder(latent_dim=cfg.latent_dim)
    backbone = DiTBackbone(
        latent_dim=cfg.latent_dim,
        ssl_dim=cfg.ssl_dim,
        hidden_dim=cfg.hidden_dim,
        depth=cfg.depth,
        heads=cfg.heads,
        dim_head=cfg.dim_head,
        ff_mult=cfg.ff_mult,
        dropout=cfg.dropout,
        attn_backend=cfg.attn_backend,
    )
    return MeanFlowSE(
        ssl_encoder=ssl_encoder,
        vae_encoder=vae_encoder,
        vae_decoder=vae_decoder,
        dit_backbone=backbone,
        flow_ratio=cfg.flow_ratio,
        time_mu=cfg.time_mu,
        time_sigma=cfg.time_sigma,
        adaptive_gamma=cfg.adaptive_gamma,
        adaptive_c=cfg.adaptive_c,
    )


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(cfg: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---- CUDA performance knobs ----
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True      # TF32 matmul (~2x for non-fp16 ops)
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True              # auto-tune conv kernels
        torch.set_float32_matmul_precision("high")
        print("Enabled: TF32 matmul, cuDNN benchmark")

    # Dataset & loader
    dataset = DNSDataset(
        dns_root=cfg.data_root,
        noise_dir=cfg.noise_dir,
        sample_rate=cfg.sample_rate,
        clip_len=cfg.clip_len,
        mix_on_the_fly=cfg.mix_on_the_fly,
        snr_low=cfg.snr_low,
        snr_high=cfg.snr_high,
        augment=not cfg.no_augment,
        rir_dir=cfg.rir_dir,
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=cfg.num_workers > 0,
        prefetch_factor=8 if cfg.num_workers > 0 else None,
    )

    # Model
    model = build_model(cfg).to(device)

    # Compile models for faster execution (PyTorch 2.x)
    if cfg.compile:
        print("Compiling SSL encoder + DiT backbone with torch.compile ...")
        model.ssl_encoder.ssl_model = torch.compile(model.ssl_encoder.ssl_model, mode="reduce-overhead")
        model.dit_backbone = torch.compile(model.dit_backbone)

    # Only train the backbone + SSL layer weights; VAE is frozen inside MeanFlowSE
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable parameters: {sum(p.numel() for p in trainable) / 1e6:.1f} M")

    optimizer = AdamW(trainable, lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs * len(loader), eta_min=cfg.lr_min)

    # Optional: resume from checkpoint
    start_epoch = 0
    if cfg.resume:
        ckpt = torch.load(cfg.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed from epoch {start_epoch}")

    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Mixed precision scaler
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.fp16)

    global_step = start_epoch * len(loader)

    epoch_bar = tqdm(range(start_epoch, cfg.epochs), desc="Epochs", unit="epoch")
    for epoch in epoch_bar:
        model.train()
        epoch_loss = 0.0

        step_bar = tqdm(loader, desc=f"Epoch {epoch:03d}", unit="batch", leave=False)
        for step, (noisy, clean) in enumerate(step_bar):
            noisy = noisy.to(device, non_blocking=True)
            clean = clean.to(device, non_blocking=True)

            optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=cfg.fp16):
                loss, stats = model.forward_train(noisy, clean)

            scaler.scale(loss).backward()

            if cfg.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, cfg.grad_clip)

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_loss += loss.item()
            global_step += 1

            lr_now = scheduler.get_last_lr()[0]
            step_bar.set_postfix(
                loss=f"{stats['loss']:.4f}",
                delta_sq=f"{stats['delta_sq']:.4f}",
                w=f"{stats['mean_weight']:.4f}",
                lr=f"{lr_now:.2e}",
            )

        avg_loss = epoch_loss / len(loader)
        epoch_bar.set_postfix(avg_loss=f"{avg_loss:.4f}")
        tqdm.write(f"=== Epoch {epoch:03d} avg loss: {avg_loss:.4f} ===")

        # Save checkpoint every N epochs and at the last epoch
        if (epoch + 1) % cfg.save_every == 0 or epoch == cfg.epochs - 1:
            ckpt_path = save_dir / f"ckpt_epoch{epoch:03d}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "cfg": vars(cfg),
                },
                ckpt_path,
            )
            tqdm.write(f"Saved checkpoint → {ckpt_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def get_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train MeanFlowSE on DNS Challenge")
    cfg = default_config  # seed defaults from config.py

    # Data
    p.add_argument("--data_root", type=str, default=cfg.data.data_root,
                   help="DNS root dir: contains clean/ or datasets.clean.*/ subdirs")
    p.add_argument("--noise_dir", type=str, default=cfg.data.noise_dir,
                   help="Directory of noise .wav files (searched recursively)")
    p.add_argument("--sample_rate", type=int, default=cfg.data.sample_rate)
    p.add_argument("--clip_len", type=float, default=cfg.data.clip_len, help="Training clip length (seconds)")
    p.add_argument("--mix_on_the_fly", action="store_true", default=cfg.data.mix_on_the_fly,
                   help="Mix clean+noise at runtime")
    p.add_argument("--snr_low", type=float, default=cfg.data.snr_low)
    p.add_argument("--snr_high", type=float, default=cfg.data.snr_high)
    p.add_argument("--no_augment", action="store_true", default=not cfg.data.augment,
                   help="Disable waveform augmentation")
    p.add_argument("--rir_dir", type=str, default=cfg.data.rir_dir,
                   help="Directory of RIR .wav files; if set, real RIRs are randomly mixed with synthetic IRs")

    # Model
    p.add_argument("--ssl_model", type=str, default=cfg.model.ssl_model)
    p.add_argument("--ssl_layers", type=int, default=cfg.model.ssl_layers)
    p.add_argument("--ssl_dim", type=int, default=cfg.model.ssl_dim, help="WavLM hidden size (768 for base-plus, 1024 for large)")
    p.add_argument("--latent_dim", type=int, default=cfg.model.latent_dim)
    p.add_argument("--hidden_dim", type=int, default=cfg.model.hidden_dim)
    p.add_argument("--depth", type=int, default=cfg.model.depth)
    p.add_argument("--heads", type=int, default=cfg.model.heads)
    p.add_argument("--dim_head", type=int, default=cfg.model.dim_head)
    p.add_argument("--ff_mult", type=int, default=cfg.model.ff_mult)
    p.add_argument("--dropout", type=float, default=cfg.model.dropout)
    p.add_argument("--attn_backend", type=str, default="torch",
                   choices=["flash_attn", "torch"],
                   help="Attention backend: 'flash_attn' (recommended) or 'torch'")

    # Mean-Flow hyperparams
    p.add_argument("--flow_ratio", type=float, default=cfg.mean_flow.flow_ratio)
    p.add_argument("--time_mu", type=float, default=cfg.mean_flow.time_mu)
    p.add_argument("--time_sigma", type=float, default=cfg.mean_flow.time_sigma)
    p.add_argument("--adaptive_gamma", type=float, default=cfg.mean_flow.adaptive_gamma)
    p.add_argument("--adaptive_c", type=float, default=cfg.mean_flow.adaptive_c)

    # Training
    p.add_argument("--epochs", type=int, default=cfg.train.epochs)
    p.add_argument("--batch_size", type=int, default=cfg.train.batch_size)
    p.add_argument("--num_workers", type=int, default=cfg.train.num_workers)
    p.add_argument("--compile", action="store_true", default=False,
                   help="torch.compile the DiT backbone for faster training (PyTorch 2.x)")
    p.add_argument("--lr", type=float, default=cfg.train.lr)
    p.add_argument("--lr_min", type=float, default=cfg.train.lr_min)
    p.add_argument("--weight_decay", type=float, default=cfg.train.weight_decay)
    p.add_argument("--grad_clip", type=float, default=cfg.train.grad_clip)
    p.add_argument("--fp16", action="store_true", default=cfg.train.fp16)

    # Logging / checkpointing
    p.add_argument("--save_dir", type=str, default=cfg.train.save_dir)
    p.add_argument("--save_every", type=int, default=cfg.train.save_every)
    p.add_argument("--log_every", type=int, default=cfg.train.log_every)
    p.add_argument("--resume", type=str, default=cfg.train.resume, help="Path to checkpoint to resume from")

    return p


if __name__ == "__main__":
    args = get_parser().parse_args()
    train(args)
