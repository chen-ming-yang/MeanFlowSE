"""
Training script for OneToManyDysarthriaSE model.

Expected dataset layout:
    <data_root>/
        dysarthric/     *.wav   dysarthric speech utterances
        normal/         *.wav   corresponding normal speech
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import soundfile as sf

from meandsr.dsr_mean_flow import (
    OneToManyDysarthriaSE,
    DirectMappingDiTBackbone,
    PerceiverBottleneck,
    LengthPredictor,
)
from mean_flow import SSLEncoder, VAEEncoder, VAEDecoder


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class DysarthriaDataset(Dataset):
    """Paired dysarthric-normal speech dataset."""

    def __init__(
        self,
        dys_root: str,
        normal_root: str,
        sample_rate: int = 16000,
        max_samples: int = 96000,  # 6 seconds at 16kHz
    ):
        self.sample_rate = sample_rate
        self.max_samples = max_samples

        self.dys_files = sorted(Path(dys_root).glob("**/*.wav"))
        self.normal_files = sorted(Path(normal_root).glob("**/*.wav"))

        assert len(self.dys_files) > 0, f"No dysarthric files found in {dys_root}"
        assert len(self.normal_files) > 0, f"No normal files found in {normal_root}"
        
        # Assume 1:1 pairing by filename
        print(f"Loaded {len(self.dys_files)} dysarthric-normal pairs")

    def __len__(self) -> int:
        return min(len(self.dys_files), len(self.normal_files))

    def _load_audio(self, path: Path) -> torch.Tensor:
        """Load audio, convert to mono, resample if needed."""
        wav_data, sr = sf.read(str(path), always_2d=True)
        wav = torch.from_numpy(wav_data.T).float()
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        wav = wav.squeeze(0)
        
        # Cap length
        if wav.shape[-1] > self.max_samples:
            start = torch.randint(0, wav.shape[-1] - self.max_samples + 1, (1,)).item()
            wav = wav[start: start + self.max_samples]
        
        return wav

    def __getitem__(self, idx: int):
        dys_wav = self._load_audio(self.dys_files[idx])
        normal_wav = self._load_audio(self.normal_files[idx])
        
        return {
            "dys_wav": dys_wav,
            "clean_wav": normal_wav,  # Match expected key in forward_train
        }


def collate_fn(batch: list[dict]) -> dict:
    """Collate variable-length samples with padding masks."""
    dys_wavs = [item["dys_wav"] for item in batch]
    clean_wavs = [item["clean_wav"] for item in batch]
    
    # Track original lengths
    dys_lengths = torch.tensor([w.shape[-1] for w in dys_wavs], dtype=torch.long)
    clean_lengths = torch.tensor([w.shape[-1] for w in clean_wavs], dtype=torch.long)
    
    max_len = max(max(w.shape[-1] for w in dys_wavs), max(w.shape[-1] for w in clean_wavs))
    
    dys_batch = torch.stack([F.pad(w, (0, max_len - w.shape[-1])) for w in dys_wavs])
    clean_batch = torch.stack([F.pad(w, (0, max_len - w.shape[-1])) for w in clean_wavs])
    
    return {
        "dys_wav": dys_batch,
        "clean_wav": clean_batch,
        "dys_lengths": dys_lengths,
        "clean_lengths": clean_lengths,
    }


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_model(cfg: argparse.Namespace) -> OneToManyDysarthriaSE:
    # SSL encoder for dysarthric speech
    ssl_encoder = SSLEncoder(
        model_name=cfg.ssl_model,
        num_layers=cfg.ssl_layers,
    )
    
    # VAE encoder/decoder (frozen, pretrained)
    vae_encoder = VAEEncoder(latent_dim=cfg.latent_dim)
    vae_decoder = VAEDecoder(latent_dim=cfg.latent_dim)
    
    # Perceiver bottleneck
    perceiver = PerceiverBottleneck(
        input_dim=cfg.ssl_dim,
        output_dim=cfg.bottleneck_dim,
        num_latents_tokens=cfg.num_latents,
        num_heads=cfg.perceiver_heads,
        num_layers=cfg.perceiver_layers,
        dropout=cfg.dropout,
    )
    
    # Length predictor
    length_predictor = LengthPredictor(
        input_dim=cfg.bottleneck_dim,
        hidden_dim=cfg.hidden_dim // 2,
    )
    
    # DiT backbone with cross-attention
    dit_backbone = DirectMappingDiTBackbone(
        latent_dim=cfg.latent_dim,
        bottleneck_dim=cfg.bottleneck_dim,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.depth,
        num_heads=cfg.heads,
        ff_dim=cfg.hidden_dim * cfg.ff_mult,
        dropout=cfg.dropout,
    )
    
    return OneToManyDysarthriaSE(
        ssl_encoder=ssl_encoder,
        vae_encoder=vae_encoder,
        vae_decoder=vae_decoder,
        dit_backbone=dit_backbone,
        perceiver=perceiver,
        length_predictor=length_predictor,
        flow_ratio=cfg.flow_ratio,
        time_mu=cfg.time_mu,
        time_sigma=cfg.time_sigma,
        adaptive_gamma=cfg.adaptive_gamma,
        adaptive_c=cfg.adaptive_c,
        length_loss_weight=cfg.length_loss_weight,
    )


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(cfg: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Setup logger
    log_dir = Path(cfg.save_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "train.log"
    logger = logging.getLogger("OneToManyDSR")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(log_path, mode="a")
        fh.setFormatter(logging.Formatter("%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
        logger.addHandler(fh)
    logger.info(f"Training started | device={device} | cfg={vars(cfg)}")
    print(f"Logging to {log_path}")

    # CUDA optimizations
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
        print("Enabled: TF32 matmul, cuDNN benchmark")

    # Dataset & loader
    dataset = DysarthriaDataset(
        dys_root=cfg.dys_root,
        normal_root=cfg.normal_root,
        sample_rate=cfg.sample_rate,
        max_samples=int(cfg.clip_len * cfg.sample_rate),
    )
    
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
    )

    # Model
    model = build_model(cfg).to(device)
    
    # Count trainable parameters
    trainable = [p for p in model.parameters() if p.requires_grad]
    frozen = [p for p in model.parameters() if not p.requires_grad]
    print(f"Trainable parameters: {sum(p.numel() for p in trainable) / 1e6:.1f} M")
    print(f"Frozen parameters: {sum(p.numel() for p in frozen) / 1e6:.1f} M")

    optimizer = AdamW(trainable, lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs * len(loader), eta_min=cfg.lr_min)

    # Resume from checkpoint
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
        for step, batch in enumerate(step_bar):
            # Move to device
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=cfg.fp16):
                loss, metrics = model.forward_train(batch)

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
                total=f"{metrics['total_loss']:.4f}",
                flow=f"{metrics['flow_loss']:.4f}",
                length=f"{metrics['length_loss']:.4f}",
                lr=f"{lr_now:.2e}",
            )

            # Log to file
            if (step + 1) % cfg.log_every == 0:
                logger.info(
                    f"epoch={epoch} step={step+1}/{len(loader)}  "
                    f"total={metrics['total_loss']:.4f}  "
                    f"flow={metrics['flow_loss']:.4f}  "
                    f"length={metrics['length_loss']:.4f}  "
                    f"pred_len={metrics['pred_length']:.1f}  "
                    f"gt_len={metrics['gt_length']:.1f}  "
                    f"lr={lr_now:.2e}"
                )

        avg_loss = epoch_loss / len(loader)
        epoch_bar.set_postfix(avg_loss=f"{avg_loss:.4f}")
        epoch_summary = f"Epoch {epoch:03d}  avg_loss={avg_loss:.4f}"
        tqdm.write(f"=== {epoch_summary} ===")
        logger.info(epoch_summary)

        # Save checkpoint
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
            logger.info(f"Saved checkpoint → {ckpt_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def get_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train OneToManyDysarthriaSE")

    # Data
    p.add_argument("--dys_root", type=str, required=True, help="Dysarthric speech directory")
    p.add_argument("--normal_root", type=str, required=True, help="Normal speech directory")
    p.add_argument("--sample_rate", type=int, default=16000)
    p.add_argument("--clip_len", type=float, default=6.0, help="Max training clip length (seconds)")

    # Model - SSL encoder
    p.add_argument("--ssl_model", type=str, default="microsoft/wavlm-base-plus")
    p.add_argument("--ssl_layers", type=int, default=13)
    p.add_argument("--ssl_dim", type=int, default=768, help="WavLM hidden size")

    # Model - VAE
    p.add_argument("--latent_dim", type=int, default=256)

    # Model - Perceiver
    p.add_argument("--bottleneck_dim", type=int, default=512)
    p.add_argument("--num_latents", type=int, default=64, help="Number of latent tokens in Perceiver")
    p.add_argument("--perceiver_heads", type=int, default=8)
    p.add_argument("--perceiver_layers", type=int, default=2)

    # Model - DiT backbone
    p.add_argument("--hidden_dim", type=int, default=512)
    p.add_argument("--depth", type=int, default=8)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--ff_mult", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)

    # Mean-Flow hyperparams
    p.add_argument("--flow_ratio", type=float, default=0.25)
    p.add_argument("--time_mu", type=float, default=-0.4)
    p.add_argument("--time_sigma", type=float, default=1.0)
    p.add_argument("--adaptive_gamma", type=float, default=0.5)
    p.add_argument("--adaptive_c", type=float, default=1e-3)
    p.add_argument("--length_loss_weight", type=float, default=0.1)

    # Training
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lr_min", type=float, default=1e-6)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--fp16", action="store_true", default=False)

    # Logging / checkpointing
    p.add_argument("--save_dir", type=str, default="checkpoints_dsr")
    p.add_argument("--save_every", type=int, default=10)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")

    return p


if __name__ == "__main__":
    args = get_parser().parse_args()
    train(args)
