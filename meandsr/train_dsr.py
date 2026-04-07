"""
Training script for MeanFlowSE model (dysarthria speech restoration).

Loads pre-aligned equal-length paired audio from a JSON manifest produced by
meandsr/build_equal_length_pairs.py.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Ensure project root is on sys.path so top-level modules (mean_flow, dit, …) are importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import soundfile as sf

from meandsr.dsr_mean_flow import MeanFlowDSR, SSLEncoder
from dit import DiTBackbone


def _compute_grad_l2_norm(parameters) -> float:
    total = 0.0
    for p in parameters:
        if p.grad is None:
            continue
        g = p.grad.detach()
        total += g.pow(2).sum().item()
    return total ** 0.5


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class DysarthriaDataset(Dataset):
    """Paired dysarthric-normal speech dataset loaded from a JSON manifest.

    Each entry in the manifest contains paths to equal-length dysarthria and
    normal audio files (produced by build_equal_length_pairs.py).
    """

    def __init__(
        self,
        manifest_path: str,
        project_root: str,
        sample_rate: int = 16000,
        max_samples: int = 96000,  # 6 seconds at 16kHz
    ):
        self.sample_rate = sample_rate
        self.max_samples = max_samples
        self.project_root = Path(project_root)

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        self.pairs = manifest["pairs"]
        assert len(self.pairs) > 0, f"No pairs found in {manifest_path}"
        print(f"Loaded {len(self.pairs)} equal-length dysarthric-normal pairs from {manifest_path}")

    def __len__(self) -> int:
        return len(self.pairs)

    def _load_raw(self, path: Path) -> torch.Tensor:
        """Load audio file as a mono float tensor."""
        wav_data, sr = sf.read(str(path), always_2d=True)
        wav = torch.from_numpy(wav_data.T).float()
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        return wav.squeeze(0)

    def __getitem__(self, idx: int):
        entry = self.pairs[idx]
        dys_path = self.project_root / entry["dysarthria"]
        normal_path = self.project_root / entry["normal"]

        dys_wav = self._load_raw(dys_path)
        normal_wav = self._load_raw(normal_path)

        # Both audios are already equal length; just apply max_samples truncation
        total_len = dys_wav.shape[-1]

        if total_len > self.max_samples:
            start = torch.randint(0, total_len - self.max_samples + 1, (1,)).item()
            dys_wav = dys_wav[start : start + self.max_samples]
            normal_wav = normal_wav[start : start + self.max_samples]

        return {
            "dys_wav": dys_wav,
            "clean_wav": normal_wav,
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

def build_model(cfg: argparse.Namespace) -> MeanFlowDSR:
    # SSL encoder for dysarthric speech
    ssl_encoder = SSLEncoder(
        model_name=cfg.ssl_model,
        num_layers=cfg.ssl_layers,
    )
    
    # VAE encoder/decoder (pretrained codec)
    if cfg.codec_model:
        from codec_vae import build_codec_vae
        vae_encoder, vae_decoder = build_codec_vae(cfg.codec_model, sr=cfg.sample_rate)
        latent_dim = vae_encoder.latent_dim
        print(f"Using pretrained codec: {cfg.codec_model} (latent_dim={latent_dim})")
    else:
        from mean_flow import VAEEncoder, VAEDecoder
        vae_encoder = VAEEncoder(latent_dim=cfg.latent_dim)
        vae_decoder = VAEDecoder(latent_dim=cfg.latent_dim)
        latent_dim = cfg.latent_dim
        print(f"WARNING: Using random VAE (latent_dim={latent_dim}). Set vae.codec_model for pretrained codec.")
    
    # DiT backbone (concat z_t + z_y, self-attention)
    dit_backbone = DiTBackbone(
        latent_dim=latent_dim,
        ssl_dim=cfg.ssl_dim,
        hidden_dim=cfg.hidden_dim,
        depth=cfg.depth,
        heads=cfg.heads,
        dim_head=cfg.dim_head,
        ff_mult=cfg.ff_mult,
        dropout=cfg.dropout,
        attn_backend=cfg.attn_backend,
    )
    
    return MeanFlowDSR(
        ssl_encoder=ssl_encoder,
        vae_encoder=vae_encoder,
        vae_decoder=vae_decoder,
        dit_backbone=dit_backbone,
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

    # Setup logger
    log_dir = Path(cfg.save_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "train.log"
    logger = logging.getLogger("MeanFlowSE_DSR")
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
        manifest_path=cfg.manifest,
        project_root=cfg.project_root,
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
                loss, metrics = model.forward_train(
                    batch["dys_wav"], batch["clean_wav"],
                    lengths=batch.get("clean_lengths"),
                )

            if not torch.isfinite(loss):
                logger.warning(
                    f"Non-finite loss at epoch={epoch} step={step+1}. "
                    f"Skipping optimizer update. metrics={metrics}"
                )
                continue

            prev_scale = scaler.get_scale()
            scaler.scale(loss).backward()

            grad_norm_pre_clip = _compute_grad_l2_norm(trainable)

            if cfg.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, cfg.grad_clip)
                grad_norm_post_clip = _compute_grad_l2_norm(trainable)
            else:
                grad_norm_post_clip = grad_norm_pre_clip

            scaler.step(optimizer)
            scaler.update()
            # If GradScaler overflows, optimizer.step is skipped; avoid advancing LR then.
            if scaler.get_scale() >= prev_scale:
                scheduler.step()

            epoch_loss += loss.item()
            global_step += 1

            lr_now = scheduler.get_last_lr()[0]
            step_bar.set_postfix(
                loss=f"{metrics['loss']:.4f}",
                lr=f"{lr_now:.2e}",
            )

            # Log to file
            if (step + 1) % cfg.log_every == 0:
                logger.info(
                    f"epoch={epoch} step={step+1}/{len(loader)}  "
                    f"loss={metrics['loss']:.4f}  "
                    f"delta_sq={metrics['delta_sq']:.4f}  "
                    f"mean_w={metrics['mean_weight']:.4f}  "
                    f"grad_pre={grad_norm_pre_clip:.4f}  "
                    f"grad_post={grad_norm_post_clip:.4f}  "
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

def load_config(config_path: str) -> dict:
    """Load YAML config file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def config_to_namespace(cfg: dict) -> argparse.Namespace:
    """Flatten nested config dict into a flat argparse.Namespace."""
    flat = {}
    # Data
    d = cfg.get("data", {})
    flat["manifest"] = d.get("manifest", "tmp_audio_equal_len/aligned_pairs_used.json")
    flat["project_root"] = d.get("project_root", ".")
    flat["sample_rate"] = d.get("sample_rate", 16000)
    flat["clip_len"] = d.get("clip_len", 6.0)
    # SSL
    s = cfg.get("ssl", {})
    flat["ssl_model"] = s.get("model", "microsoft/wavlm-base-plus")
    flat["ssl_layers"] = s.get("layers", 13)
    flat["ssl_dim"] = s.get("dim", 768)
    # VAE
    v = cfg.get("vae", {})
    flat["codec_model"] = v.get("codec_model", None)
    flat["latent_dim"] = v.get("latent_dim", 256)
    # Perceiver — removed; MeanFlowSE does not use a Perceiver bottleneck
    # DiT
    dt = cfg.get("dit", {})
    flat["hidden_dim"] = dt.get("hidden_dim", 512)
    flat["depth"] = dt.get("depth", 8)
    flat["heads"] = dt.get("heads", 8)
    flat["dim_head"] = dt.get("dim_head", 64)
    flat["ff_mult"] = dt.get("ff_mult", 4)
    flat["dropout"] = dt.get("dropout", 0.1)
    flat["attn_backend"] = dt.get("attn_backend", "flash_attn")
    # Flow
    fl = cfg.get("flow", {})
    flat["flow_ratio"] = fl.get("flow_ratio", 0.25)
    flat["time_mu"] = fl.get("time_mu", -0.4)
    flat["time_sigma"] = fl.get("time_sigma", 1.0)
    flat["adaptive_gamma"] = fl.get("adaptive_gamma", 0.5)
    flat["adaptive_c"] = fl.get("adaptive_c", 1e-3)
    # Training
    t = cfg.get("training", {})
    flat["epochs"] = t.get("epochs", 100)
    flat["batch_size"] = t.get("batch_size", 8)
    flat["num_workers"] = t.get("num_workers", 4)
    flat["lr"] = t.get("lr", 1e-4)
    flat["lr_min"] = t.get("lr_min", 1e-6)
    flat["weight_decay"] = t.get("weight_decay", 1e-4)
    flat["grad_clip"] = t.get("grad_clip", 1.0)
    flat["fp16"] = t.get("fp16", False)
    # Logging
    lg = cfg.get("logging", {})
    flat["save_dir"] = lg.get("save_dir", "checkpoints_dsr")
    flat["save_every"] = lg.get("save_every", 1)
    flat["log_every"] = lg.get("log_every", 20)
    flat["resume"] = lg.get("resume", None)

    return argparse.Namespace(**flat)


def get_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train MeanFlowSE for DSR")

    p.add_argument("--config", type=str, default="meandsr/config_dsr.yaml",
                   help="Path to YAML config file")

    # All other args serve as CLI overrides for the config file
    # Data
    p.add_argument("--manifest", type=str, default=None)
    p.add_argument("--project_root", type=str, default=None)
    p.add_argument("--sample_rate", type=int, default=None)
    p.add_argument("--clip_len", type=float, default=None)
    # SSL
    p.add_argument("--ssl_model", type=str, default=None)
    p.add_argument("--ssl_layers", type=int, default=None)
    p.add_argument("--ssl_dim", type=int, default=None)
    # VAE
    p.add_argument("--codec_model", type=str, default=None)
    p.add_argument("--latent_dim", type=int, default=None)
    # DiT
    p.add_argument("--hidden_dim", type=int, default=None)
    p.add_argument("--depth", type=int, default=None)
    p.add_argument("--heads", type=int, default=None)
    p.add_argument("--dim_head", type=int, default=None)
    p.add_argument("--ff_mult", type=int, default=None)
    p.add_argument("--dropout", type=float, default=None)
    p.add_argument("--attn_backend", type=str, default=None)
    # Flow
    p.add_argument("--flow_ratio", type=float, default=None)
    p.add_argument("--time_mu", type=float, default=None)
    p.add_argument("--time_sigma", type=float, default=None)
    p.add_argument("--adaptive_gamma", type=float, default=None)
    p.add_argument("--adaptive_c", type=float, default=None)
    # Training
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--lr_min", type=float, default=None)
    p.add_argument("--weight_decay", type=float, default=None)
    p.add_argument("--grad_clip", type=float, default=None)
    p.add_argument("--fp16", action="store_true", default=None)
    # Logging
    p.add_argument("--save_dir", type=str, default=None)
    p.add_argument("--save_every", type=int, default=None)
    p.add_argument("--log_every", type=int, default=None)
    p.add_argument("--resume", type=str, default=None)

    return p


if __name__ == "__main__":
    parser = get_parser()
    cli_args = parser.parse_args()

    # Load config from YAML
    cfg_dict = load_config(cli_args.config)
    args = config_to_namespace(cfg_dict)

    # Apply CLI overrides (any non-None CLI arg overrides the config value)
    for key, val in vars(cli_args).items():
        if key == "config":
            continue
        if val is not None:
            setattr(args, key, val)

    train(args)
