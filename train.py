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
import gc
import logging
import time
from pathlib import Path

from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR

from mean_flow import MeanFlowSE, SSLEncoder, VAEEncoder, VAEDecoder
from dit import DiTBackbone
from config import Config, default_config
from dataset import DNSDataset, DynamicBatchSampler, dynamic_collate_fn


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_model(cfg: argparse.Namespace) -> MeanFlowSE:
    ssl_encoder = SSLEncoder(model_name=cfg.ssl_model, num_layers=cfg.ssl_layers)

    # VAE encoder/decoder selection
    vae_type = getattr(cfg, "vae_type", "default")
    if vae_type == "wave_vae":
        from wave_vae import build_wave_vae
        wave_vae = build_wave_vae(
            latent_dim=cfg.latent_dim,
            hop_length=getattr(cfg, "vae_hop_length", 640),
            pretrained_path=getattr(cfg, "vae_ckpt", None),
        )
        vae_encoder = wave_vae.encoder
        vae_decoder = wave_vae.decoder
    elif vae_type == "codec":
        from codec_vae import build_codec_vae
        vae_encoder, vae_decoder = build_codec_vae(
            getattr(cfg, "codec_model", "facebook/encodec_24khz"),
            sr=cfg.sample_rate,
        )
    else:
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

    # ---- Set up file logger ----
    log_dir = Path(cfg.save_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "train.log"
    logger = logging.getLogger("MeanFlowSE")
    logger.setLevel(logging.INFO)
    # Avoid duplicate handlers on resume
    if not logger.handlers:
        fh = logging.FileHandler(log_path, mode="a")
        fh.setFormatter(logging.Formatter("%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
        logger.addHandler(fh)
    logger.info(f"Training started | device={device} | cfg={vars(cfg)}")
    print(f"Logging to {log_path}")

    # ---- CUDA performance knobs ----
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True      # TF32 matmul (~2x for non-fp16 ops)
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True              # auto-tune conv kernels
        torch.set_float32_matmul_precision("high")
        print("Enabled: TF32 matmul, cuDNN benchmark")

    # Dataset & loader
    dataset_clip_len = cfg.clip_len
    dataset_segment_len = cfg.fixed_clip_len if cfg.loader_mode == "fixed" else None
    dataset_use_all_segments = cfg.loader_mode == "fixed"
    if cfg.loader_mode == "fixed":
        print(
            f"Fixed mode: split each {cfg.clip_len:.2f}s file into deterministic "
            f"{cfg.fixed_clip_len:.2f}s segments and use all segments"
        )
    dataset = DNSDataset(
        dns_root=cfg.data_root,
        noise_dir=cfg.noise_dir,
        sample_rate=cfg.sample_rate,
        clip_len=dataset_clip_len,
        segment_len=dataset_segment_len,
        use_all_segments=dataset_use_all_segments,
        mix_on_the_fly=cfg.mix_on_the_fly,
        snr_low=cfg.snr_low,
        snr_high=cfg.snr_high,
        augment=not cfg.no_augment,
        rir_dir=cfg.rir_dir,
        dns_layout=cfg.dns_layout,
    )
    if cfg.loader_mode == "dynamic":
        batch_sampler = DynamicBatchSampler(
            durations=dataset.durations,
            max_tokens=cfg.max_tokens,
            max_batch_size=cfg.max_batch_size,
            shuffle=True,
            drop_last=True,
        )
        print(
            f"DynamicBatchSampler: max_tokens={cfg.max_tokens}, "
            f"max_batch_size={cfg.max_batch_size}, "
            f"~{len(batch_sampler)} batches/epoch"
        )
        _pf = getattr(cfg, 'prefetch_factor', 4)
        loader = DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=cfg.num_workers,
            pin_memory=True,
            persistent_workers=cfg.num_workers > 0,
            prefetch_factor=_pf if cfg.num_workers > 0 else None,
            collate_fn=dynamic_collate_fn,
        )
    else:
        print(
            f"FixedBatchLoader: batch_size={cfg.batch_size}, "
            f"shuffle=True, drop_last=True"
        )
        _pf = getattr(cfg, 'prefetch_factor', 4)
        loader = DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=cfg.num_workers,
            pin_memory=True,
            persistent_workers=cfg.num_workers > 0,
            prefetch_factor=_pf if cfg.num_workers > 0 else None,
            collate_fn=dynamic_collate_fn,
        )

    # Model
    model = build_model(cfg).to(device)

    # Compile models for faster execution (PyTorch 2.x)
    if cfg.compile:
        print("Compiling SSL encoder + DiT backbone with torch.compile ...")
        model.ssl_encoder.ssl_model = torch.compile(model.ssl_encoder.ssl_model, mode="reduce-overhead")
        model.dit_backbone = torch.compile(model.dit_backbone)

    # Only train the backbone; SSL encoder + VAE are frozen inside MeanFlowSE
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable parameters: {sum(p.numel() for p in trainable) / 1e6:.1f} M")

    optimizer = AdamW(trainable, lr=cfg.lr, weight_decay=cfg.weight_decay)
    lr_gamma = getattr(cfg, 'lr_gamma', 0.99)
    scheduler = ExponentialLR(optimizer, gamma=lr_gamma)

    # Optional: resume from checkpoint
    start_epoch = 0
    if cfg.resume:
        ckpt = torch.load(cfg.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        # Only restore scheduler if the saved type matches the current one
        saved_sched = ckpt.get("scheduler", {})
        saved_type = saved_sched.get("_scheduler_type", None)
        if saved_type == type(scheduler).__name__:
            scheduler.load_state_dict(saved_sched)
        else:
            print(f"Scheduler type changed ({saved_type or 'unknown'} → "
                  f"{type(scheduler).__name__}), resetting scheduler with lr={cfg.lr}")
        # Always reset optimizer lr to cfg.lr so new training config takes effect
        for pg in optimizer.param_groups:
            pg["lr"] = cfg.lr
            pg["initial_lr"] = cfg.lr
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed from epoch {start_epoch}")

    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Mixed precision scaler
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.fp16)

    global_step = start_epoch * len(loader)

    # Disable automatic garbage collection to prevent random stalls.
    # We'll manually collect every GC_EVERY steps.
    gc.disable()
    GC_EVERY = 50

    epoch_bar = tqdm(range(start_epoch, cfg.epochs), desc="Epochs", unit="epoch")
    for epoch in epoch_bar:
        model.train()
        epoch_loss = 0.0
        # Timing accumulators for the whole epoch:
        #   epoch_data_time – wall-clock seconds the training loop was blocked waiting
        #                      for the DataLoader (workers + disk I/O + augmentation).
        #                      High data% → CPU/IO bottleneck; consider more workers,
        #                      faster storage, or lighter augmentations.
        #   epoch_gpu_time  – wall-clock seconds spent on GPU work per step
        #                     (forward pass, loss, backward, optimizer, scheduler).
        #                     High gpu% → model is compute-bound, which is ideal.
        epoch_data_time = 0.0
        epoch_gpu_time = 0.0
        epoch_start = time.perf_counter()
        # Per-window tracking to catch stalls on non-logged steps
        window_data_max = 0.0
        window_data_sum = 0.0
        window_data_count = 0
        # Timestamp right before the first batch is fetched so the very first
        # data_elapsed measurement covers the initial DataLoader warm-up time.
        data_start = time.perf_counter()

        step_bar = tqdm(loader, desc=f"Epoch {epoch:03d}", unit="batch", leave=False)
        for step, (noisy, clean, lengths) in enumerate(step_bar):
            # ---- data loading time ----
            # data_elapsed measures the time between the end of the previous GPU step
            # and the moment this batch is ready in Python (i.e. the DataLoader's
            # collate + prefetch latency seen by the main thread).
            data_end = time.perf_counter()
            data_elapsed = data_end - data_start
            epoch_data_time += data_elapsed
            window_data_max = max(window_data_max, data_elapsed)
            window_data_sum += data_elapsed
            window_data_count += 1

            # ---- GPU compute time ----
            # Start timing just before transferring tensors to the GPU.
            # torch.cuda.synchronize() is called at the end of this section to
            # ensure all CUDA kernels have finished before we record gpu_end,
            # giving an accurate wall-clock measure of GPU work.
            gpu_start = time.perf_counter()

            noisy = noisy.to(device, non_blocking=True)
            clean = clean.to(device, non_blocking=True)
            lengths = lengths.to(device, non_blocking=True)

            optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=cfg.fp16):
                loss, stats = model.forward_train(noisy, clean, lengths=lengths)

            scaler.scale(loss).backward()

            if cfg.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, cfg.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            # Only synchronize on logging steps to measure accurate timing.
            # Syncing every step kills CPU-GPU pipelining and causes GPU idle gaps.
            is_log_step = (step + 1) % cfg.log_every == 0
            if is_log_step and device.type == "cuda":
                torch.cuda.synchronize()
            gpu_end = time.perf_counter()
            gpu_elapsed = gpu_end - gpu_start
            epoch_gpu_time += gpu_elapsed

            epoch_loss += loss.item()
            global_step += 1

            lr_now = scheduler.get_last_lr()[0]
            step_bar.set_postfix(
                loss=f"{stats['loss']:.4f}",
                data=f"{data_elapsed:.3f}s",
                gpu=f"{gpu_elapsed:.3f}s",
                bs=f"{noisy.shape[0]}",
                lr=f"{lr_now:.2e}",
            )

            # Log to file periodically
            if is_log_step:
                window_data_avg = window_data_sum / window_data_count if window_data_count > 0 else 0.0
                logger.info(
                    f"epoch={epoch} step={step+1}/{len(loader)}  "
                    f"loss={stats['loss']:.4f}  delta_sq={stats['delta_sq']:.4f}  "
                    f"w={stats['mean_weight']:.4f}  "
                    f"data={data_elapsed:.3f}s  data_max={window_data_max:.3f}s  data_avg={window_data_avg:.3f}s  "
                    f"gpu={gpu_elapsed:.3f}s  "
                    f"bs={noisy.shape[0]}  lr={lr_now:.2e}"
                )
                # Reset window counters
                window_data_max = 0.0
                window_data_sum = 0.0
                window_data_count = 0

            # Manual garbage collection every GC_EVERY steps to avoid
            # random GC stalls that starve the GPU.
            if (step + 1) % GC_EVERY == 0:
                gc.collect()

            # Reset the data-loading stopwatch so the next iteration measures
            # only the time spent waiting for the following batch, not GPU compute.
            data_start = time.perf_counter()

        scheduler.step()
        avg_loss = epoch_loss / len(loader)
        epoch_elapsed = time.perf_counter() - epoch_start
        # data_pct + gpu_pct ≈ 100%.  The remainder is Python overhead
        # (loss.item() device→host copy, tqdm/logging, scheduler.step(), etc.).
        # Rule of thumb: gpu_pct > 80% is healthy; data_pct > 30% signals a
        # data pipeline bottleneck worth addressing (more workers, prefetch, etc.).
        data_pct = 100.0 * epoch_data_time / epoch_elapsed if epoch_elapsed > 0 else 0
        gpu_pct = 100.0 * epoch_gpu_time / epoch_elapsed if epoch_elapsed > 0 else 0
        epoch_bar.set_postfix(avg_loss=f"{avg_loss:.4f}")
        epoch_summary = (
            f"Epoch {epoch:03d}  loss={avg_loss:.4f}  "
            f"total={epoch_elapsed:.1f}s  "
            f"data={epoch_data_time:.1f}s ({data_pct:.0f}%)  "
            f"gpu={epoch_gpu_time:.1f}s ({gpu_pct:.0f}%)"
        )
        tqdm.write(f"=== {epoch_summary} ===")
        logger.info(epoch_summary)

        # Save checkpoint every N epochs and at the last epoch
        if (epoch + 1) % cfg.save_every == 0 or epoch == cfg.epochs - 1:
            ckpt_path = save_dir / f"ckpt_epoch{epoch:03d}.pt"
            sched_state = scheduler.state_dict()
            sched_state["_scheduler_type"] = type(scheduler).__name__
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": sched_state,
                    "cfg": vars(cfg),
                },
                ckpt_path,
            )
            tqdm.write(f"Saved checkpoint → {ckpt_path}")
            logger.info(f"Saved checkpoint → {ckpt_path}")

            # Keep only the most recent 20 checkpoints
            existing = sorted(save_dir.glob("ckpt_epoch*.pt"))
            if len(existing) > 20:
                for old_ckpt in existing[:-20]:
                    old_ckpt.unlink()
                    tqdm.write(f"Removed old checkpoint → {old_ckpt}")
                    logger.info(f"Removed old checkpoint → {old_ckpt}")


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
                   help="Mix clean+noise at runtime (default True)")
    p.add_argument("--no_mix_on_the_fly", action="store_false", dest="mix_on_the_fly",
                   help="Use pre-mixed noisy audio (data_root must have noisy/ subdir)")
    p.add_argument("--snr_low", type=float, default=cfg.data.snr_low)
    p.add_argument("--snr_high", type=float, default=cfg.data.snr_high)
    p.add_argument("--no_augment", action="store_true", default=not cfg.data.augment,
                   help="Disable waveform augmentation")
    p.add_argument("--rir_dir", type=str, default=cfg.data.rir_dir,
                   help="Directory of RIR .wav files; if set, real RIRs are randomly mixed with synthetic IRs")
    p.add_argument("--dns_layout", type=str, default=cfg.data.dns_layout,
                   choices=["default", "paired_dir"],
                   help=(
                       "Dataset layout: 'default' (datasets.clean.*/ + noise_dir) or "
                       "'paired_dir' (flat clean/ + noisy/ under data_root, paired by fileid_N)"
                   ))

    # Model
    p.add_argument("--ssl_model", type=str, default=cfg.model.ssl_model)
    p.add_argument("--ssl_layers", type=int, default=cfg.model.ssl_layers)
    p.add_argument("--ssl_dim", type=int, default=cfg.model.ssl_dim, help="WavLM hidden size (768 for base-plus, 1024 for large)")
    p.add_argument("--vae_type", type=str, default=cfg.model.vae_type,
                   choices=["default", "wave_vae", "codec"],
                   help="VAE type: 'default' (placeholder), 'wave_vae', or 'codec' (EnCodec/DAC)")
    p.add_argument("--codec_model", type=str, default=cfg.model.codec_model,
                   help="Codec model name when vae_type='codec' (e.g. 'dac_16khz', 'facebook/encodec_24khz')")
    p.add_argument("--vae_ckpt", type=str, default=cfg.model.vae_ckpt,
                   help="Path to pretrained WaveVAE checkpoint (for vae_type='wave_vae')")
    p.add_argument("--vae_hop_length", type=int, default=cfg.model.vae_hop_length,
                   help="WaveVAE hop length (for vae_type='wave_vae')")
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
    p.add_argument("--loader_mode", type=str, default=cfg.train.loader_mode,
                   choices=["dynamic", "fixed"],
                   help="Batching mode: 'dynamic' (max_tokens) or 'fixed' (batch_size)")
    p.add_argument("--batch_size", type=int, default=cfg.train.batch_size,
                   help="Batch size used when --loader_mode fixed")
    p.add_argument("--fixed_clip_len", type=float, default=cfg.train.fixed_clip_len,
                   help="Clip length (seconds) used when --loader_mode fixed")
    p.add_argument("--max_tokens", type=int, default=cfg.train.max_tokens,
                   help="Max total audio samples (tokens) per batch for dynamic batching")
    p.add_argument("--max_batch_size", type=int, default=cfg.train.max_batch_size,
                   help="Hard upper-bound on utterances per batch")
    p.add_argument("--num_workers", type=int, default=cfg.train.num_workers)
    p.add_argument("--compile", action="store_true", default=False,
                   help="torch.compile the DiT backbone for faster training (PyTorch 2.x)")
    p.add_argument("--lr", type=float, default=cfg.train.lr)
    p.add_argument("--lr_min", type=float, default=cfg.train.lr_min)
    p.add_argument("--lr_gamma", type=float, default=cfg.train.lr_gamma,
                   help="Exponential LR decay factor per epoch (default 0.99)")
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
