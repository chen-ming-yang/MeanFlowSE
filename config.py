"""
Central configuration for MeanFlowSE.

Edit the fields below to change any setting, then run:
    python train.py          # uses this config by default
    python train.py --data_root /other/path   # CLI overrides still work
"""

from dataclasses import dataclass, field


@dataclass
class DataConfig:
    # DNS root: contains clean/ or datasets.clean.*/ subdirs
    data_root: str = "/data/dns"
    # Separate noise directory
    noise_dir: str = "/data/dns/noise"
    sample_rate: int = 16000
    clip_len: float = 10.0          # seconds per training clip
    mix_on_the_fly: bool = False    # mix clean+noise at runtime
    snr_low: float = -5.0          # dB
    snr_high: float = 20.0         # dB
    augment: bool = True           # waveform-level augmentation on noisy signal
    # Optional directory of RIR .wav files for real room impulse response augmentation.
    # If None, only synthetic exponential IRs are used.
    rir_dir: str = None
    # Dataset layout:
    #   "default"    – original DNS Challenge layout: datasets.clean.*/ + separate noise_dir
    #   "paired_dir" – flat clean/ + noisy/ under data_root, paired by fileid_N suffix
    dns_layout: str = "default"


@dataclass
class ModelConfig:
    # SSL encoder (WavLM)
    ssl_model: str = "microsoft/wavlm-base-plus"
    ssl_layers: int = 13
    ssl_dim: int = 768             # WavLM-base-plus hidden size

    # VAE
    vae_type: str = "codec"        # "default" (placeholder), "wave_vae", or "codec" (EnCodec/DAC)
    codec_model: str = "dac_16khz" # codec model name (for vae_type="codec")
    vae_ckpt: str = None           # path to pretrained WaveVAE checkpoint (for vae_type="wave_vae")
    vae_hop_length: int = 640      # WaveVAE hop length (640→25Hz, 1280→12.5Hz @ 16kHz)
    latent_dim: int = 1024         # DAC 16kHz latent dim

    # DiT backbone
    hidden_dim: int = 768
    depth: int = 8
    heads: int = 12
    dim_head: int = 64
    ff_mult: int = 4               # FFN dim = hidden_dim * ff_mult = 3072
    dropout: float = 0.1


@dataclass
class MeanFlowConfig:
    # Paper configuration
    flow_ratio: float = 0.25
    time_mu: float = -0.4          # log-normal mean for time sampling
    time_sigma: float = 1.0        # log-normal std for time sampling
    adaptive_gamma: float = 0.5    # adaptive loss weighting γ
    adaptive_c: float = 1e-3       # adaptive loss weighting c


@dataclass
class TrainConfig:
    epochs: int = 200
    loader_mode: str = "fixed"      # "dynamic" (max_tokens) or "fixed" (batch_size)
    batch_size: int = 8                # used when loader_mode == "fixed"
    fixed_clip_len: float = 2.0        # seconds, used when loader_mode == "fixed"
    max_tokens: int = 960_000       # total audio samples per batch (~60s @ 16kHz)
    max_batch_size: int = 64         # hard upper-bound on utterances per batch
    num_workers: int = 16
    prefetch_factor: int = 4         # batches prefetched per worker
    lr: float = 1e-4
    lr_min: float = 1e-6
    lr_gamma: float = 0.99           # exponential decay factor per epoch
    weight_decay: float = 1e-2
    grad_clip: float = 1.0
    fp16: bool = True
    save_dir: str = "checkpoints"
    save_every: int = 1            # save checkpoint every N epochs
    log_every: int = 10            # print log every N steps
    resume: str = None             # path to checkpoint to resume from


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    mean_flow: MeanFlowConfig = field(default_factory=MeanFlowConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_namespace(self):
        """Flatten into an argparse.Namespace so train() can consume it directly."""
        import argparse
        ns = argparse.Namespace()
        for cfg_section in (self.data, self.model, self.mean_flow, self.train):
            for k, v in vars(cfg_section).items():
                setattr(ns, k, v)
        # train() expects `no_augment` (inverted flag) rather than `augment`
        ns.no_augment = not self.data.augment
        return ns


# ---------------------------------------------------------------------------
# Default config instance — import and modify this in your own scripts
# ---------------------------------------------------------------------------
default_config = Config()
