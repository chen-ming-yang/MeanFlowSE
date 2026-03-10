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
    clip_len: float = 4.0          # seconds per training clip
    mix_on_the_fly: bool = True    # mix clean+noise at runtime
    snr_low: float = -5.0          # dB
    snr_high: float = 20.0         # dB
    augment: bool = True           # waveform-level augmentation on noisy signal
    # Optional directory of RIR .wav files for real room impulse response augmentation.
    # If None, only synthetic exponential IRs are used.
    rir_dir: str = None


@dataclass
class ModelConfig:
    # SSL encoder (WavLM)
    ssl_model: str = "microsoft/wavlm-large"
    ssl_layers: int = 25
    ssl_dim: int = 1024            # WavLM-large hidden size

    # VAE
    latent_dim: int = 512

    # DiT backbone  (paper: N=8, heads=8, hidden=512, FFN=2048)
    hidden_dim: int = 512
    depth: int = 8
    heads: int = 8
    dim_head: int = 64
    ff_mult: int = 4               # FFN dim = hidden_dim * ff_mult = 2048
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
    epochs: int = 100
    batch_size: int = 16
    num_workers: int = 4
    lr: float = 1e-4
    lr_min: float = 1e-6
    weight_decay: float = 1e-2
    grad_clip: float = 1.0
    fp16: bool = False
    save_dir: str = "checkpoints"
    save_every: int = 5            # save checkpoint every N epochs
    log_every: int = 50            # print log every N steps
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
