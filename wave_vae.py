"""
WaveVAE (Flow-VAE) from KALL-E for MeanFlowSE.

Architecture based on the KALL-E paper (arXiv:2412.16846):
  - Encoder: Glow-WaveGAN style downsampling convolutions with residual blocks
  - Decoder: BigVGAN style transposed convolutions with Snake activation + residual blocks
  - Optional normalizing flow between encoder and decoder

For MeanFlowSE the VAE is pretrained and frozen. Only the encoder's mean output
(deterministic encoding) is used during MeanFlowSE training.

Usage:
    # Build model
    wave_vae = WaveVAE(latent_dim=256, hop_length=640)

    # Pretrain on clean speech  (see train_wave_vae.py)
    # Then freeze and use as drop-in replacement:
    model = MeanFlowSE(
        ssl_encoder=ssl_encoder,
        vae_encoder=wave_vae.encoder,
        vae_decoder=wave_vae.decoder,
        ...
    )
"""

from __future__ import annotations

import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Snake activation  (from BigVGAN)
# ---------------------------------------------------------------------------

class Snake(nn.Module):
    """Snake activation: x + (1/a) * sin^2(a * x)."""

    def __init__(self, channels: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + (1.0 / (self.alpha + 1e-9)) * torch.sin(self.alpha * x).pow(2)


# ---------------------------------------------------------------------------
# Residual block (HiFi-GAN / BigVGAN style)
# ---------------------------------------------------------------------------

class ResBlock(nn.Module):
    """Multi-dilation residual block."""

    def __init__(self, channels: int, kernel_size: int = 3,
                 dilations: tuple[int, ...] = (1, 3, 5)):
        super().__init__()
        self.convs1 = nn.ModuleList()
        self.convs2 = nn.ModuleList()
        self.acts1 = nn.ModuleList()
        self.acts2 = nn.ModuleList()

        for d in dilations:
            pad = (kernel_size * d - d) // 2
            self.acts1.append(Snake(channels))
            self.convs1.append(
                nn.utils.weight_norm(
                    nn.Conv1d(channels, channels, kernel_size, dilation=d, padding=pad)
                )
            )
            self.acts2.append(Snake(channels))
            self.convs2.append(
                nn.utils.weight_norm(
                    nn.Conv1d(channels, channels, kernel_size, dilation=1,
                              padding=(kernel_size - 1) // 2)
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for act1, c1, act2, c2 in zip(self.acts1, self.convs1,
                                       self.acts2, self.convs2):
            xt = act1(x)
            xt = c1(xt)
            xt = act2(xt)
            xt = c2(xt)
            x = x + xt
        return x

    def remove_weight_norm(self):
        for c in self.convs1:
            nn.utils.remove_weight_norm(c)
        for c in self.convs2:
            nn.utils.remove_weight_norm(c)


# ---------------------------------------------------------------------------
# WaveVAE Encoder
# ---------------------------------------------------------------------------

class WaveVAEEncoder(nn.Module):
    """
    Downsampling encoder: waveform → (mu, log_var) in latent space.

    For MeanFlowSE we only use the mean (deterministic encoding).

    Args:
        latent_dim:    Dimension of latent vectors (paper: 256).
        base_channels: Base channel width (doubled at each downsampling stage).
        downsample_rates: Tuple of stride factors. Product = hop_length.
                         e.g. (8, 5, 4, 4) → hop=640 → 25 Hz @ 16 kHz.
        kernel_sizes:  Kernel sizes for each downsampling conv (default: 2*stride).
        resblock_kernel_sizes: Kernel sizes of residual blocks per stage.
        resblock_dilations:    Dilations per residual block.
    """

    def __init__(
        self,
        latent_dim: int = 256,
        base_channels: int = 32,
        downsample_rates: tuple[int, ...] = (8, 5, 4, 4),
        kernel_sizes: tuple[int, ...] | None = None,
        resblock_kernel_sizes: tuple[int, ...] = (3, 7, 11),
        resblock_dilations: tuple[tuple[int, ...], ...] = ((1, 3, 5), (1, 3, 5), (1, 3, 5)),
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.hop_length = math.prod(downsample_rates)

        if kernel_sizes is None:
            kernel_sizes = tuple(2 * s for s in downsample_rates)

        # Initial projection from waveform
        self.conv_pre = nn.utils.weight_norm(
            nn.Conv1d(1, base_channels, kernel_size=7, stride=1, padding=3)
        )

        # Downsampling layers
        self.downs = nn.ModuleList()
        self.down_resblocks = nn.ModuleList()
        ch = base_channels
        for i, (stride, k) in enumerate(zip(downsample_rates, kernel_sizes)):
            ch_next = min(ch * 2, 512)
            pad = (k - stride) // 2
            self.downs.append(
                nn.Sequential(
                    Snake(ch),
                    nn.utils.weight_norm(
                        nn.Conv1d(ch, ch_next, kernel_size=k, stride=stride, padding=pad)
                    ),
                )
            )
            # Residual blocks at this resolution
            res_layers = nn.ModuleList()
            for rk, rd in zip(resblock_kernel_sizes, resblock_dilations):
                res_layers.append(ResBlock(ch_next, kernel_size=rk, dilations=rd))
            self.down_resblocks.append(res_layers)
            ch = ch_next

        # Project to mu and log_var
        self.act_out = Snake(ch)
        self.conv_mu = nn.utils.weight_norm(
            nn.Conv1d(ch, latent_dim, kernel_size=3, padding=1)
        )
        self.conv_logvar = nn.utils.weight_norm(
            nn.Conv1d(ch, latent_dim, kernel_size=3, padding=1)
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform: (B, T) raw waveform
        Returns:
            z: (B, T_latent, latent_dim) — deterministic mean encoding
        """
        x = waveform.unsqueeze(1)  # (B, 1, T)
        x = self.conv_pre(x)

        for down, res_blocks in zip(self.downs, self.down_resblocks):
            x = down(x)
            for rb in res_blocks:
                x = rb(x)

        x = self.act_out(x)
        mu = self.conv_mu(x)        # (B, latent_dim, T_latent)
        # For MeanFlowSE: return deterministic mean only
        return mu.transpose(1, 2)   # (B, T_latent, latent_dim)

    def encode(self, waveform: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Full VAE encoding returning (mu, log_var) for training the VAE."""
        x = waveform.unsqueeze(1)
        x = self.conv_pre(x)

        for down, res_blocks in zip(self.downs, self.down_resblocks):
            x = down(x)
            for rb in res_blocks:
                x = rb(x)

        x = self.act_out(x)
        mu = self.conv_mu(x)
        logvar = self.conv_logvar(x)
        return mu, logvar  # both (B, latent_dim, T_latent)

    def remove_weight_norm(self):
        nn.utils.remove_weight_norm(self.conv_pre)
        for down in self.downs:
            nn.utils.remove_weight_norm(down[1])
        for res_blocks in self.down_resblocks:
            for rb in res_blocks:
                rb.remove_weight_norm()
        nn.utils.remove_weight_norm(self.conv_mu)
        nn.utils.remove_weight_norm(self.conv_logvar)


# ---------------------------------------------------------------------------
# WaveVAE Decoder  (BigVGAN-style)
# ---------------------------------------------------------------------------

class WaveVAEDecoder(nn.Module):
    """
    Upsampling decoder: latent → waveform.

    Args:
        latent_dim:     Dimension of latent vectors.
        base_channels:  Channel width at the deepest (coarsest) stage.
        upsample_rates: Tuple of stride factors (mirror of encoder downsample_rates).
        upsample_kernel_sizes: Kernel sizes for transposed convs (default: 2*stride).
        resblock_kernel_sizes: Kernel sizes of residual blocks per stage.
        resblock_dilations:    Dilations per residual block.
    """

    def __init__(
        self,
        latent_dim: int = 256,
        base_channels: int = 512,
        upsample_rates: tuple[int, ...] = (4, 4, 5, 8),
        upsample_kernel_sizes: tuple[int, ...] | None = None,
        resblock_kernel_sizes: tuple[int, ...] = (3, 7, 11),
        resblock_dilations: tuple[tuple[int, ...], ...] = ((1, 3, 5), (1, 3, 5), (1, 3, 5)),
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.hop_length = math.prod(upsample_rates)

        if upsample_kernel_sizes is None:
            upsample_kernel_sizes = tuple(2 * s for s in upsample_rates)

        # Initial projection
        self.conv_pre = nn.utils.weight_norm(
            nn.Conv1d(latent_dim, base_channels, kernel_size=7, padding=3)
        )

        # Upsampling layers
        self.ups = nn.ModuleList()
        self.up_resblocks = nn.ModuleList()
        ch = base_channels
        for i, (stride, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            ch_next = ch // 2
            pad = (k - stride) // 2
            self.ups.append(
                nn.Sequential(
                    Snake(ch),
                    nn.utils.weight_norm(
                        nn.ConvTranspose1d(ch, ch_next, kernel_size=k,
                                           stride=stride, padding=pad)
                    ),
                )
            )
            res_layers = nn.ModuleList()
            for rk, rd in zip(resblock_kernel_sizes, resblock_dilations):
                res_layers.append(ResBlock(ch_next, kernel_size=rk, dilations=rd))
            self.up_resblocks.append(res_layers)
            ch = ch_next

        # Final projection to mono waveform
        self.act_out = Snake(ch)
        self.conv_post = nn.utils.weight_norm(
            nn.Conv1d(ch, 1, kernel_size=7, padding=3)
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (B, T_latent, latent_dim)
        Returns:
            waveform: (B, T_audio)
        """
        x = z.transpose(1, 2)     # (B, latent_dim, T_latent)
        x = self.conv_pre(x)

        for up, res_blocks in zip(self.ups, self.up_resblocks):
            x = up(x)
            for rb in res_blocks:
                x = rb(x)

        x = self.act_out(x)
        x = self.conv_post(x)      # (B, 1, T_audio)
        x = torch.tanh(x)
        return x.squeeze(1)        # (B, T_audio)

    def remove_weight_norm(self):
        nn.utils.remove_weight_norm(self.conv_pre)
        for up in self.ups:
            nn.utils.remove_weight_norm(up[1])
        for res_blocks in self.up_resblocks:
            for rb in res_blocks:
                rb.remove_weight_norm()
        nn.utils.remove_weight_norm(self.conv_post)


# ---------------------------------------------------------------------------
# Full WaveVAE model (for pretraining)
# ---------------------------------------------------------------------------

class WaveVAE(nn.Module):
    """
    Complete WaveVAE model for pretraining.

    After pretraining, use wave_vae.encoder and wave_vae.decoder as frozen
    modules inside MeanFlowSE.

    Pretrain with: train_wave_vae.py
    """

    def __init__(
        self,
        latent_dim: int = 256,
        encoder_base_channels: int = 32,
        decoder_base_channels: int = 512,
        downsample_rates: tuple[int, ...] = (8, 5, 4, 4),
        resblock_kernel_sizes: tuple[int, ...] = (3, 7, 11),
        resblock_dilations: tuple[tuple[int, ...], ...] = ((1, 3, 5), (1, 3, 5), (1, 3, 5)),
    ):
        super().__init__()
        upsample_rates = tuple(reversed(downsample_rates))

        self.encoder = WaveVAEEncoder(
            latent_dim=latent_dim,
            base_channels=encoder_base_channels,
            downsample_rates=downsample_rates,
            resblock_kernel_sizes=resblock_kernel_sizes,
            resblock_dilations=resblock_dilations,
        )
        self.decoder = WaveVAEDecoder(
            latent_dim=latent_dim,
            base_channels=decoder_base_channels,
            upsample_rates=upsample_rates,
            resblock_kernel_sizes=resblock_kernel_sizes,
            resblock_dilations=resblock_dilations,
        )
        self.hop_length = self.encoder.hop_length

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, waveform: torch.Tensor):
        """
        Args:
            waveform: (B, T) raw audio
        Returns:
            recon: (B, T') reconstructed waveform
            mu, logvar: (B, latent_dim, T_latent) for KL loss
        """
        mu, logvar = self.encoder.encode(waveform)
        z = self.reparameterize(mu, logvar)
        z_for_dec = z.transpose(1, 2)  # (B, T_latent, latent_dim)
        recon = self.decoder(z_for_dec)
        return recon, mu, logvar

    @staticmethod
    def kl_loss(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Standard VAE KL divergence: KL(q(z|x) || N(0, I))."""
        return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


# ---------------------------------------------------------------------------
# Convenience builder
# ---------------------------------------------------------------------------

def build_wave_vae(
    latent_dim: int = 256,
    hop_length: int = 640,
    pretrained_path: str | None = None,
    device: torch.device | str = "cpu",
) -> WaveVAE:
    """
    Build WaveVAE and optionally load pretrained weights.

    Args:
        latent_dim:      Latent dimension (paper: 256 for MeanFlowSE).
        hop_length:      Total downsampling factor (640 → 25 Hz @ 16 kHz).
        pretrained_path: Path to pretrained WaveVAE checkpoint.
        device:          Device to load the model on.

    Supported hop_lengths and their factorizations:
        640  → (8, 5, 4, 4) = 25.0 Hz @ 16 kHz   (MeanFlowSE paper)
        1280 → (8, 5, 4, 8) = 12.5 Hz @ 16 kHz   (KALL-E paper)
        320  → (8, 5, 4, 2) = 50.0 Hz @ 16 kHz
    """
    HOP_FACTORIZATIONS = {
        320:  (8, 5, 4, 2),
        640:  (8, 5, 4, 4),
        1280: (8, 5, 4, 8),
    }
    if hop_length not in HOP_FACTORIZATIONS:
        raise ValueError(
            f"Unsupported hop_length={hop_length}. "
            f"Supported: {list(HOP_FACTORIZATIONS.keys())}"
        )

    downsample_rates = HOP_FACTORIZATIONS[hop_length]
    wave_vae = WaveVAE(
        latent_dim=latent_dim,
        downsample_rates=downsample_rates,
    )

    if pretrained_path is not None:
        ckpt = torch.load(pretrained_path, map_location=device)
        state = ckpt["model"] if "model" in ckpt else ckpt
        wave_vae.load_state_dict(state)
        print(f"Loaded WaveVAE from {pretrained_path}")

    wave_vae.eval()
    return wave_vae.to(device)
