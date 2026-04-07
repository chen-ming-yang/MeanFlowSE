"""
Pretrained audio codec as frozen VAE for MeanFlowSE.

Drop-in replacements for VAEEncoder and VAEDecoder that provide a meaningful
latent space for flow matching. Flow matching operates in the continuous
pre-quantization latent space (bypassing the RVQ).

Supported codecs:
    - EnCodec:  build_codec_vae("facebook/encodec_24khz", sr=16000)
    - DAC:      build_codec_vae("dac_16khz", sr=16000)

Usage:
    encoder, decoder = build_codec_vae("dac_16khz", sr=16000)
    model = MeanFlowSE(ssl_encoder, encoder, decoder, dit_backbone, ...)
"""

import math

import torch
import torch.nn as nn
import torchaudio.functional as AF


class CodecEncoder(nn.Module):
    """EnCodec encoder: waveform → continuous pre-VQ latent."""

    def __init__(self, codec_model, input_sr: int = 16000):
        super().__init__()
        self.encoder = codec_model.encoder
        self.codec_sr = codec_model.config.sampling_rate
        self.input_sr = input_sr
        self.latent_dim = codec_model.config.hidden_size  # 128 for encodec_24khz

        self._stride = 1
        for r in codec_model.config.upsampling_ratios:
            self._stride *= r  # 320 for encodec_24khz

        for p in self.parameters():
            p.requires_grad = False

    def compute_latent_length(self, num_input_samples: int) -> int:
        """Return latent sequence length for a given number of input samples."""
        n = num_input_samples
        if self.input_sr != self.codec_sr:
            n = round(n * self.codec_sr / self.input_sr)
        return n // self._stride

    @torch.no_grad()
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform: (B, T) at input_sr
        Returns:
            z: (B, T_latent, latent_dim)
        """
        if self.input_sr != self.codec_sr:
            waveform = AF.resample(waveform, self.input_sr, self.codec_sr)
        x = waveform.unsqueeze(1)       # (B, 1, T)
        z = self.encoder(x)             # (B, latent_dim, T_latent)
        return z.transpose(1, 2)        # (B, T_latent, latent_dim)


class CodecDecoder(nn.Module):
    """EnCodec decoder: continuous latent → waveform."""

    def __init__(self, codec_model, output_sr: int = 16000):
        super().__init__()
        self.decoder = codec_model.decoder
        self.codec_sr = codec_model.config.sampling_rate
        self.output_sr = output_sr
        self.latent_dim = codec_model.config.hidden_size

        for p in self.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (B, T_latent, latent_dim)
        Returns:
            waveform: (B, T) at output_sr
        """
        z = z.transpose(1, 2)           # (B, latent_dim, T_latent)
        wav = self.decoder(z)           # (B, 1, T) at codec_sr
        wav = wav.squeeze(1)            # (B, T)
        if self.output_sr != self.codec_sr:
            wav = AF.resample(wav, self.codec_sr, self.output_sr)
        return wav


def build_codec_vae(
    model_name: str = "facebook/encodec_24khz",
    sr: int = 16000,
):
    """Load pretrained codec once and return (CodecEncoder, CodecDecoder).

    Supports:
        - EnCodec:  model_name = "facebook/encodec_24khz" (or any HF EnCodec id)
        - DAC:      model_name = "dac_16khz", "dac_24khz", or "dac_44khz"
    """
    if model_name.startswith("dac_"):
        return _build_dac_vae(model_type=model_name[4:], sr=sr)

    from transformers import EncodecModel
    codec = EncodecModel.from_pretrained(model_name)
    encoder = CodecEncoder(codec, input_sr=sr)
    decoder = CodecDecoder(codec, output_sr=sr)
    return encoder, decoder


# -------------------------------------------------------------------------
# Descript Audio Codec (DAC)
# -------------------------------------------------------------------------

class DACEncoder(nn.Module):
    """DAC encoder: waveform → continuous pre-VQ latent."""

    def __init__(self, dac_model, input_sr: int = 16000, latent_scale: float | None = None):
        super().__init__()
        self.encoder = dac_model.encoder
        self.codec_sr = dac_model.sample_rate
        self.input_sr = input_sr
        self.latent_dim = dac_model.latent_dim
        self._stride = dac_model.hop_length
        self.hop_length = self._stride  # alias for mask computation in forward_train

        # Latent normalization: scale DAC latents to ~unit variance so that
        # flow matching (epsilon ~ N(0,1)) has a balanced signal-to-noise ratio
        # and the adaptive loss weight w stays in a healthy range.
        # Default of 2.5 calibrated from DAC 16kHz on real speech data.
        if latent_scale is None:
            latent_scale = 2.5
        self.register_buffer("latent_scale", torch.tensor(latent_scale))

        for p in self.parameters():
            p.requires_grad = False

    def compute_latent_length(self, num_input_samples: int) -> int:
        """Return latent sequence length for a given number of input samples."""
        n = num_input_samples
        if self.input_sr != self.codec_sr:
            n = round(n * self.codec_sr / self.input_sr)
        return math.ceil(n / self._stride)

    @torch.no_grad()
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform: (B, T) at input_sr
        Returns:
            z: (B, T_latent, latent_dim)  — normalized to ~unit variance
        """
        if self.input_sr != self.codec_sr:
            waveform = AF.resample(waveform, self.input_sr, self.codec_sr)
        # Pad to multiple of hop_length (matches DAC.preprocess)
        length = waveform.shape[-1]
        right_pad = math.ceil(length / self._stride) * self._stride - length
        if right_pad > 0:
            waveform = torch.nn.functional.pad(waveform, (0, right_pad))
        x = waveform.unsqueeze(1)       # (B, 1, T)
        z = self.encoder(x)             # (B, latent_dim, T_latent)
        z = z.transpose(1, 2)           # (B, T_latent, latent_dim)
        return z / self.latent_scale    # normalize to ~unit variance


class DACDecoder(nn.Module):
    """DAC decoder: continuous latent → waveform."""

    def __init__(self, dac_model, output_sr: int = 16000, latent_scale: float = 1.0):
        super().__init__()
        self.decoder = dac_model.decoder
        self.codec_sr = dac_model.sample_rate
        self.output_sr = output_sr
        self.latent_dim = dac_model.latent_dim
        self.register_buffer("latent_scale", torch.tensor(latent_scale))

        for p in self.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (B, T_latent, latent_dim)  — normalized (from DACEncoder)
        Returns:
            waveform: (B, T) at output_sr
        """
        z = z * self.latent_scale       # undo normalization
        z = z.transpose(1, 2)           # (B, latent_dim, T_latent)
        wav = self.decoder(z)           # (B, 1, T) at codec_sr
        wav = wav.squeeze(1)            # (B, T)
        if self.output_sr != self.codec_sr:
            wav = AF.resample(wav, self.codec_sr, self.output_sr)
        return wav


def _build_dac_vae(model_type: str = "16khz", sr: int = 16000, latent_scale: float | None = None):
    """Load pretrained DAC model and return (DACEncoder, DACDecoder)."""
    import dac as dac_lib

    model_path = dac_lib.utils.download(model_type=model_type)
    dac_model = dac_lib.DAC.load(model_path)
    dac_model.eval()
    encoder = DACEncoder(dac_model, input_sr=sr, latent_scale=latent_scale)
    latent_scale = float(encoder.latent_scale.item())
    decoder = DACDecoder(dac_model, output_sr=sr, latent_scale=latent_scale)
    return encoder, decoder
