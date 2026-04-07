"""
MeanFlowSE for Dysarthria Speech Restoration (DSR).

Self-contained copy of MeanFlowSE adapted for DSR:
  - Dynamic downsample-factor detection (works with any VAE / codec)
  - Inference aligns SSL features to codec latent resolution
  - Does NOT modify the original mean_flow.py used by train.py
"""

from torch import nn
import torch
import torch.nn.functional as F
from dit import DiTBackbone  # noqa: F401


class SSLEncoder(nn.Module):

    def __init__(self, model_name: str = "microsoft/wavlm-large", num_layers: int = 25):
        super().__init__()
        from transformers import WavLMModel
        self.ssl_model = WavLMModel.from_pretrained(model_name)
        for param in self.ssl_model.parameters():
            param.requires_grad = False

        self.num_layers = num_layers
        self.layer_weights = nn.Parameter(torch.zeros(num_layers))

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            outputs = self.ssl_model(waveform, output_hidden_states=True)
        hidden_states = torch.stack(outputs.hidden_states, dim=0)
        weights = F.softmax(self.layer_weights, dim=0).view(-1, 1, 1, 1)
        weighted_sum = (weights * hidden_states).sum(dim=0)
        return weighted_sum


class MeanFlowDSR(nn.Module):
    """MeanFlowSE variant for dysarthria speech restoration.

    Uses a pretrained audio codec (e.g. EnCodec) as the frozen VAE, with
    dynamic temporal alignment between SSL and codec latent spaces.
    """

    def __init__(
        self,
        ssl_encoder,
        vae_encoder,
        vae_decoder,
        dit_backbone,
        flow_ratio: float = 0.25,
        time_mu: float = -0.4,
        time_sigma: float = 1.0,
        adaptive_gamma: float = 0.5,
        adaptive_c: float = 1e-3,
    ):
        super().__init__()
        self.ssl_encoder = ssl_encoder
        self.vae_encoder = vae_encoder
        self.vae_decoder = vae_decoder
        self.dit_backbone = dit_backbone

        self.flow_ratio = flow_ratio
        self.time_mu = time_mu
        self.time_sigma = time_sigma
        self.adaptive_gamma = adaptive_gamma
        self.adaptive_c = adaptive_c

        # Freeze SSL encoder
        for param in self.ssl_encoder.parameters():
            param.requires_grad = False

        # Freeze VAE encoder and decoder
        for param in self.vae_encoder.parameters():
            param.requires_grad = False
        for param in self.vae_decoder.parameters():
            param.requires_grad = False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _align_temporal(self, z_ssl, z_vae):
        T_vae = z_vae.shape[1]
        z_ssl = z_ssl.transpose(1, 2)
        z_ssl = F.interpolate(z_ssl, size=T_vae, mode="linear", align_corners=False)
        return z_ssl.transpose(1, 2)

    def _sample_time_steps(self, batch_size, device):
        time_steps = torch.randn(batch_size, device=device) * self.time_sigma + self.time_mu
        t = torch.sigmoid(time_steps).clamp(1e-5, 1.0)
        r = (self.flow_ratio * t).clamp(min=0.0)
        return r, t

    def _interpolate_latent(self, z_x, epsilon, t):
        t_exp = t.view(-1, 1, 1)
        return (1 - t_exp) * epsilon + t_exp * z_x

    def _target_average_velocity(self, z_x, epsilon):
        return z_x - epsilon

    def _adaptive_l2_loss(self, u_hat, u_target, mask=None):
        u_hat_f = u_hat.float()
        u_target_f = u_target.float()
        delta_sq_full = (u_hat_f - u_target_f).pow(2).sum(dim=-1)  # (B, T)

        if mask is not None:
            delta_sq_full = delta_sq_full * mask
            valid_counts = mask.sum(dim=1).clamp(min=1)
            delta_sq = delta_sq_full.sum(dim=1) / valid_counts
        else:
            delta_sq = delta_sq_full.mean(dim=1)

        delta_sq = torch.nan_to_num(delta_sq, nan=1e6, posinf=1e6, neginf=0.0)
        w = (delta_sq.detach() + self.adaptive_c) ** (-(1 - self.adaptive_gamma))
        loss = (w * delta_sq).mean()
        loss = torch.nan_to_num(loss, nan=1e6, posinf=1e6, neginf=0.0)

        return loss, {
            "loss": loss.item(),
            "delta_sq": delta_sq.mean().item(),
            "mean_weight": w.mean().item(),
        }

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def forward_train(self, noisy_wav, clean_wav, lengths=None):
        B = noisy_wav.shape[0]
        device = noisy_wav.device

        z_y_ssl = self.ssl_encoder(noisy_wav)

        with torch.no_grad():
            z_x = self.vae_encoder(clean_wav)

        z_y = self._align_temporal(z_y_ssl, z_x)

        epsilon = torch.randn_like(z_x)
        r, t = self._sample_time_steps(B, device)
        z_t = self._interpolate_latent(z_x, epsilon, t)

        u_hat = self.dit_backbone(z_t, z_y, r, t)
        u_target = self._target_average_velocity(z_x, epsilon)

        # Create latent-space mask — derive downsample factor dynamically
        mask = None
        if lengths is not None:
            ds = clean_wav.shape[-1] / z_x.shape[1]
            latent_lengths = (lengths / ds).long()
            T = z_x.shape[1]
            mask = torch.arange(T, device=device).unsqueeze(0) < latent_lengths.unsqueeze(1)
            mask = mask.float()

        return self._adaptive_l2_loss(u_hat, u_target, mask=mask)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def inference(self, noisy_waveform):
        B = noisy_waveform.shape[0]
        device = noisy_waveform.device

        z_y_ssl = self.ssl_encoder(noisy_waveform)

        # Determine target latent sequence length
        if hasattr(self.vae_encoder, "compute_latent_length"):
            T_latent = self.vae_encoder.compute_latent_length(noisy_waveform.shape[-1])
        else:
            T_latent = z_y_ssl.shape[1]

        # Align SSL features to latent temporal resolution
        z_y = z_y_ssl.transpose(1, 2)
        z_y = F.interpolate(z_y, size=T_latent, mode="linear", align_corners=False)
        z_y = z_y.transpose(1, 2)

        epsilon = torch.randn(B, T_latent, self.vae_encoder.latent_dim, device=device)
        r = torch.zeros(B, device=device)
        t = torch.ones(B, device=device)

        u_hat = self.dit_backbone(epsilon, z_y, r, t)
        z0 = epsilon - u_hat

        return self.vae_decoder(z0)
