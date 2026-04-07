from torch import nn
import torch
import torch.nn.functional as F
from dit import DiTBackbone  # noqa: F401  (re-exported for MeanFlowSE users)

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
    

class VAEEncoder(nn.Module):
    def __init__(self, latent_dim: int = 512):
        super().__init__()
        self.latent_dim = latent_dim
        # Placeholder for actual VAE encoder architecture
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 128, kernel_size=7, stride=2, padding=3), nn.GELU(),
            nn.Conv1d(128, 256, kernel_size=7, stride=2, padding=2), nn.GELU(),
            nn.Conv1d(256, 256, kernel_size=7, stride=2, padding=2), nn.GELU(),
            nn.Conv1d(256, latent_dim, kernel_size=7, stride=2, padding=2), nn.GELU(),
            nn.Conv1d(latent_dim, latent_dim, kernel_size=1),
        )

    
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        x = waveform.unsqueeze(1)
        return self.encoder(x).transpose(1, 2)



class VAEDecoder(nn.Module):
    def __init__(self, latent_dim: int = 256):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(latent_dim, 256, kernel_size=8, stride=4, padding=2), nn.GELU(),
            nn.ConvTranspose1d(256, 256, kernel_size=8, stride=4, padding=2), nn.GELU(),
            nn.ConvTranspose1d(256, 128, kernel_size=8, stride=4, padding=2), nn.GELU(),
            nn.ConvTranspose1d(128, 64, kernel_size=8, stride=4, padding=2), nn.GELU(),
            nn.ConvTranspose1d(64, 1, kernel_size=7, stride=1, padding=3), nn.Tanh(), 
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = z.transpose(1, 2)
        return self.decoder(z).squeeze(1)
    



class MeanFlowSE(nn.Module):
    def __init__(
            self,
            ssl_encoder: SSLEncoder,
            vae_encoder: VAEEncoder,
            vae_decoder: VAEDecoder,
            dit_backbone: DiTBackbone,
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

        for name, param in self.ssl_encoder.named_parameters():
            param.requires_grad = False
            
        #Freeze the vae encoder and decoder
        for param in self.vae_encoder.parameters():
            param.requires_grad = False
        for param in self.vae_decoder.parameters():
            param.requires_grad = False
    

    def _align_temporal(self, z_ssl, z_vae):
        T_vae = z_vae.shape[1]
        z_ssl = z_ssl.transpose(1, 2)
        z_ssl = F.interpolate(z_ssl, size=T_vae, mode='linear', align_corners=False)
        return z_ssl.transpose(1, 2)
    
    def _sample_time_steps(self, batch_size, device):
        time_steps = torch.randn(batch_size, device=device) * self.time_sigma + self.time_mu
        t = torch.sigmoid(time_steps).clamp(1e-5, 1.0)
        r = (self.flow_ratio * t).clamp(min=0.0)
        return r, t
    
    def _interpolate_latent(self, z_x, epsilon, t):
        # z_t = (1 - t) * epsilon + t * z_x
        t_exp = t.view(-1, 1, 1)
        return (1 - t_exp) * epsilon + t_exp * z_x
    
    def _target_average_velocity(self, z_x, epsilon):
        # u_target = (z_x - epsilon) / self.time_sigma
        # self.time_sigma = 1
        return z_x - epsilon
    
    def _adaptive_l2_loss(self, u_hat, u_target, mask=None):
        # w = (delta^2 + c)^{-(1-gamma)}, L = E[w * delta^2]
        # mask: (B, T) boolean mask where True = valid position
        delta_sq_full = (u_hat - u_target).pow(2).sum(dim=-1)  # (B, T)
        
        if mask is not None:
            # Apply mask: only compute loss over valid (non-padded) positions
            delta_sq_full = delta_sq_full * mask  # zero out padded positions
            valid_counts = mask.sum(dim=1).clamp(min=1)  # per-sample valid token count
            delta_sq = delta_sq_full.sum(dim=1) / valid_counts  # mean over valid positions
        else:
            delta_sq = delta_sq_full.mean(dim=1)  # mean over all positions
        
        w = (delta_sq.detach() + self.adaptive_c) ** (-(1 - self.adaptive_gamma))
        loss = (w * delta_sq).mean()
        return loss, {
            "loss": loss.item(),
            "delta_sq": delta_sq.mean().item(),
            "mean_weight": w.mean().item(),
        }
    
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

        # Create latent-space mask from waveform lengths
        # Downsample factor depends on VAE: default=16, WaveVAE=hop_length
        mask = None
        if lengths is not None:
            hop = getattr(self.vae_encoder, 'hop_length',
                          getattr(self.vae_encoder, '_stride', 16))
            latent_lengths = (lengths / hop).long()
            T = z_x.shape[1]
            mask = torch.arange(T, device=device).unsqueeze(0) < latent_lengths.unsqueeze(1)  # (B, T)
            mask = mask.float()  # convert to float for multiplication

        return self._adaptive_l2_loss(u_hat, u_target, mask=mask)
    
    @torch.no_grad()
    def inference(self, noisy_waveform, debug=True):
        B = noisy_waveform.shape[0]
        T_audio = noisy_waveform.shape[1]
        device = noisy_waveform.device

        def _stat(name, x):
            if debug:
                print(f"[DEBUG] {name}: shape={list(x.shape)}, "
                      f"min={x.min().item():.4f}, max={x.max().item():.4f}, "
                      f"mean={x.mean().item():.4f}, std={x.std().item():.4f}")

        if debug:
            print(f"[DEBUG] noisy_waveform: shape={list(noisy_waveform.shape)}, device={device}")
        _stat("noisy_waveform", noisy_waveform)

        z_y_ssl = self.ssl_encoder(noisy_waveform)
        _stat("z_y_ssl (SSL features)", z_y_ssl)

        # Encode noisy audio to get VAE shape reference
        z_ref = self.vae_encoder(noisy_waveform)
        _stat("z_ref (VAE-encoded noisy, for shape)", z_ref)

        # Align SSL features to VAE temporal resolution
        T_vae = z_ref.shape[1]
        z_y = z_y_ssl.transpose(1, 2)
        z_y = F.interpolate(z_y, size=T_vae, mode='linear', align_corners=False)
        z_y = z_y.transpose(1, 2)  # (B, T_vae, ssl_dim)
        _stat("z_y (aligned SSL)", z_y)

        # Sample random noise epsilon with same shape as VAE latent
        epsilon = torch.randn_like(z_ref)
        _stat("epsilon (random noise)", epsilon)

        # One-step inference: z0 = epsilon - u(epsilon, 0, 1)
        r = torch.zeros(B, device=device)
        t = torch.ones(B, device=device)
        if debug:
            print(f"[DEBUG] r={r.tolist()}, t={t.tolist()}")
        u_hat = self.dit_backbone(epsilon, z_y, r, t)
        _stat("u_hat (predicted velocity)", u_hat)

        z0 = epsilon - u_hat
        _stat("z0 (denoised latent)", z0)

        wav = self.vae_decoder(z0)
        _stat("output wav", wav)

        if debug:
            # Also decode z_ref directly to see if VAE roundtrip works
            wav_roundtrip = self.vae_decoder(z_ref)
            _stat("wav_roundtrip (decode z_ref directly)", wav_roundtrip)

        return wav[:, :T_audio]
    



