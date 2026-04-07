import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from mean_flow import MeanFlowSE, SSLEncoder, VAEEncoder, VAEDecoder
from dit import DiTBackbone
from modules import (
    AdaLayerNorm,
    ConvNeXtV2Block,
    ConvPositionEmbedding,
    DiTBlock,
    TimestepEmbedding,
    precompute_freqs_cis,
)


class DiTBlockWithCrossAttention(nn.Module):
    def __init__(self, hidden_dim: int, heads: int, ff_dim: int, dropout: float):
        super().__init__()

        self.adaln_self_attn = AdaLayerNorm(hidden_dim)
        self.self_attn = nn.MultiheadAttention(
            hidden_dim, heads, dropout=dropout, batch_first=True)
        
        self.adaln_cross_attn = AdaLayerNorm(hidden_dim)
        self.cross_attn_norm_kv = nn.LayerNorm(hidden_dim)
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, heads, dropout=dropout, batch_first=True)
        
        self.adaln_ff = AdaLayerNorm(hidden_dim)
        self.feadforward = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_dim),
            nn.Dropout(dropout),
        )



    def forward(self, x, cond_seq, time_cond):
        # Self-attention
        h, gate_self, _, _, _ = self.adaln_self_attn(x, time_cond)
        attn_out, _ = self.self_attn(h, h, h, need_weights=False)
        x = x + gate_self.unsqueeze(1) * attn_out

        # Cross-attention to bottleneck
        h, gate_cross, _, _, _ = self.adaln_cross_attn(x, time_cond)
        kv = self.cross_attn_norm_kv(cond_seq)
        cross_out, _ = self.cross_attn(h, kv, kv, need_weights=False)
        x = x + gate_cross.unsqueeze(1) * cross_out

        # FFN
        h, _, _, _, gate_mlp = self.adaln_ff(x, time_cond)
        x = x + gate_mlp.unsqueeze(1) * self.feadforward(h)
        return x



class DirectMappingDiTBackbone(nn.Module):
    """
    DiT backbone using cross-attention to Perceiver bottleneck
    """

    def __init__(
        self,
        latent_dim: int = 256,
        bottleneck_dim: int = 512,
        hidden_dim: int = 512,
        num_layers: int = 8,
        num_heads: int = 8,
        ff_dim: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        self.input_proj = nn.Linear(latent_dim, hidden_dim)
        self.cond_proj = nn.Linear(bottleneck_dim, hidden_dim)
        self.pos_enc = SinusPositionEncoding(hidden_dim, max_len=16000)
        
        self.time_embed_r = TimestepEmbedding(hidden_dim)
        self.time_embed_t = TimestepEmbedding(hidden_dim)

        self.dit_blocks = nn.ModuleList([
            DiTBlockWithCrossAttention(hidden_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])

        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, latent_dim)
        self.gradient_checkpointing = True

    def forward(self, z_t, bottleneck, r, t):
        x = self.input_proj(z_t)
        x = self.pos_enc(x)

        cond = self.cond_proj(bottleneck)

        time_cond = self.time_embed_t(t) + self.time_embed_r(r)

        for block in self.dit_blocks:
            if self.gradient_checkpointing and self.training:
                x = checkpoint(block, x, cond, time_cond, use_reentrant=False)
            else:
                x = block(x, cond, time_cond)
        
        return self.output_proj(self.output_norm(x))


class SinusPositionEncoding(nn.Module):
    def __init__(self, dim, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]



class PerceiverBottleneck(nn.Module):
    """
    dysarthria audio into fixed-size bottleneck representation
    (B, K, D)
    """

    def __init__(
      self,
      input_dim: int = 1024,
      output_dim: int = 512,
      num_latents_tokens: int = 64,
      num_heads: int = 8,
      num_layers: int = 2,
      dropout: float = 0.1,      
    ):
        super().__init__()
        self.latent_queries = nn.Parameter(torch.randn(1, num_latents_tokens, output_dim) * 0.02)

        self.input_proj = nn.Linear(input_dim, output_dim)

        self.input_pos_enc = SinusPositionEncoding(output_dim, max_len=2000)
        self.query_pos_enc = SinusPositionEncoding(output_dim, max_len=num_latents_tokens + 10)

        self.cross_attn_layers = nn.ModuleList()
        self.cross_norms_q = nn.ModuleList()
        self.cross_norms_kv = nn.ModuleList()
        self.ff_layers = nn.ModuleList()
        self.ff_norms = nn.ModuleList()

        for _ in range(num_layers):
            self.cross_attn_layers.append(nn.MultiheadAttention(output_dim, num_heads, dropout=dropout, batch_first=True))
            self.cross_norms_q.append(nn.LayerNorm(output_dim))
            self.cross_norms_kv.append(nn.LayerNorm(output_dim))
            self.ff_layers.append(nn.Sequential(
                nn.Linear(output_dim, output_dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(output_dim * 4, output_dim),
                nn.Dropout(dropout),
            ))
            self.ff_norms.append(nn.LayerNorm(output_dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = self.input_proj(x)
        x = self.input_pos_enc(x)

        latents = self.latent_queries.expand(B, -1, -1)
        latents = self.query_pos_enc(latents)

        for cross_attn, norm_q, norm_kv, ff, norm_ff in zip(
            self.cross_attn_layers, self.cross_norms_q, self.cross_norms_kv, self.ff_layers, self.ff_norms
        ):
            q = norm_q(latents)
            k = norm_kv(x)
            v = norm_kv(x)
            attn_out, _ = cross_attn(q, k, v)
            latents = latents + attn_out

            ff_out = ff(norm_ff(latents))
            latents = latents + ff_out
        
        return latents

class LengthPredictor(nn.Module):
    """
    Predict target length (number of tokens) from bottleneck representation
    """

    def __init__(self, input_dim: int = 512, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus(),  # ensure positive output
        )

    def forward(self, bottleneck: torch.Tensor) -> torch.Tensor:
        # bottleneck: (B, K, D)
        pooled = bottleneck.mean(dim=1)  # (B, D)
        length_logit = self.net(pooled).squeeze(-1)  # (B,)
        return length_logit


class OneToManyDysarthriaSE(nn.Module):
    def __init__(
        self,
        ssl_encoder: SSLEncoder,
        vae_encoder: VAEEncoder,
        vae_decoder: VAEDecoder,
        dit_backbone: DirectMappingDiTBackbone,
        perceiver: PerceiverBottleneck,
        length_predictor: LengthPredictor,
        flow_ratio: float = 0.25,
        time_mu: float = -0.4,
        time_sigma: float = 1.0,
        adaptive_gamma: float = 0.5,
        adaptive_c: float = 1e-3,
    ):
        super().__init__()

        # Store hyperparameters
        self.flow_ratio = flow_ratio
        self.time_mu = time_mu
        self.time_sigma = time_sigma
        self.adaptive_gamma = adaptive_gamma
        self.adaptive_c = adaptive_c
        self.ssl_encoder = ssl_encoder
        self.vae_encoder = vae_encoder
        self.vae_decoder = vae_decoder
        self.dit_backbone = dit_backbone
        self.perceiver = perceiver
        self.length_predictor = length_predictor


        # Freeze SSL encoder
        for param in self.ssl_encoder.parameters():
            param.requires_grad = False

        # Freeze VAE encoder and decoder
        for param in self.vae_encoder.parameters():
            param.requires_grad = False
        for param in self.vae_decoder.parameters():
            param.requires_grad = False

    def _sample_time_steps(self, batch_size, device):
        """Sample time steps for flow matching training.
        
        Returns:
            r: flow ratio (flow_ratio * t), clamped to [0, inf)
            t: sigmoid-transformed time steps in [1e-5, 1.0]
        """
        time_steps = torch.randn(batch_size, device=device) * self.time_sigma + self.time_mu
        t = torch.sigmoid(time_steps).clamp(1e-5, 1.0)
        r = (self.flow_ratio * t).clamp(min=0.0)
        return r, t

    def _masked_adaptive_l2_loss(self, u_hat, u_target, mask=None):
        """Compute adaptive L2 loss with optional masking for padded sequences.
        
        Args:
            u_hat: predicted velocity field (B, T, D)
            u_target: target velocity field (B, T, D)
            mask: optional boolean/float mask (B, T) where True/1.0 = valid position
            
        Returns:
            loss: scalar loss value
            stats: dict with loss statistics
        """
        # Compute in fp32 for numerical stability under autocast.
        u_hat_f = u_hat.float()
        u_target_f = u_target.float()
        delta_sq_full = (u_hat_f - u_target_f).pow(2).sum(dim=-1)
        
        if mask is not None:
            # Apply mask: zero out padded positions
            delta_sq_full = delta_sq_full * mask.float()
            valid_counts = mask.sum(dim=1).clamp(min=1)  # per-sample valid token count
            delta_sq = delta_sq_full.sum(dim=1) / valid_counts  # mean over valid positions
        else:
            delta_sq = delta_sq_full.mean(dim=1)  # mean over all positions

        # Prevent inf*0 -> NaN in adaptive reweighting.
        delta_sq = torch.nan_to_num(delta_sq, nan=1e6, posinf=1e6, neginf=0.0)
        
        # Adaptive weighting: w = (delta^2 + c)^{-(1-gamma)}
        w = (delta_sq.detach() + self.adaptive_c) ** (-(1 - self.adaptive_gamma))
        loss = (w * delta_sq).mean()
        loss = torch.nan_to_num(loss, nan=1e6, posinf=1e6, neginf=0.0)
        
        return loss, {
            "loss": loss.item(),
            "delta_sq": delta_sq.mean().item(),
            "mean_weight": w.mean().item(),
        }

    def forward_train(self, batch):
        dys_wav = batch["dys_wav"]
        clean_wav = batch["clean_wav"]
        clean_lengths = batch.get("clean_lengths", None)  # optional, for variable-length training

        B = dys_wav.shape[0]
        device = dys_wav.device

        ssl_feats = self.ssl_encoder(dys_wav)
        with torch.no_grad():
            vae_feats = self.vae_encoder(clean_wav)
        
        bottleneck = self.perceiver(ssl_feats)

        z_normal = vae_feats
        T_normal = z_normal.shape[1]
        
        # Create mask from lengths: 1.0 = valid, 0.0 = padding
        # VAE encoder downsamples by 16x
        if clean_lengths is not None:
            latent_lengths = (clean_lengths.to(device) / 16).long()
            vae_mask = torch.arange(T_normal, device=device).unsqueeze(0) < latent_lengths.unsqueeze(1)
            vae_mask = vae_mask.float()  # (B, T) with 1.0 for valid, 0.0 for padding
            vae_lengths = latent_lengths
        else:
            # No length info: assume all positions are valid
            vae_mask = torch.ones(B, T_normal, device=device).float()
            vae_lengths = torch.full((B,), T_normal, dtype=torch.long, device=device)
        
        # Generate noise and interpolate
        epsilon = torch.randn_like(z_normal)
        r, t = self._sample_time_steps(B, device)
        t_exp = t.view(-1, 1, 1)
        z_t = (1 - t_exp) * epsilon + t_exp * z_normal

        u_hat = self.dit_backbone(z_t, bottleneck, r, t)
        u_target = z_normal - epsilon

        flow_loss, flow_stats = self._masked_adaptive_l2_loss(u_hat, u_target, mask=vae_mask)

        total_loss = flow_loss

        metrics = {
            "total_loss": total_loss.item(),
            "flow_loss": flow_loss.item(),
            "bottleneck_norm": bottleneck.norm(dim=-1).mean().item(),
            "z_normal_norm": z_normal.norm(dim=-1).mean().item(),
            "z_t_norm": z_t.norm(dim=-1).mean().item(),
            "u_hat_norm": u_hat.norm(dim=-1).mean().item(),
            "u_target_norm": u_target.norm(dim=-1).mean().item(),
        }
        return total_loss, metrics

    @torch.no_grad()
    def inference(self, dys_wav):

        B = dys_wav.shape[0]
        device = dys_wav.device

        z_dys = self.ssl_encoder(dys_wav)
        bottleneck = self.perceiver(z_dys)

        pred_length = self.length_predictor(bottleneck)
        T_normal = max(pred_length.round().long().max().item(), 1)  # ensure at least some tokens

        epsilon = torch.randn(B, T_normal, self.vae_encoder.latent_dim, device=device)
        r = torch.zeros(B, device=device)
        t = torch.ones(B, device=device)

        u_hat = self.dit_backbone(epsilon, bottleneck, r, t)
        z0 = epsilon - u_hat

        return self.vae_decoder(z0)



    


