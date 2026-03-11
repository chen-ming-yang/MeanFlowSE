"""
Inference script for MeanFlowSE.

Usage:
    # Single file
    python inference.py --ckpt checkpoints/ckpt_epoch099.pt --input noisy.wav --output enhanced.wav

    # Directory of files
    python inference.py --ckpt checkpoints/ckpt_epoch099.pt --input noisy_dir/ --output enhanced_dir/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torchaudio
import soundfile as sf

from config import default_config
from mean_flow import MeanFlowSE, SSLEncoder, VAEEncoder, VAEDecoder
from dit import DiTBackbone


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def load_model(ckpt_path: str, device: torch.device) -> tuple[MeanFlowSE, dict]:
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt.get("cfg", vars(default_config.to_namespace()))

    ssl_encoder = SSLEncoder(model_name=cfg["ssl_model"], num_layers=cfg["ssl_layers"])
    vae_encoder = VAEEncoder(latent_dim=cfg["latent_dim"])
    vae_decoder = VAEDecoder(latent_dim=cfg["latent_dim"])
    backbone = DiTBackbone(
        latent_dim=cfg["latent_dim"],
        ssl_dim=cfg["ssl_dim"],
        hidden_dim=cfg["hidden_dim"],
        depth=cfg["depth"],
        heads=cfg["heads"],
        dim_head=cfg["dim_head"],
        ff_mult=cfg["ff_mult"],
        dropout=0.0,  # disable dropout at inference
    )
    model = MeanFlowSE(
        ssl_encoder=ssl_encoder,
        vae_encoder=vae_encoder,
        vae_decoder=vae_decoder,
        dit_backbone=backbone,
        flow_ratio=cfg["flow_ratio"],
        time_mu=cfg["time_mu"],
        time_sigma=cfg["time_sigma"],
        adaptive_gamma=cfg["adaptive_gamma"],
        adaptive_c=cfg["adaptive_c"],
    )
    model.load_state_dict(ckpt["model"])
    model.eval().to(device)
    print(f"Loaded checkpoint from {ckpt_path} (epoch {ckpt.get('epoch', '?')})")
    return model, cfg


# ---------------------------------------------------------------------------
# Audio I/O helpers
# ---------------------------------------------------------------------------

def load_wav(path: Path, target_sr: int, device: torch.device) -> torch.Tensor:
    """Load wav, convert to mono, resample → (1, samples)."""
    wav_data, sr = sf.read(str(path), always_2d=True)  # (samples, channels)
    wav = torch.from_numpy(wav_data.T).float()  # (channels, samples)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        print (f"Resampling {path} from {sr} Hz to {target_sr} Hz")
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav.to(device)  # (1, T)


def save_wav(wav: torch.Tensor, path: Path, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # wav: (1, T) or (T,)
    if wav.ndim == 1:
        wav = wav.unsqueeze(0)
    sf.write(str(path), wav.cpu().numpy().T, sample_rate)


# ---------------------------------------------------------------------------
# Enhance one file
# ---------------------------------------------------------------------------

@torch.no_grad()
def enhance_file(
    model: MeanFlowSE,
    input_path: Path,
    output_path: Path,
    sample_rate: int,
    device: torch.device,
    chunk_len: float | None = None,
) -> None:
    """Enhance a single wav file.

    Args:
        chunk_len: If set (seconds), process audio in overlapping chunks to
                   handle files longer than GPU memory allows.  Set to None
                   to process the whole file at once.
    """
    wav = load_wav(input_path, sample_rate, device)  # (1, T)

    if chunk_len is None:
        enhanced = model.inference(wav)               # (1, T)
    else:
        enhanced = _enhance_chunked(model, wav, sample_rate, chunk_len)

    save_wav(enhanced, output_path, sample_rate)


def _enhance_chunked(
    model: MeanFlowSE,
    wav: torch.Tensor,          # (1, T)
    sample_rate: int,
    chunk_len: float,
    overlap: float = 0.0625,    # 62.5 ms overlap
) -> torch.Tensor:
    """Process long audio in overlapping chunks with linear cross-fade."""
    chunk_samples = int(chunk_len * sample_rate)
    overlap_samples = int(overlap * sample_rate)
    hop = chunk_samples - overlap_samples
    total = wav.shape[-1]

    output = torch.zeros_like(wav)
    weight = torch.zeros(total, device=wav.device)

    # Hann cross-fade window
    fade = torch.hann_window(chunk_samples, device=wav.device)

    pos = 0
    while pos < total:
        end = min(pos + chunk_samples, total)
        chunk = wav[..., pos:end]

        # Pad last chunk to chunk_samples
        pad = chunk_samples - chunk.shape[-1]
        if pad > 0:
            chunk = torch.nn.functional.pad(chunk, (0, pad))

        out_chunk = model.inference(chunk).squeeze(0)  # (T,)
        out_chunk = out_chunk[:chunk_samples - pad] if pad > 0 else out_chunk

        seg_len = end - pos
        output[0, pos:end] += out_chunk[:seg_len] * fade[:seg_len]
        weight[pos:end] += fade[:seg_len]

        pos += hop

    output[0] /= weight.clamp(min=1e-8)
    return output


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    print(f"Using device: {device}")

    model, cfg = load_model(args.ckpt, device)
    sample_rate: int = cfg.get("sample_rate", 16000)

    input_path = Path(args.input)
    output_path = Path(args.output)

    if input_path.is_dir():
        # Batch mode: process all wav files in the directory
        wav_files = sorted(input_path.glob("**/*.wav"))
        print(f"Found {len(wav_files)} wav file(s) in {input_path}")
        for wav_file in wav_files:
            rel = wav_file.relative_to(input_path)
            out_file = output_path / rel
            enhance_file(model, wav_file, out_file, sample_rate, device, args.chunk_len)
            print(f"  {wav_file.name} → {out_file}")
    else:
        # Single file mode
        enhance_file(model, input_path, output_path, sample_rate, device, args.chunk_len)
        print(f"Saved enhanced audio → {output_path}")


def get_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="MeanFlowSE inference")
    p.add_argument("--ckpt", type=str, required=True,
                   help="Path to model checkpoint (.pt)")
    p.add_argument("--input", type=str, required=True,
                   help="Input noisy wav file or directory")
    p.add_argument("--output", type=str, required=True,
                   help="Output enhanced wav file or directory")
    p.add_argument("--chunk_len", type=float, default=None,
                   help="Chunk length in seconds for long-file processing (default: whole file)")
    p.add_argument("--cpu", action="store_true", default=False,
                   help="Force CPU inference even if CUDA is available")
    return p


if __name__ == "__main__":
    main(get_parser().parse_args())
