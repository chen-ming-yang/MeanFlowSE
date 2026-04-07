"""
Quick demo: test DAC encoder/decoder round-trip on a real audio file.

Usage:
    python demo_dac.py --input noisy.wav --output dac_roundtrip.wav
"""

import argparse
import torch
import torchaudio
import soundfile as sf
from codec_vae import build_codec_vae


def main():
    p = argparse.ArgumentParser(description="DAC encode→decode round-trip demo")
    p.add_argument("--input", type=str, required=True, help="Input .wav file")
    p.add_argument("--output", type=str, default="dac_roundtrip.wav",
                   help="Output reconstructed .wav file")
    p.add_argument("--sr", type=int, default=16000, help="Target sample rate")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Build DAC encoder/decoder
    print("Loading DAC model...")
    encoder, decoder = build_codec_vae("dac_16khz", sr=args.sr)
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    print(f"  Latent dim: {encoder.latent_dim}")
    print(f"  Hop length: {encoder._stride} → {args.sr / encoder._stride:.1f} Hz frame rate")

    # Load audio
    wav, sr = sf.read(args.input, always_2d=True)
    wav = torch.from_numpy(wav.T).float()  # (channels, T)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != args.sr:
        wav = torchaudio.functional.resample(wav, sr, args.sr)
    wav = wav.squeeze(0).unsqueeze(0).to(device)  # (1, T)
    print(f"Input: {args.input} ({wav.shape[-1]} samples, {wav.shape[-1]/args.sr:.2f}s)")

    # Encode
    with torch.no_grad():
        z = encoder(wav)
        print(f"Latent: {z.shape} (B, T_latent={z.shape[1]}, D={z.shape[2]})")

        # Decode
        recon = decoder(z)
        print(f"Reconstructed: {recon.shape}")

    # Trim to original length
    orig_len = wav.shape[-1]
    recon = recon[:, :orig_len]

    # Save
    sf.write(args.output, recon.cpu().numpy().T, args.sr)
    print(f"Saved → {args.output}")

    # Compute simple metrics
    wav_np = wav[:, :recon.shape[-1]].cpu()
    recon_np = recon.cpu()
    mse = ((wav_np - recon_np) ** 2).mean().item()
    snr = 10 * torch.log10(wav_np.pow(2).mean() / max(mse, 1e-10)).item()
    print(f"Reconstruction MSE: {mse:.6f}")
    print(f"Reconstruction SNR: {snr:.1f} dB")


if __name__ == "__main__":
    main()
