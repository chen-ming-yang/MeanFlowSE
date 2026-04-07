"""
Inference CLI for MeanFlowSE (dysarthria speech restoration).

Examples:
    python meandsr/infer_dsr.py \
      --ckpt checkpoints_dsr/ckpt_epoch002.pt \
      --input tmp_audio/dysarthria/01/S001T001E000N00000.wav \
      --output dsr_out.wav

    python meandsr/infer_dsr.py \
      --ckpt checkpoints_dsr/ckpt_epoch002.pt \
      --input tmp_audio/dysarthria \
      --output enhanced_dysarthria
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is importable (for mean_flow, dit, etc.)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import soundfile as sf

from meandsr.train_dsr import build_model



def _load_wav_mono(path: Path) -> tuple[torch.Tensor, int]:
    wav, sr = sf.read(str(path), always_2d=True)
    wav_t = torch.from_numpy(wav).float()
    wav_mono = wav_t.mean(dim=1)
    return wav_mono, sr



def _save_wav(path: Path, wav: torch.Tensor, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), wav.detach().cpu().numpy(), sample_rate)



def _iter_input_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() != ".wav":
            raise ValueError(f"Input file must be .wav: {input_path}")
        return [input_path]

    if input_path.is_dir():
        files = sorted(input_path.rglob("*.wav"))
        if not files:
            raise ValueError(f"No .wav files found under: {input_path}")
        return files

    raise ValueError(f"Input path does not exist: {input_path}")



def _resolve_output_path(src: Path, input_root: Path, output_path: Path, is_input_dir: bool) -> Path:
    if not is_input_dir:
        if output_path.suffix.lower() != ".wav":
            raise ValueError("When --input is a file, --output must be a .wav file path.")
        return output_path

    rel = src.relative_to(input_root)
    return output_path / rel



def main() -> None:
    parser = argparse.ArgumentParser(description="Inference for MeanFlowSE DSR")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to DSR checkpoint")
    parser.add_argument("--input", type=str, required=True, help="Input .wav file or directory")
    parser.add_argument("--output", type=str, required=True, help="Output .wav file or directory")
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference")
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt)
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    ckpt = torch.load(ckpt_path, map_location=device)
    cfg_dict = ckpt.get("cfg")
    if cfg_dict is None:
        raise ValueError("Checkpoint is missing 'cfg'.")

    cfg = argparse.Namespace(**cfg_dict)
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    files = _iter_input_files(input_path)
    is_input_dir = input_path.is_dir()

    print(f"Loaded checkpoint: {ckpt_path}")
    print(f"Device: {device}")
    print(f"Processing {len(files)} file(s)...")

    for i, src in enumerate(files, start=1):
        wav, sr = _load_wav_mono(src)
        if sr != cfg.sample_rate:
            raise ValueError(
                f"Sample rate mismatch for {src}: got {sr}, expected {cfg.sample_rate}. "
                "Please resample input first."
            )

        with torch.no_grad():
            enhanced = model.inference(wav.unsqueeze(0).to(device)).squeeze(0)

        dst = _resolve_output_path(src, input_path, output_path, is_input_dir)
        _save_wav(dst, enhanced, sr)

        if i % 10 == 0 or i == len(files):
            print(f"[{i}/{len(files)}] saved: {dst}")

    print("Done.")


if __name__ == "__main__":
    main()
