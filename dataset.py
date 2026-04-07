"""
Dataset and batching utilities for MeanFlowSE training.
"""

from __future__ import annotations

import re
import random
from pathlib import Path

from tqdm import tqdm

import torch
import torch.nn.functional as F
import torchaudio
import soundfile as sf
from torch.utils.data import Dataset, Sampler


class AudioAugment:
    """Lightweight waveform-level augmentations applied to the noisy mixture."""

    def __init__(self, sample_rate: int = 16000, rir_dir: str | None = None):
        self.sample_rate = sample_rate
        self.rir_files: list[Path] = []

    def _random_gain(self, wav: torch.Tensor, low: float = 0.7, high: float = 1.0) -> torch.Tensor:
        return wav * random.uniform(low, high)

    def _random_speed(self, wav: torch.Tensor) -> torch.Tensor:
        """Sox-style speed perturbation ±5 % via resampling."""
        factor = random.uniform(0.95, 1.05)
        orig_len = wav.shape[-1]
        resampled = torchaudio.functional.resample(
            wav,
            orig_freq=self.sample_rate,
            new_freq=int(self.sample_rate * factor),
        )
        if resampled.shape[-1] > orig_len:
            resampled = resampled[..., :orig_len]
        else:
            resampled = F.pad(resampled, (0, orig_len - resampled.shape[-1]))
        return resampled

    def _random_reverb(self, wav: torch.Tensor) -> torch.Tensor:
        """RIR/reverb is disabled; keep waveform unchanged."""
        return wav

    def __call__(self, noisy: torch.Tensor) -> torch.Tensor:
        if random.random() < 0.5:
            noisy = self._random_gain(noisy)
        # if random.random() < 0.15:
        #     noisy = self._random_speed(noisy)
        # RIR/reverb is disabled: construct noisy audio using only clean + noise.
        # if random.random() < 0.15:
        #     noisy = self._random_reverb(noisy)
        return noisy


class DNSDataset(Dataset):
    """DNS Challenge dataset loader with optional on-the-fly mixing."""

    def __init__(
        self,
        dns_root: str,
        noise_dir: str,
        sample_rate: int = 16000,
        clip_len: float = 6.0,
        segment_len: float | None = None,
        use_all_segments: bool = False,
        mix_on_the_fly: bool = True,
        snr_low: float = -5.0,
        snr_high: float = 20.0,
        augment: bool = True,
        rir_dir: str | None = None,
        dns_layout: str = "default",
    ):
        self.dns_root = Path(dns_root)
        self.noise_dir = Path(noise_dir)
        self.sample_rate = sample_rate
        self.max_clip_samples = int(clip_len * sample_rate)
        self.segment_samples = int(segment_len * sample_rate) if segment_len is not None else None
        self.use_all_segments = use_all_segments and self.segment_samples is not None
        self.mix_on_the_fly = mix_on_the_fly
        self.snr_low = snr_low
        self.snr_high = snr_high
        self.augment = augment
        self.aug = AudioAugment(sample_rate, rir_dir=rir_dir)

        if not self.dns_root.is_dir():
            raise FileNotFoundError(
                f"dns_root does not exist: {self.dns_root.resolve()}\n"
                "  Did you forget a leading '/' in the path?"
            )

        _fileid_re = re.compile(r"fileid_(\d+)", re.IGNORECASE)

        if dns_layout == "paired_dir":
            # Layout: <dns_root>/clean/clean_fileid_N.wav
            #         <dns_root>/noisy/<anything>_fileid_N.wav
            # Pair by the numeric file-id that appears at the end of the stem.
            clean_dir = self.dns_root / "clean"
            noisy_dir = self.dns_root / "noisy"
            assert clean_dir.is_dir(), f"Expected {clean_dir} to exist for paired_dir layout"
            assert noisy_dir.is_dir(), f"Expected {noisy_dir} to exist for paired_dir layout"

            # Build id -> path maps
            clean_by_id: dict[str, Path] = {}
            for p in sorted(clean_dir.glob("*.wav")):
                m = _fileid_re.search(p.stem)
                if m:
                    clean_by_id[m.group(1)] = p

            noisy_by_id: dict[str, Path] = {}
            for p in sorted(noisy_dir.glob("*.wav")):
                m = _fileid_re.search(p.stem)
                if m:
                    noisy_by_id[m.group(1)] = p

            common_ids = sorted(clean_by_id.keys() & noisy_by_id.keys(), key=lambda x: int(x))
            assert len(common_ids) > 0, (
                f"No matching file IDs found between {clean_dir} and {noisy_dir}.\n"
                f"  Clean IDs (sample): {list(clean_by_id.keys())[:5]}\n"
                f"  Noisy IDs (sample): {list(noisy_by_id.keys())[:5]}"
            )

            self.clean_files = [clean_by_id[fid] for fid in common_ids]
            self.noisy_files = [noisy_by_id[fid] for fid in common_ids]
            self.noise_files = []   # not used in paired_dir mode
            self.mix_on_the_fly = False   # paired_dir always uses pre-mixed noisy audio

            n_clean_only = len(clean_by_id) - len(common_ids)
            n_noisy_only = len(noisy_by_id) - len(common_ids)
            print(
                f"DNSDataset [paired_dir]: {len(common_ids)} paired files "
                f"(clean-only: {n_clean_only}, noisy-only: {n_noisy_only}) "
                f"from {self.dns_root}"
            )
        else:
            # Original layout: <dns_root>/datasets.clean.*/**/*.wav
            clean_subdirs = sorted(self.dns_root.glob("datasets.clean.*"))
            self.clean_files = sorted(self.dns_root.glob("datasets.clean.*/**/*.wav"))
            for directory in clean_subdirs:
                self.clean_files = sorted(set(self.clean_files) | set(directory.glob("*.wav")))
            self.clean_files = sorted(self.clean_files)
            assert len(self.clean_files) > 0, (
                f"No .wav files found under {self.dns_root.resolve()}/datasets.clean.*/\n"
                f"  Found subdirs: {[d.name for d in clean_subdirs] or 'none'}"
            )
            print(f"DNSDataset: loaded {len(self.clean_files)} clean files from {self.dns_root}/datasets.clean.*/")
            self.noise_files = sorted(self.noise_dir.glob("**/*.wav"))
            assert len(self.noise_files) > 0, f"No .wav files found recursively under {self.noise_dir}"
            print(f"DNSDataset: loaded {len(self.noise_files)} noise files from {self.noise_dir}/")

            if not mix_on_the_fly:
                self.noisy_files = sorted((self.dns_root / "noisy").glob("**/*.wav"))
                assert len(self.noisy_files) == len(self.clean_files), (
                    f"Noisy ({len(self.noisy_files)}) and clean ({len(self.clean_files)}) "
                    "counts must match when mix_on_the_fly=False"
                )

        print("Pre-scanning audio durations for dynamic batching ...")
        self.durations: list[int] = []
        self.sample_index: list[tuple[int, int]] | None = [] if self.use_all_segments else None
        for file_idx, path in enumerate(tqdm(self.clean_files, desc="Scanning durations", unit="file", leave=False)):
            info = sf.info(str(path))
            n_samples = int(info.duration * self.sample_rate)
            # print(f"File: {path.name}, Duration: {info.duration:.2f}s, Samples: {n_samples}")
            capped_samples = min(n_samples, self.max_clip_samples)
            if self.use_all_segments and self.segment_samples is not None:
                num_segments = max(1, capped_samples // self.segment_samples)
                self.durations.extend([self.segment_samples] * num_segments)
                assert self.sample_index is not None
                for segment_idx in range(num_segments):
                    self.sample_index.append((file_idx, segment_idx))
            else:
                self.durations.append(capped_samples)
        print(
            f"Duration scan complete. Range: "
            f"{min(self.durations)/sample_rate:.2f}s – {max(self.durations)/sample_rate:.2f}s"
        )
        if self.sample_index is not None:
            print(
                f"Deterministic segment mode: {len(self.clean_files)} files -> "
                f"{len(self.sample_index)} segments of {self.segment_samples / self.sample_rate:.2f}s"
            )

    def __len__(self) -> int:
        if self.sample_index is not None:
            return len(self.sample_index)
        return len(self.clean_files)

    def _load_mono(self, path: Path) -> torch.Tensor:
        wav_data, sr = sf.read(str(path), always_2d=True)
        wav = torch.from_numpy(wav_data.T).float()
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != self.sample_rate:
            print(f"Resampling {path} from {sr} Hz to {self.sample_rate} Hz")
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
        return wav.squeeze(0)

    def _cap_length(self, wav: torch.Tensor) -> torch.Tensor:
        if wav.shape[-1] > self.max_clip_samples:
            start = random.randint(0, wav.shape[-1] - self.max_clip_samples)
            return wav[start: start + self.max_clip_samples]
        return wav

    def _mix_at_snr(self, clean: torch.Tensor, noise: torch.Tensor, snr_db: float) -> torch.Tensor:
        clean_rms = clean.pow(2).mean().sqrt().clamp(min=1e-9)
        noise_rms = noise.pow(2).mean().sqrt().clamp(min=1e-9)
        snr_linear = 10 ** (snr_db / 20.0)
        noise_scaled = noise * (clean_rms / (noise_rms * snr_linear))
        return (clean + noise_scaled).clamp(-1.0, 1.0)

    def _slice_fixed_segment(
        self,
        noisy: torch.Tensor,
        clean: torch.Tensor,
        segment_idx: int | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.segment_samples is None:
            return noisy, clean

        aligned_len = min(noisy.shape[-1], clean.shape[-1])
        noisy = noisy[:aligned_len]
        clean = clean[:aligned_len]
        seg_len = self.segment_samples

        if segment_idx is None:
            start = random.randint(0, aligned_len - seg_len) if aligned_len > seg_len else 0
        else:
            start = segment_idx * seg_len

        end = start + seg_len
        if end <= aligned_len:
            return noisy[start:end], clean[start:end]

        noisy_seg = torch.zeros(seg_len, dtype=noisy.dtype)
        clean_seg = torch.zeros(seg_len, dtype=clean.dtype)
        if start < aligned_len:
            valid = aligned_len - start
            noisy_seg[:valid] = noisy[start:aligned_len]
            clean_seg[:valid] = clean[start:aligned_len]
        return noisy_seg, clean_seg

    def __getitem__(self, idx: int):
        mapped_idx = idx
        segment_idx: int | None = None
        if self.sample_index is not None:
            mapped_idx, segment_idx = self.sample_index[idx]
        try:
            return self._getitem_inner(mapped_idx, segment_idx)
        except Exception:
            fallback = (idx + 1) % len(self)
            if self.sample_index is not None:
                mapped_idx, segment_idx = self.sample_index[fallback]
            else:
                mapped_idx, segment_idx = fallback, None
            return self._getitem_inner(mapped_idx, segment_idx)

    def _getitem_inner(self, idx: int, segment_idx: int | None = None):
        clean = self._load_mono(self.clean_files[idx])
        if self.use_all_segments:
            clean = clean[: self.max_clip_samples]
        else:
            clean = self._cap_length(clean)
        n_samples = clean.shape[-1]

        if self.mix_on_the_fly:
            noise_path = random.choice(self.noise_files)
            noise = self._load_mono(noise_path)
            if noise.shape[-1] < n_samples:
                valid_len = noise.shape[-1]
                noise = F.pad(noise, (0, n_samples - valid_len))
                mask = torch.zeros(n_samples, dtype=noise.dtype)
                mask[:valid_len] = 1.0
                noise = noise * mask
            elif noise.shape[-1] > n_samples:
                start = random.randint(0, noise.shape[-1] - n_samples)
                noise = noise[start: start + n_samples]
            snr = random.uniform(self.snr_low, self.snr_high)
            noisy = self._mix_at_snr(clean, noise, snr)
        else:
            noisy = self._load_mono(self.noisy_files[idx])
            if self.use_all_segments:
                noisy = noisy[: self.max_clip_samples]

        if self.use_all_segments:
            noisy, clean = self._slice_fixed_segment(noisy, clean, segment_idx)

        if self.augment:
            noisy = self.aug(noisy)
            noisy = noisy.clamp(-1.0, 1.0)

        return noisy, clean


class DynamicBatchSampler(Sampler):
    """Form batches so each batch has roughly `max_tokens` audio samples."""

    def __init__(
        self,
        durations: list[int],
        max_tokens: int,
        max_batch_size: int = 64,
        shuffle: bool = True,
        drop_last: bool = True,
    ):
        self.durations = durations
        self.max_tokens = max_tokens
        self.max_batch_size = max_batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self._cached_batches: list[list[int]] | None = None
        self._epoch = 0

    def _build_batches(self) -> list[list[int]]:
        indices = list(range(len(self.durations)))
        if self.shuffle:
            indices.sort(key=lambda i: self.durations[i])
            n_buckets = 20
            bucket_size = max(1, len(indices) // n_buckets)
            for start in range(0, len(indices), bucket_size):
                chunk = indices[start: start + bucket_size]
                random.shuffle(chunk)
                indices[start: start + bucket_size] = chunk
        else:
            indices.sort(key=lambda i: self.durations[i])

        batches: list[list[int]] = []
        current_batch: list[int] = []
        current_max_dur = 0

        for idx in indices:
            dur = self.durations[idx]
            new_max_dur = max(current_max_dur, dur)
            new_total = new_max_dur * (len(current_batch) + 1)
            if len(current_batch) > 0 and (
                new_total > self.max_tokens or len(current_batch) >= self.max_batch_size
            ):
                batches.append(current_batch)
                current_batch = [idx]
                current_max_dur = dur
            else:
                current_batch.append(idx)
                current_max_dur = new_max_dur

        if current_batch and not self.drop_last:
            batches.append(current_batch)
        elif current_batch and len(current_batch) > 1:
            batches.append(current_batch)

        return batches

    def __iter__(self):
        batches = self._build_batches()
        self._cached_batches = batches
        self._epoch += 1
        if self.shuffle:
            random.shuffle(batches)
        for batch in batches:
            yield batch

    def __len__(self) -> int:
        if self._cached_batches is not None:
            return len(self._cached_batches)
        return len(self._build_batches())


def dynamic_collate_fn(batch: list[tuple[torch.Tensor, torch.Tensor]]):
    """Collate variable-length (noisy, clean) pairs → padded batch tensors + lengths."""
    noisys, cleans = zip(*batch)
    lengths = torch.tensor([x.shape[-1] for x in cleans], dtype=torch.long)
    max_len = max(x.shape[-1] for x in cleans)
    noisy_batch = torch.stack([F.pad(x, (0, max_len - x.shape[-1])) for x in noisys])
    clean_batch = torch.stack([F.pad(x, (0, max_len - x.shape[-1])) for x in cleans])
    return noisy_batch, clean_batch, lengths
