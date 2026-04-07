#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import wave


DYS_DIR = Path("/home/cmy/MeanFlowSE/tmp_audio/dysarthria")
NORMAL_DIR = Path("/home/cmy/MeanFlowSE/tmp_audio/normal")
OUTPUT_FILE = Path("/home/cmy/MeanFlowSE/paired_audio_lengths.txt")
AUDIO_EXTS = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}


def collect_audio_index(root: Path, normal_suffix: str = "") -> tuple[dict[str, Path], int, int]:
    index: dict[str, Path] = {}
    total_files = 0
    duplicate_ids = 0

    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in AUDIO_EXTS:
            continue

        total_files += 1
        stem = path.stem
        if normal_suffix and stem.endswith(normal_suffix):
            stem = stem[: -len(normal_suffix)]
        if stem in index:
            duplicate_ids += 1
            continue
        index[stem] = path

    return index, total_files, duplicate_ids


def wav_duration_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as wav_file:
        frame_rate = wav_file.getframerate()
        if frame_rate <= 0:
            raise ValueError(f"Invalid sample rate for {path}: {frame_rate}")
        return wav_file.getnframes() / frame_rate


def seconds_to_hms(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


def format_pair_line(audio_id: str, dys_sec: float, normal_sec: float) -> str:
    diff_sec = dys_sec - normal_sec
    return (
        f"{audio_id} | dys={dys_sec:.3f}s | normal={normal_sec:.3f}s | "
        f"diff(dys-normal)={diff_sec:.3f}s"
    )


def main() -> None:
    if not DYS_DIR.exists():
        raise FileNotFoundError(f"Dysarthria directory not found: {DYS_DIR}")
    if not NORMAL_DIR.exists():
        raise FileNotFoundError(f"Normal directory not found: {NORMAL_DIR}")

    dys_index, dys_total, dys_duplicates = collect_audio_index(DYS_DIR)
    normal_index, normal_total, normal_duplicates = collect_audio_index(
        NORMAL_DIR, normal_suffix="_normal"
    )

    dys_ids = set(dys_index)
    normal_ids = set(normal_index)
    paired_ids = sorted(dys_ids & normal_ids)
    only_dys = dys_ids - normal_ids
    only_normal = normal_ids - dys_ids

    dys_total_sec = 0.0
    normal_total_sec = 0.0
    diffs: list[tuple[str, float]] = []
    failed_pairs = 0

    pair_lines: list[str] = []

    for audio_id in paired_ids:
        dys_path = dys_index[audio_id]
        normal_path = normal_index[audio_id]

        if dys_path.suffix.lower() != ".wav" or normal_path.suffix.lower() != ".wav":
            failed_pairs += 1
            continue

        try:
            dys_sec = wav_duration_seconds(dys_path)
            normal_sec = wav_duration_seconds(normal_path)
        except (wave.Error, ValueError, OSError):
            failed_pairs += 1
            continue

        dys_total_sec += dys_sec
        normal_total_sec += normal_sec
        diffs.append((audio_id, dys_sec - normal_sec))
        pair_lines.append(format_pair_line(audio_id, dys_sec, normal_sec))

    valid_pairs = len(diffs)
    if valid_pairs:
        total_diff_sec = dys_total_sec - normal_total_sec
        avg_diff_sec = total_diff_sec / valid_pairs
        avg_abs_diff_sec = sum(abs(diff) for _, diff in diffs) / valid_pairs
        max_longer_dys = max(diffs, key=lambda item: item[1])
        max_longer_normal = min(diffs, key=lambda item: item[1])
        dys_longer_count = sum(1 for _, diff in diffs if diff > 0)
        normal_longer_count = sum(1 for _, diff in diffs if diff < 0)
        equal_count = valid_pairs - dys_longer_count - normal_longer_count
    else:
        total_diff_sec = 0.0
        avg_diff_sec = 0.0
        avg_abs_diff_sec = 0.0
        max_longer_dys = ("N/A", 0.0)
        max_longer_normal = ("N/A", 0.0)
        dys_longer_count = 0
        normal_longer_count = 0
        equal_count = 0

    print(f"Dysarthria audio files: {dys_total}")
    print(f"Normal audio files: {normal_total}")
    print(f"Paired audio ids: {len(paired_ids)}")
    print(f"Unpaired dysarthria ids: {len(only_dys)}")
    print(f"Unpaired normal ids: {len(only_normal)}")
    print(f"Duplicate dysarthria ids skipped: {dys_duplicates}")
    print(f"Duplicate normal ids skipped: {normal_duplicates}")
    print(f"Pairs analyzed for duration: {valid_pairs}")
    print(f"Pairs skipped for duration: {failed_pairs}")
    print(f"Total dysarthria duration: {dys_total_sec:.3f}s ({seconds_to_hms(dys_total_sec)})")
    print(f"Total normal duration: {normal_total_sec:.3f}s ({seconds_to_hms(normal_total_sec)})")
    print(f"Total duration diff (dys - normal): {total_diff_sec:.3f}s")
    print(f"Average duration diff per pair (dys - normal): {avg_diff_sec:.6f}s")
    print(f"Average absolute duration diff per pair: {avg_abs_diff_sec:.6f}s")
    print(f"Dysarthria longer pairs: {dys_longer_count}")
    print(f"Normal longer pairs: {normal_longer_count}")
    print(f"Equal-length pairs: {equal_count}")
    print(
        "Max dysarthria longer pair: "
        f"{max_longer_dys[0]} ({max_longer_dys[1]:.6f}s)"
    )
    print(
        "Max normal longer pair: "
        f"{max_longer_normal[0]} ({abs(max_longer_normal[1]):.6f}s)"
    )

    with OUTPUT_FILE.open("w", encoding="utf-8") as output_fp:
        for line in pair_lines:
            print(line)
            output_fp.write(line + "\n")

    print(f"Saved per-pair duration lines to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()