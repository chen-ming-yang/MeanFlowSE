#!/usr/bin/env python3
from __future__ import annotations

import json
import wave
from pathlib import Path

DYS_DIR = Path("/home/cmy/MeanFlowSE/tmp_audio/dysarthria")
NORMAL_DIR = Path("/home/cmy/MeanFlowSE/tmp_audio/normal")
OUTPUT_JSON = Path("/home/cmy/MeanFlowSE/pairs.json")
DIFF_THRESHOLD = 4.0  # seconds — pairs with |diff| > this are excluded
AUDIO_EXTS = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}
ROOT = Path("/home/cmy/MeanFlowSE")


def collect_audio_index(root: Path, normal_suffix: str = "") -> dict[str, Path]:
    index: dict[str, Path] = {}
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in AUDIO_EXTS:
            continue
        stem = path.stem
        if normal_suffix and stem.endswith(normal_suffix):
            stem = stem[: -len(normal_suffix)]
        if stem not in index:
            index[stem] = path
    return index


def wav_duration_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as f:
        rate = f.getframerate()
        if rate <= 0:
            raise ValueError(f"Invalid sample rate in {path}")
        return f.getnframes() / rate


def main() -> None:
    dys_index = collect_audio_index(DYS_DIR)
    normal_index = collect_audio_index(NORMAL_DIR, normal_suffix="_normal")

    paired_ids = sorted(set(dys_index) & set(normal_index))

    used: list[dict] = []
    excluded: list[dict] = []

    for audio_id in paired_ids:
        dys_path = dys_index[audio_id]
        normal_path = normal_index[audio_id]

        try:
            dys_sec = wav_duration_seconds(dys_path)
            normal_sec = wav_duration_seconds(normal_path)
        except (wave.Error, ValueError, OSError) as exc:
            print(f"[WARN] Skipping {audio_id}: {exc}")
            continue

        diff = dys_sec - normal_sec
        entry = {
            "id": audio_id,
            "dysarthria": str(dys_path.relative_to(ROOT)),
            "normal": str(normal_path.relative_to(ROOT)),
            "dys_duration": round(dys_sec, 6),
            "normal_duration": round(normal_sec, 6),
            "diff": round(diff, 6),
        }

        if abs(diff) > DIFF_THRESHOLD:
            excluded.append(entry)
        else:
            used.append(entry)

    output = {
        "threshold_seconds": DIFF_THRESHOLD,
        "total_pairs": len(used) + len(excluded),
        "used_count": len(used),
        "excluded_count": len(excluded),
        "used": used,
        "excluded": excluded,
    }

    with OUTPUT_JSON.open("w", encoding="utf-8") as fp:
        json.dump(output, fp, indent=2, ensure_ascii=False)

    print(f"Total pairs:    {output['total_pairs']}")
    print(f"Used pairs:     {output['used_count']}")
    print(f"Excluded pairs: {output['excluded_count']}  (|diff| > {DIFF_THRESHOLD}s)")
    print(f"Saved to:       {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
