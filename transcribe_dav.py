from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path
import re

from faster_whisper import WhisperModel


def format_timestamp(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def extract_start_datetime(video_path: Path) -> datetime | None:
    match = re.search(r"-(\d{14})-\d{14}$", video_path.stem)
    if not match:
        return None
    try:
        return datetime.strptime(match.group(1), "%Y%m%d%H%M%S")
    except ValueError:
        return None


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python transcribe_dav.py <video_path> [output_path]")
        return 1

    video_path = Path(sys.argv[1]).expanduser()
    if not video_path.exists():
        print(f"File not found: {video_path}")
        return 1

    if len(sys.argv) >= 3:
        output_path = Path(sys.argv[2]).expanduser()
    else:
        output_path = Path.cwd() / f"{video_path.stem}.txt"

    recording_start = extract_start_datetime(video_path)

    model = WhisperModel("small", device="cpu", compute_type="int8")
    segments, info = model.transcribe(
        str(video_path),
        language="pt",
        beam_size=5,
        vad_filter=True,
        condition_on_previous_text=True,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as handle:
        handle.write(
            f"Arquivo: {video_path.name}\n"
            f"Idioma: {info.language}\n"
            f"Probabilidade idioma: {info.language_probability:.4f}\n\n"
        )
        for segment in segments:
            text = segment.text.strip()
            if not text:
                continue
            start = format_timestamp(segment.start)
            end = format_timestamp(segment.end)
            if recording_start is not None:
                absolute_start = (recording_start + timedelta(seconds=segment.start)).strftime("%H:%M:%S")
                absolute_end = (recording_start + timedelta(seconds=segment.end)).strftime("%H:%M:%S")
                handle.write(
                    f"[{absolute_start} - {absolute_end}] "
                    f"(offset {start} - {end}) {text}\n"
                )
            else:
                handle.write(f"[{start} - {end}] {text}\n")

    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
