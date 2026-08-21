#!/usr/bin/env python3
"""Mask videos with SAM3, then export square-cropped frames with ffmpeg.

Finds videos in a folder by filename suffix / glob, runs
``frames_for_fig/run_sam3.py --create_output_vid`` on each, copies the
masked mp4 into ``<output>/masked_videos/``, and writes every Nth frame
(cropped to a square) into ``<output>/frames/<clip>/``.

Videos are expected at 25 fps, 1080x720; actual size/fps are probed.
The square side can be set per video as ``SUFFIX:CROP:SIZE``. If omitted
it falls back to ``--square-size``, then to min(width, height). On
1080x720 that default is 720x720, so top/mid/bot collapse to the same y.

Example
-------
python frames_for_fig/make_masked_frames.py \\
    --video-dir /path/to/clips \\
    --output-dir /path/to/out \\
    --video '*_clip00006.mp4:top_left:480' \\
    --video '*_clip00012.mp4:mid_center'
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


CROP_ALIASES = {
    "top_left": "top_left",
    "tl": "top_left",
    "top-left": "top_left",
    "topleft": "top_left",
    "top_center": "top_center",
    "tc": "top_center",
    "top-center": "top_center",
    "topcenter": "top_center",
    "top": "top_center",
    "top_right": "top_right",
    "tr": "top_right",
    "top-right": "top_right",
    "topright": "top_right",
    "mid_left": "mid_left",
    "ml": "mid_left",
    "mid-left": "mid_left",
    "midleft": "mid_left",
    "left": "mid_left",
    "mid_center": "mid_center",
    "mc": "mid_center",
    "mid-center": "mid_center",
    "midcenter": "mid_center",
    "center": "mid_center",
    "centre": "mid_center",
    "middle": "mid_center",
    "mid_right": "mid_right",
    "mr": "mid_right",
    "mid-right": "mid_right",
    "midright": "mid_right",
    "right": "mid_right",
    "bot_left": "bot_left",
    "bl": "bot_left",
    "bot-left": "bot_left",
    "bottom_left": "bot_left",
    "bottom-left": "bot_left",
    "bottomleft": "bot_left",
    "bot_center": "bot_center",
    "bc": "bot_center",
    "bot-center": "bot_center",
    "bottom_center": "bot_center",
    "bottom-center": "bot_center",
    "bottomcenter": "bot_center",
    "bottom": "bot_center",
    "bot": "bot_center",
    "bot_right": "bot_right",
    "br": "bot_right",
    "bot-right": "bot_right",
    "bottom_right": "bot_right",
    "bottom-right": "bot_right",
    "bottomright": "bot_right",
}

# Fraction of leftover (width-side, height-side) used as the crop origin.
CROP_FRACTIONS = {
    "top_left": (0.0, 0.0),
    "top_center": (0.5, 0.0),
    "top_right": (1.0, 0.0),
    "mid_left": (0.0, 0.5),
    "mid_center": (0.5, 0.5),
    "mid_right": (1.0, 0.5),
    "bot_left": (0.0, 1.0),
    "bot_center": (0.5, 1.0),
    "bot_right": (1.0, 1.0),
}

HERE = Path(__file__).resolve().parent
DEFAULT_SAM3 = HERE / "run_sam3.py"


def parse_args() -> argparse.Namespace:
    crop_help = ", ".join(CROP_FRACTIONS)
    parser = argparse.ArgumentParser(
        description="SAM3-mask videos, then ffmpeg-export square-cropped frames.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Crop positions: {crop_help}",
    )
    parser.add_argument(
        "--video-dir",
        required=True,
        type=Path,
        help="Folder containing the source videos",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Output root; creates masked_videos/ and frames/ inside it",
    )
    parser.add_argument(
        "--video",
        action="append",
        dest="videos",
        metavar="SUFFIX:CROP[:SIZE]",
        required=True,
        help=(
            "Video selector, crop, and optional square side in pixels, "
            "e.g. '*_clip00006.mp4:top_left:480' or '*_clip00012.mp4:mid_center'. "
            "The part before the first ':' is a glob or filename suffix in "
            "--video-dir. Repeat this flag once per video."
        ),
    )
    parser.add_argument(
        "--every",
        type=int,
        default=8,
        help="Keep every Nth frame (default: 8)",
    )
    parser.add_argument(
        "--square-size",
        type=int,
        default=None,
        help=(
            "Default square crop side in pixels for videos that omit SIZE. "
            "If neither is set, uses min(width, height)"
        ),
    )
    parser.add_argument(
        "--prompt",
        default="fish",
        help="SAM3 text prompt (default: fish)",
    )
    parser.add_argument(
        "--sam3-script",
        type=Path,
        default=DEFAULT_SAM3,
        help="Path to run_sam3.py",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter used to run SAM3",
    )
    parser.add_argument(
        "--skip-sam3",
        action="store_true",
        help="Do not run SAM3; extract frames from existing masked videos",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run SAM3 and overwrite existing masked videos / frames",
    )
    parser.add_argument(
        "--ext",
        default="png",
        help="Frame image extension (default: png)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the plan without running SAM3 or ffmpeg",
    )
    return parser.parse_args()


def normalize_crop(name: str) -> str:
    key = name.strip().lower().replace(" ", "_")
    if key not in CROP_ALIASES:
        valid = ", ".join(CROP_FRACTIONS)
        raise ValueError(f"Unknown crop '{name}'. Choose one of: {valid}")
    return CROP_ALIASES[key]


def parse_video_spec(
    spec: str,
    default_size: int | None = None,
) -> tuple[str, str, int | None]:
    """Parse SUFFIX:CROP or SUFFIX:CROP:SIZE. SIZE falls back to default_size."""
    if ":" not in spec:
        raise ValueError(
            f"Expected SUFFIX:CROP[:SIZE], got '{spec}'. "
            "Example: '*_clip00006.mp4:top_left:480'"
        )
    rest, last = spec.rsplit(":", 1)
    last = last.strip()
    square_size: int | None
    if last.isdigit():
        square_size = int(last)
        if ":" not in rest:
            raise ValueError(
                f"Expected SUFFIX:CROP[:SIZE], got '{spec}'. "
                "Example: '*_clip00006.mp4:top_left:480'"
            )
        pattern, crop = rest.rsplit(":", 1)
    else:
        square_size = default_size
        pattern, crop = rest, last
    pattern = pattern.strip()
    if not pattern:
        raise ValueError(f"Empty video pattern in '{spec}'")
    if square_size is not None and square_size <= 0:
        raise ValueError(f"Square size must be positive, got {square_size} in '{spec}'")
    return pattern, normalize_crop(crop), square_size


def to_glob(pattern: str) -> str:
    if any(ch in pattern for ch in "*?["):
        return pattern
    if pattern.startswith("."):
        return f"*{pattern}"
    return f"*{pattern}" if not pattern.startswith("*") else pattern


def find_videos(video_dir: Path, pattern: str) -> list[Path]:
    matches = sorted(
        p for p in video_dir.glob(to_glob(pattern))
        if p.is_file()
    )
    if not matches:
        raise FileNotFoundError(
            f"No files in {video_dir} matching '{to_glob(pattern)}'"
        )
    return matches


def ffprobe_size(video: Path) -> tuple[int, int, float]:
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate",
        "-of", "json",
        str(video),
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    stream = json.loads(result.stdout)["streams"][0]
    width = int(stream["width"])
    height = int(stream["height"])
    num, den = stream["r_frame_rate"].split("/")
    fps = float(num) / float(den)
    return width, height, fps


def crop_xy(
    width: int,
    height: int,
    crop_name: str,
    square_size: int | None,
) -> tuple[int, int, int]:
    side = square_size if square_size is not None else min(width, height)
    if side <= 0:
        raise ValueError(f"square-size must be positive, got {side}")
    if side > width or side > height:
        raise ValueError(
            f"square-size {side} does not fit in {width}x{height}"
        )
    fx, fy = CROP_FRACTIONS[crop_name]
    x = int(round(fx * (width - side)))
    y = int(round(fy * (height - side)))
    return side, x, y


def run_sam3(
    python: str,
    script: Path,
    video: Path,
    work_dir: Path,
    prompt: str,
) -> Path:
    work_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        python, str(script),
        str(video),
        "--create_output_vid",
        "--output_dir", str(work_dir),
        "--prompt", prompt,
    ]
    print(f"\n[sam3] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    stem = video.stem
    fixed = work_dir / f"{stem}_output" / f"{stem}_fixed.mp4"
    if not fixed.is_file():
        raise FileNotFoundError(f"SAM3 did not produce {fixed}")
    return fixed


def export_frames(
    masked_video: Path,
    frames_dir: Path,
    every: int,
    side: int,
    x: int,
    y: int,
    ext: str,
) -> int:
    frames_dir.mkdir(parents=True, exist_ok=True)
    # settb+setpts keeps the original frame index as the image filename.
    vf = (
        f"settb=1,setpts=N,"
        f"select='not(mod(n\\,{every}))',"
        f"crop={side}:{side}:{x}:{y}"
    )
    out_pattern = str(frames_dir / f"frame_%06d.{ext}")
    cmd = [
        "ffmpeg", "-y",
        "-i", str(masked_video),
        "-vf", vf,
        "-vsync", "vfr",
        "-frame_pts", "1",
        out_pattern,
    ]
    print(f"[ffmpeg] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    return len(list(frames_dir.glob(f"frame_*.{ext}")))


def main() -> int:
    args = parse_args()
    video_dir = args.video_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    sam3_script = args.sam3_script.expanduser().resolve()

    if not video_dir.is_dir():
        raise SystemExit(f"video-dir is not a directory: {video_dir}")
    if args.every < 1:
        raise SystemExit("--every must be >= 1")
    if not args.skip_sam3 and not sam3_script.is_file():
        raise SystemExit(f"SAM3 script not found: {sam3_script}")

    masked_dir = output_dir / "masked_videos"
    frames_root = output_dir / "frames"
    work_dir = masked_dir / "_sam3_work"

    jobs = []
    for spec in args.videos:
        pattern, crop, square_size = parse_video_spec(spec, args.square_size)
        for src in find_videos(video_dir, pattern):
            jobs.append((src, crop, pattern, square_size))

    print(f"Found {len(jobs)} video(s) in {video_dir}")
    for src, crop, pattern, square_size in jobs:
        size_label = str(square_size) if square_size is not None else "min(w,h)"
        print(f"  {src.name}  [{pattern}]  crop={crop}  size={size_label}")

    if args.dry_run:
        return 0

    masked_dir.mkdir(parents=True, exist_ok=True)
    frames_root.mkdir(parents=True, exist_ok=True)

    for src, crop, _pattern, square_size in jobs:
        dest_masked = masked_dir / f"{src.stem}_masked.mp4"
        frames_dir = frames_root / src.stem

        if args.force and frames_dir.exists():
            shutil.rmtree(frames_dir)

        if not args.skip_sam3 and (args.force or not dest_masked.is_file()):
            produced = run_sam3(
                args.python, sam3_script, src, work_dir, args.prompt,
            )
            shutil.copy2(produced, dest_masked)
            print(f"[copy] {produced} -> {dest_masked}")
        elif not dest_masked.is_file():
            raise SystemExit(
                f"Masked video missing: {dest_masked}\n"
                "Run without --skip-sam3, or pass --force."
            )
        else:
            print(f"[skip sam3] {dest_masked}")

        width, height, fps = ffprobe_size(dest_masked)
        side, x, y = crop_xy(width, height, crop, square_size)
        print(
            f"[crop] {src.stem}: {width}x{height} @ {fps:.3g} fps -> "
            f"{side}x{side} at x={x}, y={y} ({crop})"
        )
        n_frames = export_frames(
            dest_masked, frames_dir, args.every, side, x, y, args.ext,
        )
        print(f"[done] {src.stem}: {n_frames} frames -> {frames_dir}")

    print(f"\nMasked videos: {masked_dir}")
    print(f"Frames:        {frames_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
