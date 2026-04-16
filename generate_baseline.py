#!/usr/bin/env python3
"""Generate a text-perception noise animation as MP4 or GIF.

The output is a full-frame noise field where the background slides left while
the text region uses a separate vertically moving noise pattern. A single
frame should look like uniform noise, but the text becomes perceptible across
time.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable, Optional, Tuple

try:
    import imageio.v2 as imageio
    import numpy as np
    from PIL import Image, ImageDraw, ImageFilter, ImageFont
except ImportError as exc:
    raise SystemExit(
        "Missing dependency. Install with: pip install numpy pillow imageio imageio-ffmpeg"
    ) from exc


DEFAULT_OUTPUT = "text_noise.mp4"
DEFAULT_WIDTH = 1920
DEFAULT_HEIGHT = 1080
DEFAULT_FPS = 30
DEFAULT_DURATION = 5.0
DEFAULT_SPEED = 3.0
DEFAULT_FONT_SIZE = 340
DEFAULT_TEXT = "VISIBLE"
DEFAULT_GRAIN = 3
DEFAULT_FEATHER = 1.25
DEFAULT_TEXT_DRIFT = 200.0
DEFAULT_TEXT_DRIFT_SPEED = 0.16
DEFAULT_SEED = None

FONT_CANDIDATES = (
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Supplemental/Helvetica.ttc",
    "/System/Library/Fonts/Supplemental/Impact.ttf",
    "/Library/Fonts/Arial Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a noise animation where text emerges because the "
            "background and text-region noise move in different directions."
        )
    )
    parser.add_argument(
        "--text",
        default=DEFAULT_TEXT,
        help=f'Text to render. Defaults to "{DEFAULT_TEXT}".',
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Output path. Defaults to {DEFAULT_OUTPUT}.",
    )
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--duration", type=float, default=DEFAULT_DURATION)
    parser.add_argument(
        "--speed",
        type=float,
        default=DEFAULT_SPEED,
        help="Horizontal background speed in pixels per frame.",
    )
    parser.add_argument("--font-size", type=int, default=DEFAULT_FONT_SIZE)
    parser.add_argument(
        "--gif",
        action="store_true",
        help="Write GIF output. If set and --output has no .gif extension, one is added.",
    )
    parser.add_argument(
        "--grain",
        type=int,
        default=DEFAULT_GRAIN,
        help="Noise block size in pixels. Larger values create chunkier noise.",
    )
    parser.add_argument(
        "--feather",
        type=float,
        default=DEFAULT_FEATHER,
        help="Text mask blur radius in pixels.",
    )
    parser.add_argument(
        "--text-drift",
        type=float,
        default=DEFAULT_TEXT_DRIFT,
        help="Maximum whole-text drift in pixels.",
    )
    parser.add_argument(
        "--text-drift-speed",
        type=float,
        default=DEFAULT_TEXT_DRIFT_SPEED,
        help="Whole-text drift speed in cycles per second.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for reproducible output.",
    )
    parser.add_argument(
        "--font",
        default=None,
        help="Optional path to a .ttf/.otf font file. Defaults to a common bold system font.",
    )
    args = parser.parse_args()
    validate_args(args)
    return args


def validate_args(args: argparse.Namespace) -> None:
    if args.width <= 0 or args.height <= 0:
        raise SystemExit("Width and height must be positive integers.")
    if args.fps <= 0:
        raise SystemExit("FPS must be a positive integer.")
    if args.duration <= 0:
        raise SystemExit("Duration must be greater than 0.")
    if args.font_size <= 0:
        raise SystemExit("Font size must be greater than 0.")
    if args.grain <= 0:
        raise SystemExit("Grain must be greater than 0.")
    if args.speed < 0:
        raise SystemExit("Speed must be 0 or greater.")
    if args.text_drift < 0:
        raise SystemExit("Text drift must be 0 or greater.")
    if args.text_drift_speed < 0:
        raise SystemExit("Text drift speed must be 0 or greater.")
    if not args.text.strip():
        raise SystemExit("Text cannot be empty.")


def resolve_output_path(output: str, force_gif: bool) -> Path:
    path = Path(output)
    if force_gif and path.suffix.lower() != ".gif":
        path = path.with_suffix(".gif")
    if not force_gif and path.suffix == "":
        path = path.with_suffix(".mp4")
    return path


def load_font(font_path: Optional[str], font_size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = []
    if font_path:
        candidates.append(font_path)
    candidates.extend(FONT_CANDIDATES)

    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, font_size)
        except OSError:
            continue

    return ImageFont.load_default()


def wrap_text_to_width(
    text: str,
    font: ImageFont.ImageFont,
    max_width: int,
    draw: ImageDraw.ImageDraw,
) -> str:
    paragraphs = text.splitlines() or [text]
    wrapped_paragraphs = []

    for paragraph in paragraphs:
        words = paragraph.split()
        if not words:
            wrapped_paragraphs.append("")
            continue

        current = words[0]
        for word in words[1:]:
            trial = f"{current} {word}"
            trial_box = draw.multiline_textbbox((0, 0), trial, font=font, spacing=0)
            if trial_box[2] - trial_box[0] <= max_width:
                current = trial
            else:
                wrapped_paragraphs.append(current)
                current = word
        wrapped_paragraphs.append(current)

    return "\n".join(wrapped_paragraphs)


def fit_text(
    text: str,
    width: int,
    height: int,
    preferred_size: int,
    font_path: Optional[str],
) -> Tuple[str, ImageFont.ImageFont]:
    probe_image = Image.new("L", (width, height), 0)
    probe_draw = ImageDraw.Draw(probe_image)
    max_width = int(width * 0.82)
    max_height = int(height * 0.58)

    for size in range(preferred_size, 11, -4):
        font = load_font(font_path, size)
        wrapped = wrap_text_to_width(text, font, max_width, probe_draw)
        bbox = probe_draw.multiline_textbbox((0, 0), wrapped, font=font, align="center", spacing=int(size * 0.12))
        box_width = bbox[2] - bbox[0]
        box_height = bbox[3] - bbox[1]
        if box_width <= max_width and box_height <= max_height:
            return wrapped, font

    return wrap_text_to_width(text, load_font(font_path, 12), max_width, probe_draw), load_font(font_path, 12)


def render_text_image(
    text: str,
    width: int,
    height: int,
    font_size: int,
    font_path: Optional[str],
) -> Image.Image:
    wrapped_text, font = fit_text(text, width, height, font_size, font_path)
    image = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(image)
    spacing = max(0, int(getattr(font, "size", font_size) * 0.12))
    bbox = draw.multiline_textbbox((0, 0), wrapped_text, font=font, align="center", spacing=spacing)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    origin = (
        (width - text_width) / 2.0 - bbox[0],
        (height - text_height) / 2.0 - bbox[1],
    )
    draw.multiline_text(origin, wrapped_text, fill=255, font=font, align="center", spacing=spacing)

    return image


def make_text_mask(
    text_image: Image.Image,
    feather: float,
) -> np.ndarray:
    image = text_image
    if feather > 0:
        image = image.filter(ImageFilter.GaussianBlur(radius=feather))

    return np.asarray(image, dtype=np.float32) / 255.0


def make_noise(
    rng: np.random.Generator,
    height: int,
    width: int,
    grain: int,
) -> np.ndarray:
    reduced_h = math.ceil(height / grain)
    reduced_w = math.ceil(width / grain)
    base = rng.random((reduced_h, reduced_w), dtype=np.float32)
    expanded = np.repeat(np.repeat(base, grain, axis=0), grain, axis=1)
    return expanded[:height, :width]


def compose_frame(
    background_noise: np.ndarray,
    text_noise: np.ndarray,
    text_mask: np.ndarray,
    background_offset: int,
    text_offset: int,
    width: int,
    height: int,
) -> np.ndarray:
    background = background_noise[:, background_offset : background_offset + width]
    text_layer = text_noise[text_offset : text_offset + height, :]
    frame = background * (1.0 - text_mask) + text_layer * text_mask
    rgb = np.repeat(frame[:, :, None], 3, axis=2) * 255.0
    return np.clip(rgb, 0.0, 255.0).astype(np.uint8)


def shift_mask(mask: np.ndarray, x_offset: int, y_offset: int) -> np.ndarray:
    shifted = np.zeros_like(mask)
    height, width = mask.shape

    src_x0 = max(0, -x_offset)
    src_x1 = min(width, width - x_offset)
    src_y0 = max(0, -y_offset)
    src_y1 = min(height, height - y_offset)
    dest_x0 = max(0, x_offset)
    dest_x1 = min(width, width + x_offset)
    dest_y0 = max(0, y_offset)
    dest_y1 = min(height, height + y_offset)

    if src_x0 >= src_x1 or src_y0 >= src_y1:
        return shifted

    shifted[dest_y0:dest_y1, dest_x0:dest_x1] = mask[src_y0:src_y1, src_x0:src_x1]
    return shifted


def frame_offsets(frame_count: int, speed: float, width: int) -> Iterable[int]:
    for index in range(frame_count):
        yield int(round(index * speed)) % width


def text_drift_offsets(
    frame_count: int,
    fps: int,
    drift: float,
    speed: float,
) -> Iterable[Tuple[int, int]]:
    if drift <= 0 or speed <= 0:
        for _ in range(frame_count):
            yield 0, 0
        return

    phase_step = (2.0 * math.pi * speed) / fps
    for index in range(frame_count):
        phase = index * phase_step
        x_offset = int(round(math.sin(phase) * drift))
        y_offset = int(round(math.cos(phase * 0.73 + math.pi / 6.0) * drift * 0.45))
        yield x_offset, y_offset


def build_animation(
    width: int,
    height: int,
    text: str,
    font_size: int,
    frame_count: int,
    fps: int,
    speed: float,
    grain: int,
    feather: float,
    text_drift: float,
    text_drift_speed: float,
    seed: Optional[int],
    font_path: Optional[str],
) -> Iterable[np.ndarray]:
    rng = np.random.default_rng(seed)
    text_image = render_text_image(text, width, height, font_size, font_path)
    base_text_mask = make_text_mask(text_image, feather)
    text_noise = make_noise(rng, height, width, grain)
    text_noise = np.concatenate((text_noise, text_noise), axis=0)

    # Duplicate horizontally so each frame can read a contiguous sliding window.
    background_noise = make_noise(rng, height, width, grain)
    background_noise = np.concatenate((background_noise, background_noise), axis=1)

    for background_offset, text_offset, (x_offset, y_offset) in zip(
        frame_offsets(frame_count, speed, width),
        frame_offsets(frame_count, speed, height),
        text_drift_offsets(frame_count, fps, text_drift, text_drift_speed),
    ):
        text_mask = shift_mask(base_text_mask, x_offset, y_offset)
        yield compose_frame(
            background_noise,
            text_noise,
            text_mask,
            background_offset,
            text_offset,
            width,
            height,
        )


def write_animation(
    frames: Iterable[np.ndarray],
    output_path: Path,
    fps: int,
    as_gif: bool,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if as_gif or output_path.suffix.lower() == ".gif":
        with imageio.get_writer(output_path, mode="I", duration=1.0 / fps, loop=0) as writer:
            for frame in frames:
                writer.append_data(frame)
        return

    with imageio.get_writer(
        output_path,
        fps=fps,
        codec="libx264",
        format="FFMPEG",
        quality=8,
        pixelformat="yuv420p",
        macro_block_size=None,
    ) as writer:
        for frame in frames:
            writer.append_data(frame)


def main() -> None:
    args = parse_args()
    output_path = resolve_output_path(args.output, args.gif)
    as_gif = args.gif or output_path.suffix.lower() == ".gif"
    frame_count = max(1, int(round(args.duration * args.fps)))
    frames = build_animation(
        width=args.width,
        height=args.height,
        text=args.text,
        font_size=args.font_size,
        frame_count=frame_count,
        fps=args.fps,
        speed=args.speed,
        grain=args.grain,
        feather=args.feather,
        text_drift=args.text_drift,
        text_drift_speed=args.text_drift_speed,
        seed=args.seed,
        font_path=args.font,
    )
    write_animation(frames, output_path, args.fps, as_gif)
    print(f"Wrote {frame_count} frames to {output_path}")


if __name__ == "__main__":
    main()
