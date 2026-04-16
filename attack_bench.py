#!/usr/bin/env python3
"""Benchmark the block-flow reconstruction attack against GOTCHA videos."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

try:
    import imageio.v2 as imageio
    import numpy as np
    from PIL import Image
except ImportError as exc:
    raise SystemExit(
        "Missing dependency. Install with: pip install numpy pillow imageio imageio-ffmpeg"
    ) from exc


DEFAULT_OUTPUT_DIR = "attack_bench"
DEFAULT_DOWNSCALE = 0.25
DEFAULT_BLOCK_SIZE = 8
DEFAULT_SEARCH_RADIUS = 3
DEFAULT_PAIR_STEP = 1
DEFAULT_MAX_PAIRS = 1
DEFAULT_WINDOW_SIZE = 0
DEFAULT_WINDOW_STRIDE = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the block-flow attack against a GOTCHA clip, "
            "save the output image, and log timing plus simple image metrics."
        )
    )
    parser.add_argument("input", help="Input video path.")
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for output PNGs and the report JSON. Defaults to {DEFAULT_OUTPUT_DIR}.",
    )
    parser.add_argument(
        "--report-name",
        default="report.json",
        help="JSON report filename written inside --output-dir.",
    )
    parser.add_argument(
        "--downscale",
        type=float,
        default=DEFAULT_DOWNSCALE,
        help="Scale factor applied before analysis. Lower values are much faster.",
    )
    parser.add_argument(
        "--frame-step",
        type=int,
        default=1,
        help="Keep every Nth frame from the source video.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Maximum frames to analyze. 0 means use all sampled frames.",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=DEFAULT_BLOCK_SIZE,
        help="Block size in pixels for the block-matching attack.",
    )
    parser.add_argument(
        "--search-radius",
        type=int,
        default=DEFAULT_SEARCH_RADIUS,
        help="Search radius in pixels for block matching.",
    )
    parser.add_argument(
        "--pair-step",
        type=int,
        default=DEFAULT_PAIR_STEP,
        help="Frame gap for consecutive motion pairs.",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=DEFAULT_MAX_PAIRS,
        help="Maximum frame pairs used by the block-matching attack.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=DEFAULT_WINDOW_SIZE,
        help=(
            "Optional frame-window length to scan. 0 means analyze the full sampled clip "
            "as one span."
        ),
    )
    parser.add_argument(
        "--window-stride",
        type=int,
        default=DEFAULT_WINDOW_STRIDE,
        help="Stride in sampled frames between candidate windows.",
    )
    parser.add_argument(
        "--include-full-window",
        action="store_true",
        help="Also evaluate the full sampled clip when --window-size is set.",
    )
    args = parser.parse_args()
    validate_args(args)
    return args


def validate_args(args: argparse.Namespace) -> None:
    if args.downscale <= 0 or args.downscale > 1:
        raise SystemExit("--downscale must be in the range (0, 1].")
    if args.frame_step <= 0:
        raise SystemExit("--frame-step must be greater than 0.")
    if args.max_frames < 0:
        raise SystemExit("--max-frames must be 0 or greater.")
    if args.block_size <= 0:
        raise SystemExit("--block-size must be greater than 0.")
    if args.search_radius < 0:
        raise SystemExit("--search-radius must be 0 or greater.")
    if args.pair_step <= 0:
        raise SystemExit("--pair-step must be greater than 0.")
    if args.max_pairs <= 0:
        raise SystemExit("--max-pairs must be greater than 0.")
    if args.window_size < 0:
        raise SystemExit("--window-size must be 0 or greater.")
    if args.window_stride <= 0:
        raise SystemExit("--window-stride must be positive.")


def rgb_to_grayscale(frame: np.ndarray) -> np.ndarray:
    frame = np.asarray(frame, dtype=np.float32)
    if frame.ndim == 2:
        grayscale = frame
    else:
        grayscale = (
            0.299 * frame[..., 0]
            + 0.587 * frame[..., 1]
            + 0.114 * frame[..., 2]
        )
    return grayscale.astype(np.float32) / 255.0


def resize_grayscale(frame: np.ndarray, scale: float) -> np.ndarray:
    if scale == 1.0:
        return frame

    height, width = frame.shape
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    image = Image.fromarray(np.clip(frame * 255.0, 0.0, 255.0).astype(np.uint8))
    resized = image.resize((new_width, new_height), Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.float32) / 255.0


def load_video_frames(
    input_path: Path,
    downscale: float,
    frame_step: int,
    max_frames: int,
) -> tuple[np.ndarray, dict]:
    reader = imageio.get_reader(input_path)
    metadata = reader.get_meta_data()
    frames = []

    try:
        for index, frame in enumerate(reader):
            if index % frame_step != 0:
                continue

            grayscale = rgb_to_grayscale(frame)
            grayscale = resize_grayscale(grayscale, downscale)
            frames.append(grayscale)

            if max_frames and len(frames) >= max_frames:
                break
    finally:
        reader.close()

    if len(frames) < 2:
        raise SystemExit("Need at least two sampled frames to run the attack benchmark.")

    stacked = np.stack(frames, axis=0).astype(np.float32)
    effective_fps = metadata.get("fps")
    if effective_fps:
        effective_fps = effective_fps / frame_step

    summary = {
        "source_fps": metadata.get("fps"),
        "effective_fps": effective_fps,
        "sampled_frames": int(stacked.shape[0]),
        "height": int(stacked.shape[1]),
        "width": int(stacked.shape[2]),
    }
    return stacked, summary


def normalize_image(image: np.ndarray) -> np.ndarray:
    finite = np.nan_to_num(image.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    low, high = np.percentile(finite, [1, 99])
    if high - low < 1e-6:
        return np.zeros_like(finite, dtype=np.float32)
    normalized = (finite - low) / (high - low)
    return np.clip(normalized, 0.0, 1.0).astype(np.float32)


def otsu_separation_score(image: np.ndarray) -> float:
    histogram, _ = np.histogram(image, bins=256, range=(0.0, 1.0))
    total = histogram.sum()
    if total == 0:
        return 0.0

    probability = histogram.astype(np.float64) / total
    values = np.linspace(0.0, 1.0, num=256, dtype=np.float64)
    omega = np.cumsum(probability)
    mu = np.cumsum(probability * values)
    mu_total = mu[-1]
    sigma_total = np.sum(((values - mu_total) ** 2) * probability)
    if sigma_total <= 1e-12:
        return 0.0

    sigma_between = ((mu_total * omega - mu) ** 2) / np.maximum(omega * (1.0 - omega), 1e-12)
    return float(np.max(sigma_between) / sigma_total)


def image_metrics(image: np.ndarray) -> dict[str, float]:
    p05, p50, p95 = np.percentile(image, [5, 50, 95])
    grad_y, grad_x = np.gradient(image)
    edge_energy = np.mean(np.hypot(grad_x, grad_y))
    return {
        "p05": round(float(p05), 6),
        "p50": round(float(p50), 6),
        "p95": round(float(p95), 6),
        "contrast_span": round(float(p95 - p05), 6),
        "edge_energy": round(float(edge_energy), 6),
        "otsu_separation": round(otsu_separation_score(image), 6),
    }


def readability_proxy_score(metrics: dict[str, float]) -> float:
    normalized_edge = min(1.0, metrics["edge_energy"] * 4.0)
    score = (
        0.45 * normalized_edge
        + 0.35 * metrics["contrast_span"]
        + 0.20 * metrics["otsu_separation"]
    )
    return round(float(score), 6)


def save_grayscale_png(image: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    png = Image.fromarray(np.clip(image * 255.0, 0.0, 255.0).astype(np.uint8))
    png.save(output_path)


def block_sum(image: np.ndarray, block_size: int) -> np.ndarray:
    blocks_y = image.shape[0] // block_size
    blocks_x = image.shape[1] // block_size
    trimmed = image[: blocks_y * block_size, : blocks_x * block_size]
    return trimmed.reshape(blocks_y, block_size, blocks_x, block_size).sum(axis=(1, 3))


def estimate_block_motion(
    first: np.ndarray,
    second: np.ndarray,
    block_size: int,
    search_radius: int,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = first.shape
    inner_height = ((height - (2 * search_radius)) // block_size) * block_size
    inner_width = ((width - (2 * search_radius)) // block_size) * block_size
    if inner_height <= 0 or inner_width <= 0:
        raise SystemExit(
            "Downscaled frame is too small for the current --block-size and --search-radius."
        )

    base = first[
        search_radius : search_radius + inner_height,
        search_radius : search_radius + inner_width,
    ]
    blocks_y = inner_height // block_size
    blocks_x = inner_width // block_size
    best_sad = np.full((blocks_y, blocks_x), np.inf, dtype=np.float32)
    best_dx = np.zeros((blocks_y, blocks_x), dtype=np.float32)
    best_dy = np.zeros((blocks_y, blocks_x), dtype=np.float32)

    for dy in range(-search_radius, search_radius + 1):
        for dx in range(-search_radius, search_radius + 1):
            shifted = second[
                search_radius + dy : search_radius + dy + inner_height,
                search_radius + dx : search_radius + dx + inner_width,
            ]
            sad = block_sum(np.abs(base - shifted), block_size)
            improved = sad < best_sad
            best_sad = np.where(improved, sad, best_sad)
            best_dx = np.where(improved, dx, best_dx)
            best_dy = np.where(improved, dy, best_dy)

    return best_dx, best_dy


def expand_blocks(
    values: np.ndarray,
    block_size: int,
    output_height: int,
    output_width: int,
    pad: int,
) -> np.ndarray:
    expanded = np.repeat(np.repeat(values, block_size, axis=0), block_size, axis=1)
    canvas = np.zeros((output_height, output_width), dtype=np.float32)
    height = min(expanded.shape[0], output_height - pad)
    width = min(expanded.shape[1], output_width - pad)
    canvas[pad : pad + height, pad : pad + width] = expanded[:height, :width]
    return canvas


def sampled_pair_indices(frame_count: int, pair_step: int, max_pairs: int) -> np.ndarray:
    last_start = frame_count - pair_step
    if last_start <= 0:
        raise SystemExit("Need more sampled frames for the current --pair-step value.")

    pair_count = min(last_start, max_pairs)
    indices = np.linspace(0, last_start - 1, num=pair_count, dtype=int)
    return np.unique(indices)


def block_flow_fields(
    frames: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray]:
    pair_indices = sampled_pair_indices(frames.shape[0], args.pair_step, args.max_pairs)
    first_dx, first_dy = estimate_block_motion(
        frames[pair_indices[0]],
        frames[pair_indices[0] + args.pair_step],
        args.block_size,
        args.search_radius,
    )
    accum_dx = np.zeros_like(first_dx, dtype=np.float32)
    accum_dy = np.zeros_like(first_dy, dtype=np.float32)

    for index in pair_indices:
        motion_dx, motion_dy = estimate_block_motion(
            frames[index],
            frames[index + args.pair_step],
            args.block_size,
            args.search_radius,
        )
        accum_dx += motion_dx
        accum_dy += motion_dy

    return accum_dx, accum_dy


def block_flow_angle_attack(frames: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    accum_dx, accum_dy = block_flow_fields(frames, args)
    angle = np.arctan2(accum_dy, accum_dx)
    normalized = (angle + math.pi) / (2.0 * math.pi)
    return expand_blocks(
        normalized,
        args.block_size,
        frames.shape[1],
        frames.shape[2],
        args.search_radius,
    )


def frame_windows(
    frame_count: int,
    window_size: int,
    window_stride: int,
    include_full_window: bool,
) -> list[tuple[int, int]]:
    if frame_count < 2:
        raise SystemExit("Need at least two sampled frames to run the attack benchmark.")

    windows: list[tuple[int, int]] = []
    if window_size <= 0 or window_size >= frame_count:
        windows.append((0, frame_count))
        return windows

    last_start = frame_count - window_size
    starts = list(range(0, last_start + 1, window_stride))
    if starts[-1] != last_start:
        starts.append(last_start)
    windows.extend((start, start + window_size) for start in starts)

    if include_full_window:
        windows.append((0, frame_count))

    deduped: list[tuple[int, int]] = []
    seen = set()
    for window in windows:
        if window in seen:
            continue
        seen.add(window)
        deduped.append(window)
    return deduped


def run_attack(
    frames: np.ndarray,
    args: argparse.Namespace,
    output_dir: Path,
) -> dict:
    windows = frame_windows(
        frame_count=frames.shape[0],
        window_size=args.window_size,
        window_stride=args.window_stride,
        include_full_window=args.include_full_window,
    )
    best_result = None
    window_results = []

    for start_frame, end_frame in windows:
        windowed_frames = frames[start_frame:end_frame]
        start = time.perf_counter()
        recovered = block_flow_angle_attack(windowed_frames, args)
        elapsed = time.perf_counter() - start
        normalized = normalize_image(recovered)
        metrics = image_metrics(normalized)
        selection_score = readability_proxy_score(metrics)
        candidate = {
            "start_frame": start_frame,
            "end_frame_exclusive": end_frame,
            "frame_count": end_frame - start_frame,
            "seconds": round(elapsed, 4),
            "selection_score": selection_score,
            "metrics": metrics,
            "image": normalized,
        }
        window_results.append(
            {
                key: value
                for key, value in candidate.items()
                if key != "image"
            }
        )
        if best_result is None or candidate["selection_score"] > best_result["selection_score"]:
            best_result = candidate

    assert best_result is not None
    output_path = output_dir / "block_flow_angle.png"
    save_grayscale_png(best_result["image"], output_path)
    return {
        "algorithm": "block_flow_angle",
        "seconds": best_result["seconds"],
        "output_image": str(output_path),
        "metrics": best_result["metrics"],
        "window_selection_score": best_result["selection_score"],
        "best_window": {
            "start_frame": best_result["start_frame"],
            "end_frame_exclusive": best_result["end_frame_exclusive"],
            "frame_count": best_result["frame_count"],
        },
        "evaluated_windows": window_results,
    }


def print_summary(result: dict) -> None:
    metrics = result["metrics"]
    best_window = result["best_window"]
    window_label = f"{best_window['start_frame']}:{best_window['end_frame_exclusive'] - 1}"
    print(
        f"{'algorithm':<22} {'window':<11} {'seconds':>8} {'score':>8} {'otsu':>8}  output"
    )
    print(
        f"{'block_flow_angle':<22} "
        f"{window_label:<11} "
        f"{result['seconds']:>8.4f} "
        f"{result['window_selection_score']:>8.4f} "
        f"{metrics['otsu_separation']:>8.4f} "
        f"  "
        f"{result['output_image']}"
    )


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Input video does not exist: {input_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frames, video_summary = load_video_frames(
        input_path=input_path,
        downscale=args.downscale,
        frame_step=args.frame_step,
        max_frames=args.max_frames,
    )
    result = run_attack(frames, args, output_dir)
    report_path = output_dir / args.report_name
    report = {
        "input": str(input_path),
        "analysis": {
            "downscale": args.downscale,
            "frame_step": args.frame_step,
            "max_frames": args.max_frames,
            "block_size": args.block_size,
            "search_radius": args.search_radius,
            "pair_step": args.pair_step,
            "max_pairs": args.max_pairs,
            "window_size": args.window_size,
            "window_stride": args.window_stride,
            "include_full_window": args.include_full_window,
        },
        "video": video_summary,
        "result": result,
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print_summary(result)
    print(f"\nWrote report to {report_path}")


if __name__ == "__main__":
    main()
