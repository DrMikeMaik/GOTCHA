#!/usr/bin/env python3
"""Sweep pair gaps and windows for a single video, then rank attack artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

try:
    import numpy as np
    from PIL import Image, ImageDraw
except ImportError as exc:
    raise SystemExit(
        "Missing dependency. Install with: pip install numpy pillow imageio imageio-ffmpeg"
    ) from exc

from attack_bench import (
    DEFAULT_BLOCK_SIZE,
    DEFAULT_DOWNSCALE,
    DEFAULT_SEARCH_RADIUS,
    DEFAULT_WINDOW_STRIDE,
    block_flow_angle_attack,
    frame_windows,
    image_metrics,
    load_video_frames,
    normalize_image,
    readability_proxy_score,
    save_grayscale_png,
)


DEFAULT_PAIR_STEPS = "1"
DEFAULT_TOP_K = 12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a ranked block-flow sweep across many pair gaps and frame windows for one video."
        )
    )
    parser.add_argument("input", help="Input video path.")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for ranked PNGs, montage, CSV, and report JSON.",
    )
    parser.add_argument(
        "--report-name",
        default="report.json",
        help="JSON report filename written inside --output-dir.",
    )
    parser.add_argument(
        "--summary-name",
        default="summary.csv",
        help="CSV summary filename written inside --output-dir.",
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
        help="Block size in pixels for block matching.",
    )
    parser.add_argument(
        "--search-radius",
        type=int,
        default=DEFAULT_SEARCH_RADIUS,
        help="Search radius in pixels for block matching.",
    )
    parser.add_argument(
        "--pair-steps",
        default=DEFAULT_PAIR_STEPS,
        help="Comma-separated pair gaps to sweep, for example 1,2,3,4,5,6.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=0,
        help=(
            "Fixed frame-window length. 0 means use pair_step + 1 so each candidate is "
            "a single pair window."
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
        help="Also evaluate the full sampled clip alongside the scanned windows.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="How many highest-scoring artifacts to save and include in the montage.",
    )
    parser.add_argument(
        "--montage-cols",
        type=int,
        default=4,
        help="Number of columns in the top-k montage.",
    )
    args = parser.parse_args()
    args.pair_steps = parse_int_list(args.pair_steps)
    validate_args(args)
    return args


def parse_int_list(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


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
    if not args.pair_steps or any(step <= 0 for step in args.pair_steps):
        raise SystemExit("--pair-steps must contain at least one positive integer.")
    if args.window_size < 0:
        raise SystemExit("--window-size must be 0 or greater.")
    if args.window_stride <= 0:
        raise SystemExit("--window-stride must be greater than 0.")
    if args.top_k < 0:
        raise SystemExit("--top-k must be 0 or greater.")
    if args.montage_cols <= 0:
        raise SystemExit("--montage-cols must be greater than 0.")
    if args.window_size and any(args.window_size <= step for step in args.pair_steps):
        raise SystemExit("--window-size must be greater than every swept pair step.")


def attack_namespace(args: argparse.Namespace, pair_step: int) -> argparse.Namespace:
    return argparse.Namespace(
        block_size=args.block_size,
        search_radius=args.search_radius,
        pair_step=pair_step,
        max_pairs=1,
    )


def effective_window_size(args: argparse.Namespace, pair_step: int) -> int:
    if args.window_size > 0:
        return args.window_size
    return pair_step + 1


def analyze_candidates(frames: np.ndarray, args: argparse.Namespace) -> list[dict]:
    candidates = []
    for pair_step in args.pair_steps:
        if frames.shape[0] <= pair_step:
            continue

        window_size = effective_window_size(args, pair_step)
        windows = frame_windows(
            frame_count=frames.shape[0],
            window_size=window_size,
            window_stride=args.window_stride,
            include_full_window=args.include_full_window,
        )
        bench_args = attack_namespace(args, pair_step)

        for start_frame, end_frame in windows:
            windowed_frames = frames[start_frame:end_frame]
            if windowed_frames.shape[0] <= pair_step:
                continue

            start = time.perf_counter()
            recovered = normalize_image(block_flow_angle_attack(windowed_frames, bench_args))
            seconds = time.perf_counter() - start
            metrics = image_metrics(recovered)
            selection_score = readability_proxy_score(metrics)
            candidates.append(
                {
                    "pair_step": pair_step,
                    "start_frame": start_frame,
                    "end_frame_exclusive": end_frame,
                    "frame_count": end_frame - start_frame,
                    "seconds": round(seconds, 4),
                    "selection_score": selection_score,
                    "metrics": metrics,
                    "image": recovered,
                }
            )

    return sorted(candidates, key=lambda item: item["selection_score"], reverse=True)


def save_ranked_images(candidates: list[dict], output_dir: Path, top_k: int) -> list[dict]:
    saved = []
    for rank, candidate in enumerate(candidates[:top_k], start=1):
        filename = (
            f"rank_{rank:02d}_ps{candidate['pair_step']}_"
            f"w{candidate['start_frame']}_{candidate['end_frame_exclusive'] - 1}.png"
        )
        output_path = output_dir / filename
        save_grayscale_png(candidate["image"], output_path)
        saved_candidate = dict(candidate)
        saved_candidate["output_image"] = str(output_path)
        saved.append(saved_candidate)
    return saved


def tile_with_label(image: np.ndarray, label: str) -> Image.Image:
    tile = Image.fromarray(np.clip(image * 255.0, 0.0, 255.0).astype(np.uint8))
    canvas = Image.new("L", (tile.width, tile.height + 22), color=255)
    canvas.paste(tile, (0, 22))
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 5), label, fill=0)
    return canvas


def save_montage(candidates: list[dict], output_path: Path, cols: int) -> None:
    if not candidates:
        return

    labeled_tiles = []
    for candidate in candidates:
        label = (
            f"#{candidate['rank']} ps={candidate['pair_step']} "
            f"w={candidate['start_frame']}:{candidate['end_frame_exclusive'] - 1}"
        )
        labeled_tiles.append(tile_with_label(candidate["image"], label))

    tile_width = labeled_tiles[0].width
    tile_height = labeled_tiles[0].height
    rows = math.ceil(len(labeled_tiles) / cols)
    montage = Image.new("L", (cols * tile_width, rows * tile_height), color=255)

    for index, tile in enumerate(labeled_tiles):
        row = index // cols
        col = index % cols
        montage.paste(tile, (col * tile_width, row * tile_height))

    montage.save(output_path)


def scrub_candidates(candidates: list[dict]) -> list[dict]:
    cleaned = []
    for rank, candidate in enumerate(candidates, start=1):
        cleaned.append(
            {
                "rank": rank,
                "pair_step": candidate["pair_step"],
                "start_frame": candidate["start_frame"],
                "end_frame_exclusive": candidate["end_frame_exclusive"],
                "frame_count": candidate["frame_count"],
                "seconds": candidate["seconds"],
                "selection_score": candidate["selection_score"],
                "metrics": candidate["metrics"],
                "output_image": candidate.get("output_image"),
            }
        )
    return cleaned


def write_csv(candidates: list[dict], output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "rank",
                "pair_step",
                "start_frame",
                "end_frame_exclusive",
                "frame_count",
                "seconds",
                "selection_score",
                "contrast_span",
                "edge_energy",
                "otsu_separation",
            ],
        )
        writer.writeheader()
        for rank, candidate in enumerate(candidates, start=1):
            writer.writerow(
                {
                    "rank": rank,
                    "pair_step": candidate["pair_step"],
                    "start_frame": candidate["start_frame"],
                    "end_frame_exclusive": candidate["end_frame_exclusive"],
                    "frame_count": candidate["frame_count"],
                    "seconds": candidate["seconds"],
                    "selection_score": candidate["selection_score"],
                    "contrast_span": candidate["metrics"]["contrast_span"],
                    "edge_energy": candidate["metrics"]["edge_energy"],
                    "otsu_separation": candidate["metrics"]["otsu_separation"],
                }
            )


def print_summary(candidates: list[dict], top_k: int) -> None:
    print(
        f"{'rank':<6} {'pair':<6} {'window':<11} {'score':>8} {'otsu':>8} {'seconds':>8}"
    )
    for rank, candidate in enumerate(candidates[:top_k], start=1):
        window_label = f"{candidate['start_frame']}:{candidate['end_frame_exclusive'] - 1}"
        print(
            f"{rank:<6} "
            f"{candidate['pair_step']:<6} "
            f"{window_label:<11} "
            f"{candidate['selection_score']:>8.4f} "
            f"{candidate['metrics']['otsu_separation']:>8.4f} "
            f"{candidate['seconds']:>8.4f}"
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
    candidates = analyze_candidates(frames, args)
    if not candidates:
        raise SystemExit("No valid sweep candidates were generated for the requested pair steps.")

    saved = save_ranked_images(candidates, output_dir, args.top_k)
    for rank, candidate in enumerate(saved, start=1):
        candidate["rank"] = rank

    montage_path = output_dir / f"top{min(args.top_k, len(candidates))}_montage.png"
    save_montage(saved, montage_path, args.montage_cols)
    write_csv(candidates, output_dir / args.summary_name)

    report = {
        "input": str(input_path),
        "analysis": {
            "downscale": args.downscale,
            "frame_step": args.frame_step,
            "max_frames": args.max_frames,
            "block_size": args.block_size,
            "search_radius": args.search_radius,
            "pair_steps": args.pair_steps,
            "window_size": args.window_size,
            "window_stride": args.window_stride,
            "include_full_window": args.include_full_window,
            "top_k": args.top_k,
            "montage_cols": args.montage_cols,
        },
        "video": video_summary,
        "top_montage": str(montage_path),
        "top_candidates": scrub_candidates(saved),
        "all_candidates": scrub_candidates(candidates),
    }
    report_path = output_dir / args.report_name
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print_summary(candidates, args.top_k)
    print(f"\nWrote report to {report_path}")


if __name__ == "__main__":
    main()
