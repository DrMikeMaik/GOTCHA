#!/usr/bin/env python3
"""Sweep defense generator settings and rank them by attack resistance."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import time
from pathlib import Path

import numpy as np

from attack_bench import (
    resize_grayscale,
    rgb_to_grayscale,
    save_grayscale_png,
)
from attack_pair_sweep import analyze_candidates
from generate_baseline import (
    DEFAULT_DURATION,
    DEFAULT_FEATHER,
    DEFAULT_FONT_SIZE,
    DEFAULT_FPS,
    DEFAULT_GRAIN,
    DEFAULT_HEIGHT,
    DEFAULT_TEXT,
    DEFAULT_TEXT_DRIFT,
    DEFAULT_TEXT_DRIFT_SPEED,
    DEFAULT_WIDTH,
    write_animation,
)
from generate_defense import (
    DEFAULT_ACTIVE_PHASES,
    DEFAULT_BACKGROUND_CYCLE_HOLD,
    DEFAULT_BACKGROUND_CYCLE_STEP,
    DEFAULT_PALETTE,
    DEFAULT_PHASE_COUNT,
    DEFAULT_PHASE_HOLD,
    DEFAULT_PHASE_MODE,
    DEFAULT_SCHEDULE_MODE,
    DEFAULT_SCHEDULE_SPAN,
    DEFAULT_TILE_SIZE,
    DEFAULT_TEXT_VECTOR_INDEX,
    build_animation,
    parse_palette,
)


DEFAULT_SWEEP_OUTPUT_DIR = "attack_resistance_sweep"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a grid of defense generator variants, attack each with "
            "the pair-sweep method, and rank settings by attack resistance."
        )
    )

    # --- Fixed generation params ---
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--output-dir", default=DEFAULT_SWEEP_OUTPUT_DIR)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--duration", type=float, default=DEFAULT_DURATION)
    parser.add_argument("--font-size", type=int, default=180)
    parser.add_argument("--font", default=None)
    parser.add_argument("--seed", type=int, default=1, help="Base random seed.")
    parser.add_argument("--random-digits", action="store_true")
    parser.add_argument("--palette", default=DEFAULT_PALETTE)
    parser.add_argument("--text-vector-index", type=int, default=DEFAULT_TEXT_VECTOR_INDEX)
    parser.add_argument("--background-vector-index", type=int, default=None)
    parser.add_argument("--pair-safe-max-gap", type=int, default=6)

    # --- Sweepable generation params (comma-separated) ---
    parser.add_argument(
        "--grains", default=str(DEFAULT_GRAIN),
        help="Comma-separated grain values. Sets both background and text grain unless overridden.",
    )
    parser.add_argument(
        "--background-grains", default=None,
        help="Comma-separated background grain values. Overrides --grains for background.",
    )
    parser.add_argument(
        "--text-grains", default=None,
        help="Comma-separated text grain values. Overrides --grains for text.",
    )
    parser.add_argument("--tile-sizes", default=str(DEFAULT_TILE_SIZE))
    parser.add_argument("--phase-counts", default=str(DEFAULT_PHASE_COUNT))
    parser.add_argument("--active-phases-values", default=str(DEFAULT_ACTIVE_PHASES))
    parser.add_argument("--phase-holds", default=str(DEFAULT_PHASE_HOLD))
    parser.add_argument("--schedule-spans", default=str(DEFAULT_SCHEDULE_SPAN))
    parser.add_argument("--background-cycle-steps", default=str(DEFAULT_BACKGROUND_CYCLE_STEP))
    parser.add_argument("--background-cycle-holds", default=str(DEFAULT_BACKGROUND_CYCLE_HOLD))
    parser.add_argument("--feathers", default=str(DEFAULT_FEATHER))
    parser.add_argument("--text-drifts", default=str(DEFAULT_TEXT_DRIFT))
    parser.add_argument("--text-drift-speeds", default=str(DEFAULT_TEXT_DRIFT_SPEED))
    parser.add_argument("--phase-modes", default=DEFAULT_PHASE_MODE)
    parser.add_argument("--schedule-modes", default=DEFAULT_SCHEDULE_MODE)

    # --- Attack config ---
    parser.add_argument("--downscale", type=float, default=0.25)
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--search-radius", type=int, default=3)
    parser.add_argument("--window-size", type=int, default=0)
    parser.add_argument("--window-stride", type=int, default=1)
    parser.add_argument("--include-full-window", action="store_true")

    # --- Output config ---
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--save-top-k", type=int, default=3)

    args = parser.parse_args()
    validate_args(args)
    return args


def validate_args(args: argparse.Namespace) -> None:
    if args.width <= 0 or args.height <= 0:
        raise SystemExit("--width and --height must be positive.")
    if args.fps <= 0:
        raise SystemExit("--fps must be positive.")
    if args.duration <= 0:
        raise SystemExit("--duration must be greater than 0.")
    if args.font_size <= 0:
        raise SystemExit("--font-size must be positive.")
    if args.downscale <= 0 or args.downscale > 1:
        raise SystemExit("--downscale must be in the range (0, 1].")
    if args.block_size <= 0:
        raise SystemExit("--block-size must be positive.")
    if args.search_radius < 0:
        raise SystemExit("--search-radius must be 0 or greater.")
    if args.window_size < 0:
        raise SystemExit("--window-size must be 0 or greater.")
    if args.window_stride <= 0:
        raise SystemExit("--window-stride must be positive.")
    if args.max_cases < 0 or args.save_top_k < 0:
        raise SystemExit("--max-cases and --save-top-k must be 0 or greater.")


def parse_int_list(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def parse_float_list(raw: str) -> list[float]:
    return [float(part.strip()) for part in raw.split(",") if part.strip()]


def parse_str_list(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def frame_count_for(args: argparse.Namespace) -> int:
    return max(1, int(round(args.duration * args.fps)))


def build_case_grid(args: argparse.Namespace) -> list[dict]:
    background_grains = parse_int_list(args.background_grains or args.grains)
    text_grains = parse_int_list(args.text_grains or args.grains)

    combinations = itertools.product(
        background_grains,
        text_grains,
        parse_int_list(args.tile_sizes),
        parse_int_list(args.phase_counts),
        parse_int_list(args.active_phases_values),
        parse_int_list(args.phase_holds),
        parse_int_list(args.schedule_spans),
        parse_int_list(args.background_cycle_steps),
        parse_int_list(args.background_cycle_holds),
        parse_float_list(args.feathers),
        parse_float_list(args.text_drifts),
        parse_float_list(args.text_drift_speeds),
        parse_str_list(args.phase_modes),
        parse_str_list(args.schedule_modes),
    )

    cases = []
    for index, (
        background_grain, text_grain, tile_size,
        phase_count, active_phases, phase_hold,
        schedule_span, background_cycle_step, background_cycle_hold,
        feather, text_drift, text_drift_speed,
        phase_mode, schedule_mode,
    ) in enumerate(combinations):
        if args.max_cases and len(cases) >= args.max_cases:
            break
        cases.append({
            "case_id": f"case_{index:03d}",
            "background_grain": background_grain,
            "text_grain": text_grain,
            "tile_size": tile_size,
            "phase_count": phase_count,
            "active_phases": active_phases,
            "phase_hold": phase_hold,
            "schedule_span": schedule_span,
            "background_cycle_step": background_cycle_step,
            "background_cycle_hold": background_cycle_hold,
            "feather": feather,
            "text_drift": text_drift,
            "text_drift_speed": text_drift_speed,
            "phase_mode": phase_mode,
            "schedule_mode": schedule_mode,
            "seed": args.seed + index,
        })

    if not cases:
        raise SystemExit("No sweep cases were generated.")
    return cases


def build_generator_namespace(case: dict, args: argparse.Namespace) -> argparse.Namespace:
    palette_vectors = parse_palette(args.palette)
    return argparse.Namespace(
        text=args.text,
        width=args.width,
        height=args.height,
        fps=args.fps,
        duration=args.duration,
        font_size=args.font_size,
        font=args.font,
        seed=case["seed"],
        random_digits=args.random_digits,
        grain=case["background_grain"],
        background_grain=case["background_grain"],
        text_grain=case["text_grain"],
        feather=case["feather"],
        text_drift=case["text_drift"],
        text_drift_speed=case["text_drift_speed"],
        tile_size=case["tile_size"],
        palette=args.palette,
        palette_vectors=palette_vectors,
        text_vector_index=args.text_vector_index,
        background_vector_index=args.background_vector_index,
        phase_count=case["phase_count"],
        phase_hold=case["phase_hold"],
        active_phases=case["active_phases"],
        background_cycle_step=case["background_cycle_step"],
        background_cycle_hold=case["background_cycle_hold"],
        phase_mode=case["phase_mode"],
        schedule_mode=case["schedule_mode"],
        schedule_span=case["schedule_span"],
        pair_safe_max_gap=args.pair_safe_max_gap,
    )


def build_sweep_namespace(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        pair_steps=[1],
        window_size=args.window_size,
        window_stride=args.window_stride,
        include_full_window=args.include_full_window,
        block_size=args.block_size,
        search_radius=args.search_radius,
    )


def generate_analysis_frames(gen_args: argparse.Namespace, downscale: float) -> np.ndarray:
    frames = []
    for frame in build_animation(gen_args):
        grayscale = rgb_to_grayscale(frame)
        frames.append(resize_grayscale(grayscale, downscale))
    return np.stack(frames, axis=0).astype(np.float32)


def analyze_case(case: dict, args: argparse.Namespace, sweep_args: argparse.Namespace) -> dict:
    gen_args = build_generator_namespace(case, args)

    generation_start = time.perf_counter()
    frames = generate_analysis_frames(gen_args, args.downscale)
    generation_seconds = time.perf_counter() - generation_start

    attack_start = time.perf_counter()
    candidates = analyze_candidates(frames, sweep_args)
    attack_seconds = time.perf_counter() - attack_start

    best = candidates[0] if candidates else None
    best_score = best["selection_score"] if best else 0.0
    best_window = (
        f"{best['start_frame']}:{best['end_frame_exclusive']}"
        if best else "N/A"
    )

    return {
        "case_id": case["case_id"],
        "params": {key: value for key, value in case.items() if key != "case_id"},
        "generation_seconds": round(generation_seconds, 4),
        "attack_seconds": round(attack_seconds, 4),
        "best_score": round(best_score, 6),
        "best_window": best_window,
        "best_metrics": best["metrics"] if best else {},
        "image": best["image"] if best else None,
    }


def save_ranked_artifacts(
    cases: list[dict],
    output_dir: Path,
    save_top_k: int,
    args: argparse.Namespace,
) -> None:
    if save_top_k <= 0:
        return

    sorted_cases = sorted(cases, key=lambda item: item["best_score"])
    hardest = sorted_cases[:save_top_k]
    easiest = sorted_cases[-save_top_k:]
    picks = hardest + [c for c in easiest if c not in hardest]

    artifact_dir = output_dir / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    for case in picks:
        if case["image"] is not None:
            attack_path = artifact_dir / f"{case['case_id']}_attack.png"
            save_grayscale_png(case["image"], attack_path)
            case["attack_image"] = str(attack_path)

        video_path = artifact_dir / f"{case['case_id']}.mp4"
        gen_args = build_generator_namespace(case["params"], args)
        frames = build_animation(gen_args)
        write_animation(frames, video_path, args.fps, False)
        case["video"] = str(video_path)


def scrub_images(cases: list[dict]) -> list[dict]:
    return [
        {key: value for key, value in case.items() if key != "image"}
        for case in cases
    ]


SWEPT_PARAM_NAMES = [
    "background_grain", "text_grain", "tile_size",
    "phase_count", "active_phases", "phase_hold",
    "schedule_span", "background_cycle_step", "background_cycle_hold",
    "feather", "text_drift", "text_drift_speed",
    "phase_mode", "schedule_mode",
]


def write_csv(cases: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "case_id",
        *SWEPT_PARAM_NAMES,
        "seed",
        "best_score",
        "best_window",
        "attack_seconds",
        "generation_seconds",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for case in sorted(cases, key=lambda item: item["best_score"]):
            row = {"case_id": case["case_id"]}
            for name in SWEPT_PARAM_NAMES:
                row[name] = case["params"][name]
            row["seed"] = case["params"]["seed"]
            row["best_score"] = case["best_score"]
            row["best_window"] = case["best_window"]
            row["attack_seconds"] = case["attack_seconds"]
            row["generation_seconds"] = case["generation_seconds"]
            writer.writerow(row)


def print_summary(cases: list[dict]) -> None:
    print(f"{'case_id':<10} {'score':>8} {'window':<11} {'attack_s':>9} {'gen_s':>8}  params")
    for case in sorted(cases, key=lambda item: item["best_score"]):
        params = case["params"]
        param_str = " ".join(
            f"{name}={params[name]}" for name in SWEPT_PARAM_NAMES
        )
        print(
            f"{case['case_id']:<10} "
            f"{case['best_score']:>8.4f} "
            f"{case['best_window']:<11} "
            f"{case['attack_seconds']:>9.4f} "
            f"{case['generation_seconds']:>8.4f}  "
            f"{param_str}"
        )


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sweep_args = build_sweep_namespace(args)

    cases = [analyze_case(case, args, sweep_args) for case in build_case_grid(args)]
    save_ranked_artifacts(cases, output_dir, args.save_top_k, args)

    cleaned = scrub_images(cases)
    report_path = output_dir / "report.json"
    report_path.write_text(
        json.dumps(
            {
                "text": args.text,
                "video": {
                    "width": args.width,
                    "height": args.height,
                    "fps": args.fps,
                    "duration": args.duration,
                    "font_size": args.font_size,
                    "font": args.font,
                },
                "attack": {
                    "downscale": args.downscale,
                    "block_size": args.block_size,
                    "search_radius": args.search_radius,
                    "window_size": args.window_size,
                    "window_stride": args.window_stride,
                    "include_full_window": args.include_full_window,
                },
                "cases": cleaned,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    write_csv(cleaned, output_dir / "summary.csv")
    print_summary(cleaned)
    print(f"\nWrote report to {report_path}")


if __name__ == "__main__":
    main()
