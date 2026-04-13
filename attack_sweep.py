#!/usr/bin/env python3
"""Sweep GOTCHA generator settings and rank them by attack recoverability."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import time
from pathlib import Path

import numpy as np

from attack_bench import (
    ALGORITHMS,
    frame_windows,
    image_metrics,
    normalize_image,
    resize_grayscale,
    rgb_to_grayscale,
)
from text_noise_video import (
    DEFAULT_DURATION,
    DEFAULT_FEATHER,
    DEFAULT_FONT_SIZE,
    DEFAULT_FPS,
    DEFAULT_GRAIN,
    DEFAULT_HEIGHT,
    DEFAULT_SPEED,
    DEFAULT_TEXT,
    DEFAULT_TEXT_DRIFT,
    DEFAULT_TEXT_DRIFT_SPEED,
    DEFAULT_WIDTH,
    build_animation,
    make_text_mask,
    shift_mask,
    text_drift_offsets,
)


DEFAULT_SWEEP_OUTPUT_DIR = "attack_sweep"
DEFAULT_SWEEP_ALGORITHMS = ("block_flow_angle",)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a grid of GOTCHA parameter variants, run the configured attack "
            "algorithms, and rank cases by how recoverable the hidden text is."
        )
    )
    parser.add_argument("--text", default=DEFAULT_TEXT, help="Text to render in generated cases.")
    parser.add_argument("--output-dir", default=DEFAULT_SWEEP_OUTPUT_DIR)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--duration", type=float, default=DEFAULT_DURATION)
    parser.add_argument("--font-size", type=int, default=180)
    parser.add_argument("--font", default=None)
    parser.add_argument("--seed", type=int, default=1, help="Base random seed.")
    parser.add_argument(
        "--grains",
        default=str(DEFAULT_GRAIN),
        help="Comma-separated grain values to sweep.",
    )
    parser.add_argument(
        "--speeds",
        default=str(DEFAULT_SPEED),
        help="Comma-separated background speed values to sweep.",
    )
    parser.add_argument(
        "--feathers",
        default=str(DEFAULT_FEATHER),
        help="Comma-separated feather values to sweep.",
    )
    parser.add_argument(
        "--text-drifts",
        default=str(DEFAULT_TEXT_DRIFT),
        help="Comma-separated text drift amplitudes to sweep.",
    )
    parser.add_argument(
        "--text-drift-speeds",
        default=str(DEFAULT_TEXT_DRIFT_SPEED),
        help="Comma-separated text drift speed values to sweep.",
    )
    parser.add_argument(
        "--algorithms",
        nargs="+",
        choices=tuple(ALGORITHMS.keys()),
        default=list(DEFAULT_SWEEP_ALGORITHMS),
        help="Attack algorithms used for scoring. Defaults to the primary attacker model only.",
    )
    parser.add_argument(
        "--downscale",
        type=float,
        default=0.25,
        help="Scale factor used before running attacks.",
    )
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--search-radius", type=int, default=3)
    parser.add_argument("--pair-step", type=int, default=1)
    parser.add_argument("--max-pairs", type=int, default=24)
    parser.add_argument(
        "--window-size",
        type=int,
        default=0,
        help="Optional frame-window length to scan. 0 means full sampled clip only.",
    )
    parser.add_argument(
        "--window-stride",
        type=int,
        default=1,
        help="Stride in sampled frames between candidate windows.",
    )
    parser.add_argument(
        "--include-full-window",
        action="store_true",
        help="Also evaluate the full sampled clip when --window-size is set.",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=0,
        help="Optional limit on the number of generated cases. 0 means all combinations.",
    )
    parser.add_argument(
        "--save-top-k",
        type=int,
        default=3,
        help="Save best-attack artifact images for the top K hardest and easiest cases.",
    )
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
    if args.pair_step <= 0 or args.max_pairs <= 0:
        raise SystemExit("--pair-step and --max-pairs must be positive.")
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


def frame_count_for(args: argparse.Namespace) -> int:
    return max(1, int(round(args.duration * args.fps)))


def build_case_grid(args: argparse.Namespace) -> list[dict]:
    cases = []
    combinations = itertools.product(
        parse_int_list(args.grains),
        parse_float_list(args.speeds),
        parse_float_list(args.feathers),
        parse_float_list(args.text_drifts),
        parse_float_list(args.text_drift_speeds),
    )

    for index, (grain, speed, feather, text_drift, text_drift_speed) in enumerate(combinations):
        if args.max_cases and len(cases) >= args.max_cases:
            break
        cases.append(
            {
                "case_id": f"case_{index:03d}",
                "grain": grain,
                "speed": speed,
                "feather": feather,
                "text_drift": text_drift,
                "text_drift_speed": text_drift_speed,
                "seed": args.seed + index,
            }
        )
    if not cases:
        raise SystemExit("No sweep cases were generated.")
    return cases


def analysis_namespace(args: argparse.Namespace, algorithms: list[str]) -> argparse.Namespace:
    return argparse.Namespace(
        algorithms=algorithms,
        block_size=args.block_size,
        search_radius=args.search_radius,
        pair_step=args.pair_step,
        max_pairs=args.max_pairs,
    )


def generate_analysis_frames(case: dict, args: argparse.Namespace) -> np.ndarray:
    frames = []
    for frame in build_animation(
        width=args.width,
        height=args.height,
        text=args.text,
        font_size=args.font_size,
        frame_count=frame_count_for(args),
        fps=args.fps,
        speed=case["speed"],
        grain=case["grain"],
        feather=case["feather"],
        text_drift=case["text_drift"],
        text_drift_speed=case["text_drift_speed"],
        seed=case["seed"],
        font_path=args.font,
    ):
        grayscale = rgb_to_grayscale(frame)
        frames.append(resize_grayscale(grayscale, args.downscale))
    return np.stack(frames, axis=0).astype(np.float32)


def generate_reference_frames(case: dict, args: argparse.Namespace) -> np.ndarray:
    base_mask = make_text_mask(
        text=args.text,
        width=args.width,
        height=args.height,
        font_size=args.font_size,
        feather=case["feather"],
        font_path=args.font,
    )
    frames = []
    for x_offset, y_offset in text_drift_offsets(
        frame_count_for(args),
        args.fps,
        case["text_drift"],
        case["text_drift_speed"],
    ):
        shifted = shift_mask(base_mask, x_offset, y_offset)
        frames.append(normalize_image(resize_grayscale(shifted, args.downscale)))
    return np.stack(frames, axis=0).astype(np.float32)


def edge_map(image: np.ndarray) -> np.ndarray:
    grad_y, grad_x = np.gradient(image)
    return normalize_image(np.hypot(grad_x, grad_y))


def pearson_correlation(first: np.ndarray, second: np.ndarray) -> float:
    a = first.astype(np.float64).ravel()
    b = second.astype(np.float64).ravel()
    a -= a.mean()
    b -= b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


def recovery_against_reference(prediction: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    reference_binary = reference >= 0.35
    if not np.any(reference_binary):
        reference_binary = reference >= 0.2

    if not np.any(reference_binary) or np.all(reference_binary):
        return {
            "correlation": 0.0,
            "masked_contrast": 0.0,
            "topk_precision": 0.0,
            "recoverability_score": 0.0,
        }

    inside = float(prediction[reference_binary].mean())
    outside = float(prediction[~reference_binary].mean())
    masked_contrast = max(0.0, inside - outside)

    k = int(reference_binary.sum())
    flat_prediction = prediction.ravel()
    topk_indices = np.argpartition(flat_prediction, -k)[-k:]
    topk_precision = float(reference_binary.ravel()[topk_indices].mean())

    correlation = max(0.0, pearson_correlation(prediction, reference))
    recoverability_score = min(
        1.0,
        (0.5 * topk_precision) + (0.3 * correlation) + (0.2 * masked_contrast),
    )
    return {
        "correlation": round(correlation, 6),
        "masked_contrast": round(masked_contrast, 6),
        "topk_precision": round(topk_precision, 6),
        "recoverability_score": round(recoverability_score, 6),
    }


def attack_recovery_metrics(
    prediction: np.ndarray,
    fill_reference: np.ndarray,
    edge_reference_map: np.ndarray,
) -> dict[str, float]:
    fill_metrics = recovery_against_reference(prediction, fill_reference)
    edge_metrics = recovery_against_reference(prediction, edge_reference_map)
    if edge_metrics["recoverability_score"] > fill_metrics["recoverability_score"]:
        combined = dict(edge_metrics)
        combined["reference_mode"] = "edge"
    else:
        combined = dict(fill_metrics)
        combined["reference_mode"] = "fill"
    combined["fill_score"] = fill_metrics["recoverability_score"]
    combined["edge_score"] = edge_metrics["recoverability_score"]
    return combined


def save_preview(image: np.ndarray, output_path: Path) -> None:
    from attack_bench import save_grayscale_png

    save_grayscale_png(image, output_path)


def analyze_case(case: dict, args: argparse.Namespace) -> dict:
    attack_args = analysis_namespace(args, args.algorithms)
    generation_start = time.perf_counter()
    frames = generate_analysis_frames(case, args)
    reference_frames = generate_reference_frames(case, args)
    generation_seconds = time.perf_counter() - generation_start
    windows = frame_windows(
        frame_count=frames.shape[0],
        window_size=args.window_size,
        window_stride=args.window_stride,
        include_full_window=args.include_full_window,
    )

    results = []
    for algorithm_name in args.algorithms:
        _, implementation = ALGORITHMS[algorithm_name]
        best_result = None
        window_results = []
        for start_frame, end_frame in windows:
            windowed_frames = frames[start_frame:end_frame]
            reference = normalize_image(reference_frames[start_frame:end_frame].mean(axis=0))
            reference_edges = edge_map(reference)
            start = time.perf_counter()
            recovered = normalize_image(implementation(windowed_frames, attack_args))
            seconds = time.perf_counter() - start
            recovery = attack_recovery_metrics(recovered, reference, reference_edges)
            candidate = {
                "algorithm": algorithm_name,
                "seconds": round(seconds, 4),
                "metrics": image_metrics(recovered),
                "recovery": recovery,
                "image": recovered,
                "reference": reference,
                "reference_edges": reference_edges,
                "start_frame": start_frame,
                "end_frame_exclusive": end_frame,
                "frame_count": end_frame - start_frame,
            }
            window_results.append(
                {
                    key: value
                    for key, value in candidate.items()
                    if key not in {"image", "reference", "reference_edges"}
                }
            )
            if (
                best_result is None
                or candidate["recovery"]["recoverability_score"]
                > best_result["recovery"]["recoverability_score"]
            ):
                best_result = candidate

        assert best_result is not None
        results.append(
            {
                "algorithm": algorithm_name,
                "seconds": best_result["seconds"],
                "metrics": best_result["metrics"],
                "recovery": best_result["recovery"],
                "image": best_result["image"],
                "reference": best_result["reference"],
                "reference_edges": best_result["reference_edges"],
                "best_window": {
                    "start_frame": best_result["start_frame"],
                    "end_frame_exclusive": best_result["end_frame_exclusive"],
                    "frame_count": best_result["frame_count"],
                },
                "evaluated_windows": window_results,
            }
        )

    best = max(results, key=lambda item: item["recovery"]["recoverability_score"])
    return {
        "case_id": case["case_id"],
        "params": {
            "grain": case["grain"],
            "speed": case["speed"],
            "feather": case["feather"],
            "text_drift": case["text_drift"],
            "text_drift_speed": case["text_drift_speed"],
            "seed": case["seed"],
        },
        "generation_seconds": round(generation_seconds, 4),
        "best_attack_algorithm": best["algorithm"],
        "best_recoverability_score": best["recovery"]["recoverability_score"],
        "best_attack_seconds": best["seconds"],
        "best_window": best["best_window"],
        "reference_metrics": image_metrics(best["reference"]),
        "results": results,
        "reference": best["reference"],
        "reference_edges": best["reference_edges"],
    }


def save_ranked_artifacts(
    cases: list[dict],
    output_dir: Path,
    save_top_k: int,
) -> None:
    if save_top_k <= 0:
        return

    sorted_cases = sorted(cases, key=lambda item: item["best_recoverability_score"])
    picks = sorted_cases[:save_top_k] + sorted_cases[-save_top_k:]
    artifact_dir = output_dir / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    for case in picks:
        reference_path = artifact_dir / f"{case['case_id']}_reference.png"
        save_preview(case["reference"], reference_path)
        edge_reference_path = artifact_dir / f"{case['case_id']}_reference_edges.png"
        save_preview(case["reference_edges"], edge_reference_path)
        for result in case["results"]:
            if result["algorithm"] != case["best_attack_algorithm"]:
                continue
            attack_path = artifact_dir / f"{case['case_id']}_{result['algorithm']}.png"
            save_preview(result["image"], attack_path)
            case["best_attack_image"] = str(attack_path)
            case["reference_image"] = str(reference_path)
            case["reference_edges_image"] = str(edge_reference_path)


def scrub_images(cases: list[dict]) -> list[dict]:
    cleaned = []
    for case in cases:
        cleaned_results = []
        for result in case["results"]:
            cleaned_results.append(
                {
                    "algorithm": result["algorithm"],
                    "seconds": result["seconds"],
                    "metrics": result["metrics"],
                    "recovery": result["recovery"],
                }
            )
        cleaned_case = {
            key: value
            for key, value in case.items()
            if key not in {"results", "reference", "reference_edges"}
        }
        cleaned_case["results"] = cleaned_results
        cleaned.append(cleaned_case)
    return cleaned


def write_csv(cases: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "case_id",
                "grain",
                "speed",
                "feather",
                "text_drift",
                "text_drift_speed",
                "seed",
                "best_attack_algorithm",
                "best_recoverability_score",
                "best_attack_seconds",
                "generation_seconds",
                "best_window_start_frame",
                "best_window_end_frame_exclusive",
                "best_window_frame_count",
            ],
        )
        writer.writeheader()
        for case in sorted(cases, key=lambda item: item["best_recoverability_score"]):
            writer.writerow(
                {
                    "case_id": case["case_id"],
                    "grain": case["params"]["grain"],
                    "speed": case["params"]["speed"],
                    "feather": case["params"]["feather"],
                    "text_drift": case["params"]["text_drift"],
                    "text_drift_speed": case["params"]["text_drift_speed"],
                    "seed": case["params"]["seed"],
                    "best_attack_algorithm": case["best_attack_algorithm"],
                    "best_recoverability_score": case["best_recoverability_score"],
                    "best_attack_seconds": case["best_attack_seconds"],
                    "generation_seconds": case["generation_seconds"],
                    "best_window_start_frame": case["best_window"]["start_frame"],
                    "best_window_end_frame_exclusive": case["best_window"]["end_frame_exclusive"],
                    "best_window_frame_count": case["best_window"]["frame_count"],
                }
            )


def print_summary(cases: list[dict]) -> None:
    print(
        f"{'case_id':<10} {'score':>8} {'attack':<18} {'window':<11} {'attack_s':>9} {'gen_s':>8}  params"
    )
    for case in sorted(cases, key=lambda item: item["best_recoverability_score"]):
        params = case["params"]
        window = case["best_window"]
        window_label = f"{window['start_frame']}:{window['end_frame_exclusive'] - 1}"
        print(
            f"{case['case_id']:<10} "
            f"{case['best_recoverability_score']:>8.4f} "
            f"{case['best_attack_algorithm']:<18} "
            f"{window_label:<11} "
            f"{case['best_attack_seconds']:>9.4f} "
            f"{case['generation_seconds']:>8.4f}  "
            f"grain={params['grain']} speed={params['speed']} feather={params['feather']} "
            f"drift={params['text_drift']} drift_speed={params['text_drift_speed']}"
        )


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cases = [analyze_case(case, args) for case in build_case_grid(args)]
    save_ranked_artifacts(cases, output_dir, args.save_top_k)

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
        "analysis": {
            "algorithms": args.algorithms,
            "downscale": args.downscale,
            "block_size": args.block_size,
            "search_radius": args.search_radius,
            "pair_step": args.pair_step,
            "max_pairs": args.max_pairs,
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
