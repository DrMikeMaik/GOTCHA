#!/usr/bin/env python3
"""Defense-oriented generator for GOTCHA.

This variant keeps the original generator intact and explores a different
tradeoff: preserve human readability while making simple reconstruction attacks
less direct. Frames are assembled from a shared palette of local motion fields,
and the text is revealed in phase-sliced groups instead of one coherent
text-shaped motion partition.
"""

from __future__ import annotations

import argparse
import itertools
import secrets
from pathlib import Path
from typing import Iterable

import numpy as np

from text_noise_video import (
    DEFAULT_DURATION,
    DEFAULT_FEATHER,
    DEFAULT_FONT_SIZE,
    DEFAULT_FPS,
    DEFAULT_GRAIN,
    DEFAULT_HEIGHT,
    DEFAULT_OUTPUT,
    DEFAULT_SEED,
    DEFAULT_TEXT,
    DEFAULT_TEXT_DRIFT,
    DEFAULT_TEXT_DRIFT_SPEED,
    DEFAULT_WIDTH,
    make_noise,
    make_text_mask,
    render_text_image,
    resolve_output_path,
    shift_mask,
    write_animation,
)

DEFAULT_TILE_SIZE = 12
DEFAULT_PALETTE = "-2,0;0,-2;2,0;0,2"
DEFAULT_TEXT_VECTOR_INDEX = 1
DEFAULT_PHASE_COUNT = 4
DEFAULT_PHASE_HOLD = 5
DEFAULT_ACTIVE_PHASES = 3
DEFAULT_BACKGROUND_CYCLE_STEP = 0
DEFAULT_BACKGROUND_CYCLE_HOLD = 12
DEFAULT_PHASE_MODE = "components"
DEFAULT_SCHEDULE_MODE = "randomized"
DEFAULT_SCHEDULE_SPAN = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a defense-oriented motion-noise clip using a shared palette "
            "of local motion vectors."
        )
    )
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--duration", type=float, default=DEFAULT_DURATION)
    parser.add_argument("--font-size", type=int, default=DEFAULT_FONT_SIZE)
    parser.add_argument("--grain", type=int, default=DEFAULT_GRAIN)
    parser.add_argument("--feather", type=float, default=DEFAULT_FEATHER)
    parser.add_argument("--text-drift", type=float, default=DEFAULT_TEXT_DRIFT)
    parser.add_argument("--text-drift-speed", type=float, default=DEFAULT_TEXT_DRIFT_SPEED)
    parser.add_argument("--tile-size", type=int, default=DEFAULT_TILE_SIZE)
    parser.add_argument(
        "--palette",
        default=DEFAULT_PALETTE,
        help='Semicolon-separated motion vectors, for example "-2,0;0,-2;2,0;0,2".',
    )
    parser.add_argument(
        "--text-vector-index",
        type=int,
        default=DEFAULT_TEXT_VECTOR_INDEX,
        help="Base palette index used to seed the text-phase vector cycle.",
    )
    parser.add_argument(
        "--phase-count",
        type=int,
        default=DEFAULT_PHASE_COUNT,
        help="Number of spatial text groups to cycle through.",
    )
    parser.add_argument(
        "--phase-hold",
        type=int,
        default=DEFAULT_PHASE_HOLD,
        help="Frames to hold each active text phase before advancing.",
    )
    parser.add_argument(
        "--active-phases",
        type=int,
        default=DEFAULT_ACTIVE_PHASES,
        help="How many text phases are active at once.",
    )
    parser.add_argument(
        "--background-cycle-step",
        type=int,
        default=DEFAULT_BACKGROUND_CYCLE_STEP,
        help="How many palette slots background tiles advance per cycle. 0 disables cycling.",
    )
    parser.add_argument(
        "--background-cycle-hold",
        type=int,
        default=DEFAULT_BACKGROUND_CYCLE_HOLD,
        help="Frames to hold each background palette update.",
    )
    parser.add_argument(
        "--phase-mode",
        choices=("bands", "components"),
        default=DEFAULT_PHASE_MODE,
        help="How to divide the text into reveal groups.",
    )
    parser.add_argument(
        "--schedule-mode",
        choices=("cycle", "randomized"),
        default=DEFAULT_SCHEDULE_MODE,
        help="How active text groups are scheduled over time.",
    )
    parser.add_argument(
        "--schedule-span",
        type=int,
        default=DEFAULT_SCHEDULE_SPAN,
        help="How many phase-hold steps a randomized reveal subset tends to persist.",
    )
    parser.add_argument(
        "--random-digits",
        action="store_true",
        help="Generate a random 4-digit secret internally instead of using --text.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--font", default=None)
    parser.add_argument("--gif", action="store_true")
    args = parser.parse_args()
    validate_args(args)
    args.palette_vectors = parse_palette(args.palette)
    if not 0 <= args.text_vector_index < len(args.palette_vectors):
        raise SystemExit("--text-vector-index must refer to a valid palette entry.")
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
    if args.feather < 0:
        raise SystemExit("Feather must be 0 or greater.")
    if args.text_drift < 0:
        raise SystemExit("Text drift must be 0 or greater.")
    if args.text_drift_speed < 0:
        raise SystemExit("Text drift speed must be 0 or greater.")
    if args.tile_size <= 0:
        raise SystemExit("Tile size must be greater than 0.")
    if args.phase_count <= 0:
        raise SystemExit("Phase count must be greater than 0.")
    if args.phase_hold <= 0:
        raise SystemExit("Phase hold must be greater than 0.")
    if args.active_phases <= 0:
        raise SystemExit("Active phases must be greater than 0.")
    if args.schedule_span <= 0:
        raise SystemExit("Schedule span must be greater than 0.")
    if args.background_cycle_hold <= 0:
        raise SystemExit("Background cycle hold must be greater than 0.")
    if not args.random_digits and not args.text.strip():
        raise SystemExit("Text cannot be empty.")


def parse_palette(raw: str) -> list[tuple[int, int]]:
    vectors: list[tuple[int, int]] = []
    for item in raw.split(";"):
        item = item.strip()
        if not item:
            continue
        parts = [part.strip() for part in item.split(",")]
        if len(parts) != 2:
            raise SystemExit(f"Invalid palette vector: {item}")
        dx, dy = (int(parts[0]), int(parts[1]))
        vectors.append((dx, dy))
    if not vectors:
        raise SystemExit("Palette must contain at least one motion vector.")
    return vectors


def resolve_render_text(args: argparse.Namespace) -> str:
    if args.random_digits:
        return f"{secrets.randbelow(10000):04d}"
    return args.text


def frame_count_from_args(args: argparse.Namespace) -> int:
    return max(1, int(round(args.duration * args.fps)))


def text_drift_offsets(
    frame_count: int,
    fps: int,
    drift: float,
    speed: float,
) -> Iterable[tuple[int, int]]:
    if drift <= 0 or speed <= 0:
        for _ in range(frame_count):
            yield 0, 0
        return

    phase_step = (2.0 * np.pi * speed) / fps
    for index in range(frame_count):
        phase = index * phase_step
        x_offset = int(round(np.sin(phase) * drift))
        y_offset = int(round(np.cos(phase * 0.73 + np.pi / 6.0) * drift * 0.45))
        yield x_offset, y_offset


def expand_tile_map(tile_values: np.ndarray, tile_size: int, height: int, width: int) -> np.ndarray:
    expanded = np.repeat(np.repeat(tile_values, tile_size, axis=0), tile_size, axis=1)
    return expanded[:height, :width]


def tile_mask_from_pixel_mask(mask: np.ndarray, tile_size: int) -> np.ndarray:
    height, width = mask.shape
    tiles_y = (height + tile_size - 1) // tile_size
    tiles_x = (width + tile_size - 1) // tile_size
    tile_mask = np.zeros((tiles_y, tiles_x), dtype=bool)

    for tile_y in range(tiles_y):
        y0 = tile_y * tile_size
        y1 = min(height, y0 + tile_size)
        for tile_x in range(tiles_x):
            x0 = tile_x * tile_size
            x1 = min(width, x0 + tile_size)
            tile_mask[tile_y, tile_x] = bool(np.max(mask[y0:y1, x0:x1]) > 0.05)

    return tile_mask


def build_text_phase_groups(
    text_tile_mask: np.ndarray,
    phase_count: int,
    phase_mode: str,
) -> np.ndarray:
    if phase_mode == "components":
        return build_component_phase_groups(text_tile_mask, phase_count)

    groups = np.full(text_tile_mask.shape, -1, dtype=np.int16)
    active_tiles = np.argwhere(text_tile_mask)
    if active_tiles.size == 0:
        return groups

    min_y, min_x = active_tiles.min(axis=0)
    max_y, max_x = active_tiles.max(axis=0)
    span_x = max(1, max_x - min_x + 1)
    span_y = max(1, max_y - min_y + 1)
    band_width = max(1.0, (span_x + 0.45 * span_y) / phase_count)

    for tile_y, tile_x in active_tiles:
        # Use broad diagonal bands so each phase keeps larger, more legible
        # stroke fragments while still avoiding one coherent full-word mask.
        projected = (tile_x - min_x) + 0.45 * (tile_y - min_y)
        groups[tile_y, tile_x] = int(np.floor(projected / band_width)) % phase_count
    return groups


def collect_text_components(
    text_tile_mask: np.ndarray,
) -> list[list[tuple[int, int]]]:
    visited = np.zeros(text_tile_mask.shape, dtype=bool)
    components: list[list[tuple[int, int]]] = []
    neighbors = (
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    )

    height, width = text_tile_mask.shape
    for tile_y in range(height):
        for tile_x in range(width):
            if visited[tile_y, tile_x] or not text_tile_mask[tile_y, tile_x]:
                continue
            stack = [(tile_y, tile_x)]
            visited[tile_y, tile_x] = True
            component: list[tuple[int, int]] = []
            while stack:
                cy, cx = stack.pop()
                component.append((cy, cx))
                for dy, dx in neighbors:
                    ny = cy + dy
                    nx = cx + dx
                    if ny < 0 or ny >= height or nx < 0 or nx >= width:
                        continue
                    if visited[ny, nx] or not text_tile_mask[ny, nx]:
                        continue
                    visited[ny, nx] = True
                    stack.append((ny, nx))
            components.append(component)
    return components


def order_text_components(
    components: list[list[tuple[int, int]]],
) -> list[list[tuple[int, int]]]:
    # Assign phases in reading order so the reveal tracks whole letters instead
    # of diagonal slices through multiple glyphs.
    return sorted(
        components,
        key=lambda component: (
            min(tile_y for tile_y, _ in component),
            min(tile_x for _, tile_x in component),
        ),
    )


def build_component_phase_groups(
    text_tile_mask: np.ndarray,
    phase_count: int,
) -> np.ndarray:
    groups = np.full(text_tile_mask.shape, -1, dtype=np.int16)
    ordered_components = order_text_components(collect_text_components(text_tile_mask))
    for index, component in enumerate(ordered_components):
        phase_group = index % phase_count
        for tile_y, tile_x in component:
            groups[tile_y, tile_x] = phase_group
    return groups


def apply_phase_overrides(
    base_tile_labels: np.ndarray,
    phase_groups: np.ndarray,
    palette_size: int,
    active_groups: tuple[int, ...],
    text_vector_index: int,
) -> np.ndarray:
    frame_tile_labels = np.array(base_tile_labels, copy=True)
    for group_index in active_groups:
        group_mask = phase_groups == group_index
        if not np.any(group_mask):
            continue
        frame_tile_labels[group_mask] = (text_vector_index + group_index) % palette_size
    return frame_tile_labels


def build_cycle_schedule(
    groups: list[int],
    active_count: int,
    step_count: int,
) -> list[tuple[int, ...]]:
    if not groups:
        return [tuple()] * step_count
    if active_count >= len(groups):
        return [tuple(groups)] * step_count

    schedule: list[tuple[int, ...]] = []
    for step_index in range(step_count):
        start = step_index % len(groups)
        active_groups = tuple(groups[(start + offset) % len(groups)] for offset in range(active_count))
        schedule.append(active_groups)
    return schedule


def build_randomized_schedule(
    rng: np.random.Generator,
    groups: list[int],
    active_count: int,
    step_count: int,
    schedule_span: int,
) -> list[tuple[int, ...]]:
    if not groups:
        return [tuple()] * step_count
    if active_count >= len(groups):
        return [tuple(groups)] * step_count

    if active_count == len(groups) - 1:
        omission_weights = rng.random(len(groups)) + 0.35
        schedule: list[tuple[int, ...]] = []
        previous_omitted: int | None = None
        previous_previous_omitted: int | None = None

        while len(schedule) < step_count:
            candidate_indices = [index for index, group in enumerate(groups) if group != previous_omitted]
            weights = np.asarray([omission_weights[index] for index in candidate_indices], dtype=np.float64)
            if previous_previous_omitted is not None:
                for position, index in enumerate(candidate_indices):
                    if groups[index] == previous_previous_omitted:
                        weights[position] *= 0.45
            weights = weights / weights.sum()
            chosen_index = int(rng.choice(candidate_indices, p=weights))
            omitted_group = groups[chosen_index]
            active_groups = tuple(group for group in groups if group != omitted_group)
            run_length = int(rng.integers(1, schedule_span + 1))
            schedule.extend([active_groups] * run_length)
            previous_previous_omitted = previous_omitted
            previous_omitted = omitted_group

        return schedule[:step_count]

    combinations = [tuple(combo) for combo in itertools.combinations(groups, active_count)]
    schedule = [combinations[int(rng.integers(0, len(combinations)))]]
    while len(schedule) < step_count:
        previous = schedule[-1]
        ranked: list[tuple[float, tuple[int, ...]]] = []
        for combo in combinations:
            if combo == previous and len(combinations) > 1:
                continue
            overlap = len(set(combo) & set(previous))
            score = overlap + float(rng.random()) * 0.25
            if len(schedule) >= 2 and combo == schedule[-2]:
                score -= 0.75
            ranked.append((score, combo))
        ranked.sort(key=lambda item: item[0], reverse=True)
        top_score = ranked[0][0]
        top_combos = [combo for score, combo in ranked if score >= top_score - 0.2]
        chosen_combo = top_combos[int(rng.integers(0, len(top_combos)))]
        run_length = int(rng.integers(1, schedule_span + 1))
        schedule.extend([chosen_combo] * run_length)
    return schedule[:step_count]


def build_phase_schedule(
    rng: np.random.Generator,
    phase_groups: np.ndarray,
    active_phases: int,
    step_count: int,
    schedule_mode: str,
    schedule_span: int,
) -> list[tuple[int, ...]]:
    used_groups = sorted(int(group) for group in np.unique(phase_groups) if group >= 0)
    active_count = min(active_phases, len(used_groups))
    if schedule_mode == "cycle":
        return build_cycle_schedule(used_groups, active_count, step_count)
    return build_randomized_schedule(rng, used_groups, active_count, step_count, schedule_span)


def shifted_field(field: np.ndarray, dx: int, dy: int, frame_index: int) -> np.ndarray:
    return np.roll(field, shift=(dy * frame_index, dx * frame_index), axis=(0, 1))


def build_palette_fields(
    rng: np.random.Generator,
    height: int,
    width: int,
    grain: int,
    count: int,
) -> np.ndarray:
    fields = [make_noise(rng, height, width, grain) for _ in range(count)]
    return np.stack(fields, axis=0).astype(np.float32)


def compose_palette_frame(
    shifted_fields: np.ndarray,
    label_map: np.ndarray,
) -> np.ndarray:
    return np.take_along_axis(shifted_fields, label_map[None, :, :], axis=0)[0]


def shift_phase_groups(
    phase_groups: np.ndarray,
    phase_count: int,
    tile_size: int,
    height: int,
    width: int,
    x_offset: int,
    y_offset: int,
) -> np.ndarray:
    if x_offset == 0 and y_offset == 0:
        return phase_groups

    shifted_phase_groups = np.full_like(phase_groups, -1)
    expanded_phase_groups = expand_tile_map(
        np.where(phase_groups >= 0, phase_groups + 1, 0),
        tile_size,
        height,
        width,
    )
    shifted_group_pixels = shift_mask(expanded_phase_groups.astype(np.float32), x_offset, y_offset)
    shifted_group_tiles = tile_mask_from_pixel_mask(shifted_group_pixels, tile_size)

    for group_index in range(phase_count):
        group_pixels = shift_mask(
            expand_tile_map((phase_groups == group_index).astype(np.float32), tile_size, height, width),
            x_offset,
            y_offset,
        )
        group_tiles = tile_mask_from_pixel_mask(group_pixels, tile_size)
        shifted_phase_groups[group_tiles] = group_index

    shifted_phase_groups[~shifted_group_tiles] = -1
    return shifted_phase_groups


def resolve_background_tile_labels(
    base_tile_labels: np.ndarray,
    background_phase_steps: np.ndarray,
    palette_size: int,
    frame_index: int,
    cycle_step: int,
    cycle_hold: int,
) -> np.ndarray:
    background_tile_labels = np.array(base_tile_labels, copy=True)
    if cycle_step == 0:
        return background_tile_labels

    background_phase_index = frame_index // cycle_hold
    return (
        background_tile_labels
        + (background_phase_index * background_phase_steps * cycle_step)
    ) % palette_size


def build_animation(args: argparse.Namespace) -> Iterable[np.ndarray]:
    frame_count = frame_count_from_args(args)
    phase_step_count = (frame_count + args.phase_hold - 1) // args.phase_hold
    rng = np.random.default_rng(args.seed)
    palette = args.palette_vectors
    field_stack = build_palette_fields(rng, args.height, args.width, args.grain, len(palette))

    tiles_y = (args.height + args.tile_size - 1) // args.tile_size
    tiles_x = (args.width + args.tile_size - 1) // args.tile_size
    base_tile_labels = rng.integers(0, len(palette), size=(tiles_y, tiles_x), endpoint=False)
    background_phase_steps = rng.choice(np.asarray([-1, 1], dtype=np.int16), size=(tiles_y, tiles_x))

    text_image = render_text_image(args.text, args.width, args.height, args.font_size, args.font)
    base_text_mask = make_text_mask(text_image, args.feather)
    base_text_tile_mask = tile_mask_from_pixel_mask(base_text_mask, args.tile_size)
    text_phase_groups = build_text_phase_groups(
        base_text_tile_mask,
        args.phase_count,
        args.phase_mode,
    )
    phase_schedule = build_phase_schedule(
        rng=rng,
        phase_groups=text_phase_groups,
        active_phases=args.active_phases,
        step_count=phase_step_count,
        schedule_mode=args.schedule_mode,
        schedule_span=args.schedule_span,
    )

    for frame_index, (x_offset, y_offset) in enumerate(
        text_drift_offsets(frame_count, args.fps, args.text_drift, args.text_drift_speed)
    ):
        shifted_phase_groups = shift_phase_groups(
            phase_groups=text_phase_groups,
            phase_count=args.phase_count,
            tile_size=args.tile_size,
            height=args.height,
            width=args.width,
            x_offset=x_offset,
            y_offset=y_offset,
        )

        schedule_index = min(frame_index // args.phase_hold, len(phase_schedule) - 1)
        background_tile_labels = resolve_background_tile_labels(
            base_tile_labels=base_tile_labels,
            background_phase_steps=background_phase_steps,
            palette_size=len(palette),
            frame_index=frame_index,
            cycle_step=args.background_cycle_step,
            cycle_hold=args.background_cycle_hold,
        )
        frame_tile_labels = apply_phase_overrides(
            base_tile_labels=background_tile_labels,
            phase_groups=shifted_phase_groups,
            palette_size=len(palette),
            active_groups=phase_schedule[schedule_index],
            text_vector_index=args.text_vector_index,
        )
        frame_label_map = expand_tile_map(frame_tile_labels, args.tile_size, args.height, args.width)
        shifted_fields = np.stack(
            [shifted_field(field_stack[index], dx, dy, frame_index) for index, (dx, dy) in enumerate(palette)],
            axis=0,
        )
        frame = compose_palette_frame(shifted_fields, frame_label_map)
        rgb = np.repeat(frame[:, :, None], 3, axis=2) * 255.0
        yield np.clip(rgb, 0.0, 255.0).astype(np.uint8)


def main() -> None:
    args = parse_args()
    output_path = resolve_output_path(args.output, args.gif)
    args.text = resolve_render_text(args)
    frames = build_animation(args)
    write_animation(frames, output_path, args.fps, args.gif)
    frame_count = frame_count_from_args(args)
    print(f"Wrote {frame_count} frames to {output_path}")


if __name__ == "__main__":
    main()
