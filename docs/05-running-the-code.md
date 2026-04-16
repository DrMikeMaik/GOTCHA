# Running the Code

## Dependencies

```bash
poetry install
```

## Baseline Generator

The baseline generator creates a noise animation where the background slides
horizontally and the text region moves vertically. The text pops out over time
because humans are good at grouping motion, but a single frame looks like
noise.

```bash
poetry run python generate_baseline.py --text TIMBER --output timber.mp4
```

### Useful flags

| Flag | What it does |
|------|-------------|
| `--grain` | Noise block size in pixels (smaller = finer noise) |
| `--duration` | Clip length in seconds |
| `--speed` | How fast the noise slides (pixels per frame) |
| `--font-size` | Text size in pixels |
| `--text-drift` | Total pixels the word oscillates over time |
| `--text-drift-speed` | How fast the word oscillates (cycles per second) |
| `--feather` | Gaussian blur radius on the text mask edge |
| `--font` | Path to a `.ttf` or `.otf` font file |
| `--seed` | Fix the random seed for reproducibility |
| `--width`, `--height` | Output resolution (default 1920x1080) |
| `--fps` | Frame rate (default 30) |

Run `poetry run python generate_baseline.py --help` for the full list.

## Defense Generator

The defense variant tries to eliminate the clean text-shaped motion partition
that the baseline leaks. It uses a shared motion palette across the whole
frame, and reveals the text in phase-sliced groups instead of one coherent
region.

```bash
poetry run python generate_defense.py --text TIMBER --output timber_defense.mp4
```

To generate a hidden 5-digit code (the text is not printed to stdout):

```bash
poetry run python generate_defense.py \
  --random-digits \
  --grain 9 \
  --duration 10 \
  --output secret_digits.mp4
```

### Useful flags

| Flag | What it does |
|------|-------------|
| `--random-digits` | Generate a random 5-digit code internally |
| `--grain` | Noise block size (default 3) |
| `--text-grain` | Separate grain for text region (defaults to `--grain`) |
| `--tile-size` | Motion tile size in pixels (default 12) |
| `--palette` | Motion vector palette, e.g. `"-2,0;0,-2;2,0;0,2"` |
| `--phase-mode` | `components` (whole digits) or `bands` (diagonal slices) |
| `--phase-count` | Number of reveal groups |
| `--active-phases` | How many groups are visible at once |
| `--phase-hold` | Frames each phase pattern holds before rotating |
| `--schedule-mode` | `randomized` (default) or `cycle` (deterministic) |
| `--schedule-span` | How many windows a visible subset persists |
| `--background-cycle-step` | Palette rotation step for background (0 = off) |
| `--background-cycle-hold` | Frames between background palette rotations |

Run `poetry run python generate_defense.py --help` for the full list.

## Attack Bench

Run the block-flow reconstruction attack against a clip. By default it
analyzes a single pair of consecutive frames — which is all block-flow needs
to recover the baseline.

```bash
poetry run python attack_bench.py video.mp4 --output-dir attack_runs/test
```

### Useful flags

| Flag | What it does |
|------|-------------|
| `--downscale` | Scale factor before analysis (default 0.25) |
| `--block-size` | Block size for block matching (default 8) |
| `--search-radius` | Search radius for block matching (default 3) |
| `--pair-step` | Frame gap between the two frames in a pair (default 1) |
| `--max-pairs` | Frame pairs to average (default 1) |
| `--window-size` | Sliding window length in frames (0 = full clip) |
| `--window-stride` | Stride between windows (default 1) |
| `--include-full-window` | Also run full-clip analysis alongside windows |

Short windows can help against the defense generator:

```bash
poetry run python attack_bench.py video.mp4 \
  --window-size 20 \
  --window-stride 1 \
  --include-full-window \
  --output-dir attack_runs/windowed
```

## Pair Sweep

Run a ranked sweep across many pair gaps and frame windows for one existing
video. This is the montage workflow that checks many candidate two-frame
attacks and saves the strongest hits.

```bash
poetry run python attack_pair_sweep.py video.mp4 \
  --pair-steps 1,2,3,4,5,6 \
  --output-dir sweep_runs/video_pairs
```

By default this uses `window_size = pair_step + 1`, so each candidate is a
single pair window. It writes ranked PNGs, a `top12_montage.png`, `summary.csv`,
and a JSON report.

## Parameter Sweep

Generate many parameter variants of a clip, attack them all, and rank the
render settings by attack resistance. Useful for finding the weakest and strongest
parameter combinations.

```bash
poetry run python attack_resistance_sweep.py \
  --text TIMBER \
  --grains 2,3,4 \
  --text-drifts 80,200,320 \
  --window-size 20 \
  --window-stride 1 \
  --include-full-window \
  --output-dir sweep_runs/timber_windowed
```

Output is a CSV ranking cases by how well the attack recovered the text.
