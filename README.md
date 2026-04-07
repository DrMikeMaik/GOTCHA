# GOTCHA

**G**liding **O**ptical **T**rick to **C**hallenge **H**umans vs **A**lgorithms

GOTCHA is a tiny single-file proof of concept: a word is hidden inside moving noise, and humans can usually read it almost immediately while current general-purpose AI systems and simpler image-processing approaches can struggle.

Each individual frame looks like random noise. A screenshot is not enough to reveal the word; the text only emerges across time through motion.

This is a fun perception demo, not a serious CAPTCHA product, but it can also be thought of as a way to send very short human-readable messages in a format that does not survive screenshots.

## Example

Direct MP4 link: [assets/secret.mp4](assets/secret.mp4)

<video src="https://github.com/user-attachments/assets/094cc537-18a5-4854-b6d2-7a48b80a8a9f" controls muted playsinline width="720">
  Your browser does not support embedded video. Use the direct link above.
</video>

<details>
<summary>Reveal</summary>

**TIMBER**

</details>

## How it works

- The background is random noise moving in one direction.
- The text is made from a different noise pattern moving in another direction.
- The whole word can drift slowly around the frame.
- In any single frame there is no obvious screenshot-friendly text to read.
- Over time, motion makes the text pop out for humans.

## Why it is interesting

The effect leans on motion segmentation. Humans are very good at grouping pixels that move together, even when the static image looks like nonsense. That makes this a neat toy example of perception that is easy for people and awkward for a lot of naive algorithms and current off-the-shelf AI systems.

## Running it

With Poetry:

```bash
poetry install
poetry run python text_noise_video.py --text TIMBER --output timber.mp4
```

Without Poetry:

```bash
pip install numpy pillow imageio imageio-ffmpeg
python text_noise_video.py --text TIMBER --output timber.mp4
```

See all options with:

```bash
python text_noise_video.py --help
```

Useful flags:

- `--gif` to write a GIF instead of MP4
- `--seed` for reproducible output
- `--grain` to make the noise coarser or finer
- `--text-drift` and `--text-drift-speed` to change how much the word wanders
- `--border-width` to add an outline around the word
- `--border-style` to choose `solid`, per-pixel `invert`, or opposite-flow `motion` border rendering
- `--border-color` to choose a simple named outline color such as `black` or `red` for `solid` borders only
- `--border-speed`, `--border-grain`, and `--border-strength` to tune `motion` borders
- `--font` to use a specific `.ttf` or `.otf` file

Border modes:

- `solid`: draws a flat-color outline using `--border-color`
- `invert`: draws an outline where each border pixel is inverted from nearby text pixels
- `motion`: draws the outline from a third noise field that moves vertically opposite to the text layer

Examples:

```bash
# Solid black border
poetry run python text_noise_video.py --text TIMBER --border-width 6 --border-style solid --border-color black --output timber_solid_border.mp4

# Per-pixel inverted border
poetry run python text_noise_video.py --text TIMBER --border-width 1 --border-style invert --output timber_invert_border.mp4

# Motion border using a third noise field
poetry run python text_noise_video.py --text TIMBER --grain 3 --border-width 2 --border-style motion --border-speed 3 --border-grain 3 --border-strength 1.0 --output timber_motion_border.mp4
```

Motion border notes:

- The motion border uses its own vertical noise field, moving opposite to the text noise.
- `--border-speed` defaults to `--speed` if omitted.
- `--border-grain` defaults to `--grain` if omitted.
- For the cleanest paused frames, keep `--border-grain` equal to `--grain` so the border stays on the same noise grid as the rest of the image.
- `--border-strength` controls how strongly the border layer replaces the underlying frame.

## Attack bench

There is now a companion script for trying reconstruction attacks against a generated clip and measuring how useful they are.

It currently runs a small set of lightweight attacks:

- `mean`: temporal average
- `stddev`: temporal standard deviation
- `delta_energy`: mean absolute frame-to-frame change
- `pca1`: first principal dynamic component
- `block_flow_angle`: block-matching motion-angle visualization

Each run writes:

- one PNG per algorithm
- a `report.json` file with runtime and simple image metrics
- a terminal summary table so you can compare methods quickly

Example:

```bash
python attack_bench.py assets/secret.mp4 --output-dir attack_runs/secret
```

For faster iteration on large videos, start with a smaller analysis pass:

```bash
python attack_bench.py assets/secret.mp4 \
  --downscale 0.25 \
  --max-frames 90 \
  --max-pairs 18 \
  --output-dir attack_runs/quick
```

Short windows can be stronger than whole-clip aggregation because text drift smears the signal over time. To search for the best 20-frame slice:

```bash
python attack_bench.py assets/secret.mp4 \
  --window-size 20 \
  --window-stride 1 \
  --include-full-window \
  --output-dir attack_runs/windowed
```

The metrics in `report.json` are heuristic, not OCR accuracy scores. The useful signal is the combination of:

- runtime
- the saved attack image
- contrast and edge metrics in the report

That makes it easier to test whether a parameter tweak genuinely hardens the effect or just changes its look.

## Attack sweep

For generator-side hardening, there is also a parameter sweep tool that generates multiple variants, runs the attack set against each one, and ranks cases by recoverability.

Unlike the standalone benchmark, the sweep knows the ground-truth text mask because it generated the clip itself. That means it can score each attack result against the expected text region instead of relying only on visual inspection.

Example:

```bash
python attack_sweep.py \
  --text TIMBER \
  --grains 2,3,4 \
  --feathers 0.75,1.25,2.0 \
  --text-drifts 80,200,320 \
  --text-drift-speeds 0.08,0.16 \
  --output-dir sweep_runs/timber
```

The sweep writes:

- `report.json` with per-case and per-attack metrics
- `summary.csv` for quick sorting
- preview images for the hardest and easiest cases by default

You can also score short-window attacks during the sweep:

```bash
python attack_sweep.py \
  --text TIMBER \
  --grains 2,3,4 \
  --text-drifts 80,200,320 \
  --window-size 20 \
  --window-stride 1 \
  --include-full-window \
  --output-dir sweep_runs/timber_windowed
```

Lower `best_recoverability_score` means the strongest configured attack had a harder time isolating the text, after considering the configured frame windows.

## Notes

- This is intentionally small and self-contained.
- It is best treated as a demo or toy experiment.
- It is fair to describe this as AI-resistant today, not AI-proof forever.
- Stronger computer vision pipelines may still recover the text.

## Inspiration

This project was directly inspired by [this YouTube video](https://www.youtube.com/watch?v=RNhiT-SmR1Q).

## License

[MIT](LICENSE)
