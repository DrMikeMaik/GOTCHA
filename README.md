# GOTCHA

**G**liding **O**ptical **T**rick to **C**hallenge **H**umans vs **A**lgorithms

I saw a cool video about a video game noise shader and thought: what if
overlapping random noise masks with orthogonal movement could hide a secret
number, readable only by humans? A single frame looks like pure static. But
when the video plays, your visual system groups the motion and the number pops
out.

<video src="assets/version_1.mp4" controls muted playsinline width="720">
  Your browser does not support embedded video.
</video>

<details>
<summary>Reveal</summary>

**1544**

</details>

I shared it publicly and confidently claimed that biology still has a leg up
on technology. Within hours, someone in the comments cracked it using
block-matching optical flow. I personally dug into this attack vector
and realized that the algorithm only required **two frames** to retrieve
the secret message.

![Version 1 attack result](assets/version_1_attack.png)

Instead of walking away, I spent the next few weeks trying to make it harder
to break. The second version adds one more digit and 
never shows all digits at once, so no single
frame pair can recover the full secret. But sweeping across all pairs still
lets the bot piece together the whole number.

<video src="assets/version_2.mp4" controls muted playsinline width="720">
  Your browser does not support embedded video.
</video>

<details>
<summary>Reveal</summary>

**80511**

</details>

![Version 2 attack result](assets/version_2_attack.png)

The third version uses different grain sizes for the text and background.
The mismatch is actually pleasant for a human viewer — the difference in
pixel size makes edges easy to perceive. But individual frames are hard to
OCR even though you can almost see the digits, and the background palette
cycling completely defeats block-flow angle analysis.

<video src="assets/version_3.mp4" controls muted playsinline width="720">
  Your browser does not support embedded video.
</video>

<details>
<summary>Reveal</summary>

**86217**

</details>

The attack recovered nothing — pure noise.

![Version 3 attack result](assets/version_3_attack.png)

Can it be broken? Absolutely — just not by these algorithms. Single-frame
analysis with OCR would probably be a more effective angle, and I expect
someone will point that out eventually. But I learned a lot, and this repo
tells the story of that process. If you want to take the journey follow
the links.

## Tools

| File | What it does |
|------|-------------|
| `generate_baseline.py` | Original generator — two sliding noise fields. Trivially crackable. |
| `generate_defense.py` | Defense generator — tile-based motion palette with phase-sliced reveals. |
| `attack_bench.py` | Run the block-flow attack on a single video file. |
| `attack_pair_sweep.py` | Sweep consecutive frame pairs across a video and rank the best attacks. |
| `attack_resistance_sweep.py` | Generate a grid of defense settings, attack each, and rank by resistance. Saves videos for the strongest and weakest cases. |

## Try It Yourself

```bash
poetry install
```

Generate a clip with the baseline generator (the one that got cracked):

```bash
poetry run python generate_baseline.py --text HELLO --grain 16 --output hello.mp4
```

Now attack it:

```bash
poetry run python attack_bench.py hello.mp4 --output-dir attack_runs/hello
```

Open `attack_runs/hello/block_flow_angle.png` — the word is right there.

Sweep all frame pairs for a ranked montage:

```bash
poetry run python attack_pair_sweep.py hello.mp4 --output-dir sweep_runs/hello
```

Try the defense generator instead:

```bash
poetry run python generate_defense.py --random-digits --background-grain 8 --text-grain 16 --output defended.mp4
```

Attack that one and compare the results.

<details>
<summary>Flag reference</summary>

### generate_baseline.py

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

### generate_defense.py

| Flag | What it does |
|------|-------------|
| `--random-digits` | Generate a random 5-digit code internally |
| `--grain` | Noise block size (default 3) |
| `--background-grain` | Separate grain for background (defaults to `--grain`) |
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

### attack_bench.py

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

### attack\_pair\_sweep.py

| Flag | What it does |
|------|-------------|
| `--pair-steps` | Comma-separated pair gaps to sweep (default 1) |
| `--window-size` | Frame window length (0 = pair_step + 1) |
| `--window-stride` | Stride between candidate windows (default 1) |
| `--include-full-window` | Also evaluate the full clip |
| `--top-k` | How many top results to save and montage (default 12) |

### attack\_resistance\_sweep.py

| Flag | What it does |
|------|-------------|
| `--background-grains` | Comma-separated background grain values to sweep |
| `--text-grains` | Comma-separated text grain values to sweep |
| `--tile-sizes` | Comma-separated tile sizes to sweep |
| `--phase-counts` | Comma-separated phase counts to sweep |
| `--active-phases-values` | Comma-separated active phase counts to sweep |
| `--phase-modes` | Comma-separated phase modes to sweep |
| `--schedule-modes` | Comma-separated schedule modes to sweep |
| `--save-top-k` | Save videos for top K hardest and easiest cases (default 3) |

All tools support `--help` for the full flag list.

</details>

## Inspiration

This project was directly inspired by
[this YouTube video](https://www.youtube.com/watch?v=RNhiT-SmR1Q).

## License

[MIT](LICENSE)
