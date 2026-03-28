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
- `--font` to use a specific `.ttf` or `.otf` file

## Notes

- This is intentionally small and self-contained.
- It is best treated as a demo or toy experiment.
- It is fair to describe this as AI-resistant today, not AI-proof forever.
- Stronger computer vision pipelines may still recover the text.

## Inspiration

This project was directly inspired by [this YouTube video](https://www.youtube.com/watch?v=RNhiT-SmR1Q).

## License

[MIT](LICENSE)
