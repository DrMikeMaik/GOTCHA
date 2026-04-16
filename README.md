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
and realized that the algorithm only required two frames to retrieve
the secret message.

Instead of walking away, I spent the next few weeks trying to make it harder
to break. Some ideas were useless. Some were promising. None of them made it
truly bot-proof. But I learned a lot, and this repo tells the story of that
process. So if you want to take the journey follow the links.

## The Story

1. **[The Idea](docs/01-the-idea.md)** — A noise shader, orthogonal motion,
   and a confident first version.
2. **[Breaking It](docs/02-breaking-it.md)** — Someone cracked it publicly.
   Then I built the attack tools to understand exactly how broken it was.
3. **[Defending It](docs/03-defending.md)** — Shared motion palettes,
   phase-sliced reveals, and randomized scheduling made recovery harder—but
   not impossible.
4. **[What We Learned](docs/04-what-we-learned.md)** — Nothing is bot-proof.
   The interesting part was finding out why.
5. **[Running the Code](docs/05-running-the-code.md)** — Generate your own
   clips, run the attacks, and reproduce the experiments.

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

If you want the ranked montage workflow that sweeps many frame pairs for one
existing video, use:

```bash
poetry run python attack_pair_sweep.py hello.mp4 --output-dir sweep_runs/hello
```

That writes a ranked CSV/JSON summary plus `top12_montage.png` under
`sweep_runs/hello`.

Try the defense generator instead:

```bash
poetry run python generate_defense.py --random-digits --background-grain 8 --text-grain 16 --output defended.mp4
```

Attack that one and compare the results. Full flag reference in
[Running the Code](docs/05-running-the-code.md).

## Inspiration

This project was directly inspired by
[this YouTube video](https://www.youtube.com/watch?v=RNhiT-SmR1Q).

## License

[MIT](LICENSE)
