"""Diagnostic attack algorithms that were tried and didn't work.

These were part of the original attack benchmarking suite. They treat each
pixel independently and hope that the text region leaks some statistical
signature over time. In practice, none of them recovered readable text from
GOTCHA clips. The only attack that consistently worked was block_flow_angle
(tuned two-frame block matching), which remains in attack_bench.py.

These are preserved here for reference, not because they are useful.
"""

import numpy as np


def mean_attack(frames: np.ndarray) -> np.ndarray:
    """Temporal average. If the text region has a different brightness
    distribution it might show up. It didn't."""
    return frames.mean(axis=0)


def stddev_attack(frames: np.ndarray) -> np.ndarray:
    """Per-pixel temporal standard deviation to highlight motion-heavy
    regions."""
    return frames.std(axis=0)


def delta_energy_attack(frames: np.ndarray) -> np.ndarray:
    """Mean absolute difference between consecutive frames."""
    return np.mean(np.abs(np.diff(frames, axis=0)), axis=0)


def pca1_attack(frames: np.ndarray) -> np.ndarray:
    """First principal dynamic component extracted from the frame stack."""
    frame_count, height, width = frames.shape
    matrix = frames.reshape(frame_count, height * width).astype(np.float64)
    matrix = matrix - matrix.mean(axis=0, keepdims=True)
    _, _, right_vectors = np.linalg.svd(matrix, full_matrices=False)
    return np.abs(right_vectors[0]).reshape(height, width)
