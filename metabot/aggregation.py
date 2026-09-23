"""Robust aggregation of ensemble members.

* Binary: trimmed mean of log-odds (a robust geometric mean of odds), optional
  Platt recalibration, then clipping away from 0 and 1.
* Multiple choice: per-option trimmed mean of probabilities (a robust linear
  opinion pool), floor, renormalise.
* Numeric/date/discrete: point-wise trimmed mean of the members' CDFs on the
  common grid. Order statistics of non-decreasing functions are
  non-decreasing, so the result stays a valid CDF and keeps the members'
  minimum/maximum step properties.

Trimming: k = max(1, floor(trim * n)) members are dropped at each end when
n >= 3 (so n = 3 or 4 gives the median). With n <= 2 we take the plain mean.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def trim_count(n: int, trim: float) -> int:
    if n <= 2:
        return 0
    k = max(1, int(math.floor(trim * n)))
    return min(k, (n - 1) // 2)


def trimmed_mean(values: Sequence[float], trim: float = 0.2) -> float:
    arr = np.sort(np.asarray(values, dtype=float))
    if arr.size == 0:
        raise ValueError("No values to aggregate")
    k = trim_count(arr.size, trim)
    return float(arr[k : arr.size - k].mean())


def trimmed_mean_rows(matrix: np.ndarray, trim: float = 0.2) -> np.ndarray:
    """Column-wise trimmed mean of a (members x points) matrix."""
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        raise ValueError("Expected a non-empty 2-D matrix")
    ordered = np.sort(matrix, axis=0)
    k = trim_count(matrix.shape[0], trim)
    return ordered[k : matrix.shape[0] - k].mean(axis=0)


def _logit(p: float) -> float:
    return math.log(p / (1.0 - p))


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


def aggregate_binary(
    probabilities: Sequence[float],
    trim: float = 0.2,
    clip_low: float = 0.02,
    clip_high: float = 0.98,
    platt_a: float = 1.0,
    platt_b: float = 0.0,
) -> float:
    if not probabilities:
        raise ValueError("No binary forecasts to aggregate")
    logits = [_logit(min(max(float(p), 0.005), 0.995)) for p in probabilities]
    pooled = trimmed_mean(logits, trim)
    calibrated = _sigmoid(platt_a * pooled + platt_b)
    return float(min(max(calibrated, clip_low), clip_high))


def normalize_with_floor(probs: dict[str, float], floor: float) -> dict[str, float]:
    """Floor every option at ``floor`` and renormalise to sum to 1."""
    names = list(probs)
    n = len(names)
    if n == 0:
        raise ValueError("No options")
    if floor * n >= 1.0:
        return {name: 1.0 / n for name in names}
    values = np.array([max(0.0, float(probs[name])) for name in names])
    total = values.sum()
    values = values / total if total > 0 else np.full(n, 1.0 / n)
    # Water-filling: options below the floor are set to it and the rest are
    # scaled down proportionally, repeated until stable.
    fixed = np.zeros(n, dtype=bool)
    for _ in range(n):
        low = (values < floor) & ~fixed
        if not np.any(low):
            break
        fixed |= low
        free_mass = 1.0 - floor * fixed.sum()
        free_total = values[~fixed].sum()
        values[fixed] = floor
        if free_total > 0:
            values[~fixed] = values[~fixed] / free_total * free_mass
    values = values / values.sum()
    return {name: float(v) for name, v in zip(names, values)}


def aggregate_multiple_choice(
    member_probs: Sequence[dict[str, float]],
    options: Sequence[str],
    trim: float = 0.2,
    floor: float = 0.01,
) -> dict[str, float]:
    if not member_probs:
        raise ValueError("No multiple-choice forecasts to aggregate")
    pooled: dict[str, float] = {}
    for option in options:
        pooled[option] = trimmed_mean([m.get(option, 0.0) for m in member_probs], trim)
    return normalize_with_floor(pooled, floor)


def aggregate_cdfs(cdfs: Sequence[np.ndarray], trim: float = 0.2) -> np.ndarray:
    if not cdfs:
        raise ValueError("No CDFs to aggregate")
    lengths = {len(c) for c in cdfs}
    if len(lengths) != 1:
        raise ValueError(f"CDFs have different lengths: {sorted(lengths)}")
    pooled = trimmed_mean_rows(np.vstack(cdfs), trim)
    return np.maximum.accumulate(pooled)
