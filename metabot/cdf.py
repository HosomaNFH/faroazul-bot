"""Metaculus-compatible CDFs from a handful of declared percentiles.

Pipeline for one forecaster:

1. ``sanitize``: sort, drop invalid points, map values to *locations* (0 at the
   lower bound, 1 at the upper bound; log scale when the question has a
   ``zero_point``), clamp into closed bounds and spread ties.
2. Monotone cubic (PCHIP, Fritsch-Carlson) interpolation between the declared
   percentiles, with exponential tails beyond the outermost ones.
3. ``standardize_cdf``: Metaculus rules (no mass outside closed bounds, at
   least 0.1% outside open bounds, a minimum step of 0.01/(n-1) and a maximum
   of 0.2*200/(n-1) per bin).

The standardisation mirrors ``NumericDistribution._standardize_cdf`` in
forecasting-tools (MIT licence, see THIRD_PARTY_NOTICES.md) so that our CDFs
pass the same server-side checks. Only numpy is required.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# 13 percentiles, as used by several winning bots (fine tails + dense body).
STANDARD_PERCENTILES: tuple[float, ...] = (
    0.01,
    0.025,
    0.05,
    0.10,
    0.20,
    0.40,
    0.50,
    0.60,
    0.80,
    0.90,
    0.95,
    0.975,
    0.99,
)


class DistributionError(ValueError):
    """The declared percentiles cannot be turned into a valid distribution."""


@dataclass(frozen=True)
class Scale:
    lower: float
    upper: float
    open_lower: bool
    open_upper: bool
    zero_point: float | None = None
    cdf_size: int = 201
    discrete: bool = False

    def __post_init__(self) -> None:
        if not (math.isfinite(self.lower) and math.isfinite(self.upper)):
            raise DistributionError("Bounds must be finite")
        if self.upper <= self.lower:
            raise DistributionError("Upper bound must be greater than lower bound")
        if self.cdf_size < 3:
            raise DistributionError("cdf_size must be at least 3")
        if self.zero_point is not None and self.zero_point >= self.lower:
            # Invalid log scale; fall back to linear.
            object.__setattr__(self, "zero_point", None)

    @classmethod
    def from_ctx(cls, ctx) -> "Scale":  # ctx: QCtx
        if ctx.lower is None or ctx.upper is None:
            raise DistributionError("Question has no numeric range")
        return cls(
            lower=float(ctx.lower),
            upper=float(ctx.upper),
            open_lower=bool(ctx.open_lower),
            open_upper=bool(ctx.open_upper),
            zero_point=ctx.zero_point,
            cdf_size=int(ctx.cdf_size),
            discrete=ctx.kind == "discrete",
        )

    # --- coordinate transforms (same formulas as Metaculus/forecasting-tools) --
    def _ratio(self) -> float:
        assert self.zero_point is not None
        return (self.upper - self.zero_point) / (self.lower - self.zero_point)

    def to_loc(self, x) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        lo, hi = self.lower, self.upper
        if self.zero_point is None:
            return (x - lo) / (hi - lo)
        r = self._ratio()
        arg = (x - lo) * (r - 1.0) + (hi - lo)
        arg = np.maximum(arg, 1e-12 * (hi - lo))
        return (np.log(arg) - math.log(hi - lo)) / math.log(r)

    def from_loc(self, loc) -> np.ndarray:
        loc = np.asarray(loc, dtype=float)
        lo, hi = self.lower, self.upper
        if self.zero_point is None:
            return lo + (hi - lo) * loc
        r = self._ratio()
        return lo + (hi - lo) * (np.power(r, loc) - 1.0) / (r - 1.0)

    def grid_locations(self) -> np.ndarray:
        return np.linspace(0.0, 1.0, self.cdf_size)

    def grid_values(self) -> np.ndarray:
        return self.from_loc(self.grid_locations())

    @property
    def bin_width(self) -> float:
        return 1.0 / (self.cdf_size - 1)

    @property
    def min_step(self) -> float:
        return 0.01 / (self.cdf_size - 1)

    @property
    def max_step(self) -> float:
        return min(1.0, 0.2 * 200.0 / (self.cdf_size - 1))


# --------------------------------------------------------------------------- #
# Sanitising declared percentiles
# --------------------------------------------------------------------------- #


def _spread_ties(locs: np.ndarray, scale: Scale) -> np.ndarray:
    """Make locations strictly increasing.

    Runs of equal locations (typical for integer outcomes, e.g. p40=p50=p60=3)
    are spread evenly across one grid bin centred on the value, which keeps the
    mass in the bucket the forecaster named.
    """
    locs = locs.astype(float).copy()
    width = scale.bin_width
    tol = 1e-9
    lo_limit = -np.inf if scale.open_lower else 0.0
    hi_limit = np.inf if scale.open_upper else 1.0

    i = 0
    n = len(locs)
    while i < n:
        j = i
        while j + 1 < n and abs(locs[j + 1] - locs[i]) <= tol:
            j += 1
        k = j - i + 1
        if k > 1:
            centre = locs[i]
            start = max(centre - width / 2.0, lo_limit)
            end = min(centre + width / 2.0, hi_limit)
            if end - start < width / 2.0:  # squeezed against a closed bound
                if start <= lo_limit + tol:
                    end = start + width
                else:
                    start = end - width
            for m in range(k):
                locs[i + m] = start + (end - start) * (m + 0.5) / k
        i = j + 1

    # Forward then backward pass to guarantee strict monotonicity.
    eps = 1e-9
    for i in range(1, n):
        if locs[i] <= locs[i - 1]:
            locs[i] = locs[i - 1] + eps
    if not scale.open_upper and locs[-1] > 1.0:
        locs[-1] = 1.0
        for i in range(n - 2, -1, -1):
            if locs[i] >= locs[i + 1]:
                locs[i] = locs[i + 1] - eps
    return locs


def sanitize(
    declared: list[tuple[float, float]], scale: Scale
) -> tuple[np.ndarray, np.ndarray]:
    """Return strictly increasing (locations, probabilities)."""
    points = sorted(
        ((float(p), float(v)) for p, v in declared), key=lambda item: item[0]
    )
    probs: list[float] = []
    values: list[float] = []
    for p, v in points:
        if not (0.0 < p < 1.0) or not math.isfinite(v):
            continue
        if probs and abs(p - probs[-1]) < 1e-9:
            continue
        probs.append(p)
        values.append(v)
    if len(probs) < 3:
        raise DistributionError("Need at least 3 valid percentiles")

    vals = np.asarray(values, dtype=float)
    if np.any(np.diff(vals) < 0):
        # Values listed out of order: keep the set, restore monotonicity.
        vals = np.sort(vals)
    if scale.zero_point is not None:
        floor = scale.zero_point + 1e-9 * (scale.upper - scale.lower)
        vals = np.maximum(vals, floor)

    locs = scale.to_loc(vals)
    if not scale.open_lower:
        locs = np.maximum(locs, 0.0)
    if not scale.open_upper:
        locs = np.minimum(locs, 1.0)
    if locs[-1] - locs[0] <= 1e-12 and not scale.discrete:
        raise DistributionError("All percentiles are identical")
    locs = _spread_ties(locs, scale)
    return locs, np.asarray(probs, dtype=float)


def plausibility_issue(declared: list[tuple[float, float]], scale: Scale) -> str | None:
    """Detect probable unit/scale mistakes (e.g. millions vs units)."""
    if not declared:
        return "no percentiles"
    pts = sorted(declared)
    values = np.asarray([v for _, v in pts], dtype=float)
    probs = np.asarray([p for p, _ in pts], dtype=float)
    if not np.all(np.isfinite(values)):
        return "non-finite values"
    median = float(np.interp(0.5, probs, values)) if len(values) > 1 else float(values[0])
    loc_median = float(scale.to_loc(median))
    if loc_median < -1.0 or loc_median > 2.0:
        return (
            f"median {median:g} is far outside the question range "
            f"[{scale.lower:g}, {scale.upper:g}] (wrong units?)"
        )
    locs = scale.to_loc(values)
    if float(np.max(locs)) < -0.5 or float(np.min(locs)) > 1.5:
        return "all percentiles are far outside the question range"
    return None


# --------------------------------------------------------------------------- #
# Monotone cubic interpolation (Fritsch-Carlson / PCHIP)
# --------------------------------------------------------------------------- #


def _edge_slope(h0: float, h1: float, m0: float, m1: float) -> float:
    d = ((2.0 * h0 + h1) * m0 - h0 * m1) / (h0 + h1)
    if np.sign(d) != np.sign(m0):
        return 0.0
    if np.sign(m0) != np.sign(m1) and abs(d) > abs(3.0 * m0):
        return 3.0 * m0
    return float(d)


def pchip_slopes(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    n = len(x)
    h = np.diff(x)
    delta = np.diff(y) / h
    d = np.zeros(n)
    if n == 2:
        d[:] = delta[0]
        return d
    for k in range(1, n - 1):
        if delta[k - 1] * delta[k] <= 0:
            d[k] = 0.0
        else:
            w1 = 2.0 * h[k] + h[k - 1]
            w2 = h[k] + 2.0 * h[k - 1]
            d[k] = (w1 + w2) / (w1 / delta[k - 1] + w2 / delta[k])
    d[0] = _edge_slope(h[0], h[1], delta[0], delta[1])
    d[-1] = _edge_slope(h[-1], h[-2], delta[-1], delta[-2])
    return d


def pchip_eval(x: np.ndarray, y: np.ndarray, d: np.ndarray, xq: np.ndarray) -> np.ndarray:
    idx = np.clip(np.searchsorted(x, xq, side="right") - 1, 0, len(x) - 2)
    h = x[idx + 1] - x[idx]
    t = (xq - x[idx]) / h
    t2 = t * t
    t3 = t2 * t
    return (
        (2 * t3 - 3 * t2 + 1) * y[idx]
        + (t3 - 2 * t2 + t) * h * d[idx]
        + (-2 * t3 + 3 * t2) * y[idx + 1]
        + (t3 - t2) * h * d[idx + 1]
    )


def evaluate_cdf(locs: np.ndarray, probs: np.ndarray, xq: np.ndarray) -> np.ndarray:
    """CDF at locations ``xq``: PCHIP inside, exponential tails outside."""
    xq = np.asarray(xq, dtype=float)
    out = np.empty_like(xq)
    d = pchip_slopes(locs, probs)
    inside = (xq >= locs[0]) & (xq <= locs[-1])
    if np.any(inside):
        out[inside] = pchip_eval(locs, probs, d, xq[inside])
    below = xq < locs[0]
    if np.any(below):
        s_lo = (locs[1] - locs[0]) / math.log(probs[1] / probs[0])
        out[below] = probs[0] * np.exp((xq[below] - locs[0]) / s_lo)
    above = xq > locs[-1]
    if np.any(above):
        s_hi = (locs[-1] - locs[-2]) / math.log((1 - probs[-2]) / (1 - probs[-1]))
        out[above] = 1.0 - (1.0 - probs[-1]) * np.exp(-(xq[above] - locs[-1]) / s_hi)
    out = np.clip(out, 0.0, 1.0)
    return np.maximum.accumulate(out)


# --------------------------------------------------------------------------- #
# Metaculus standardisation
# --------------------------------------------------------------------------- #


def max_pmf_value(cdf_size: int, wiggle: bool = True) -> float:
    cap = 0.2 * (200.0 / (cdf_size - 1))
    return cap * 0.95 if wiggle else cap


def standardize_cdf(
    cdf, open_lower: bool, open_upper: bool, uniform_weight: float = 0.0102
) -> np.ndarray:
    """Apply Metaculus constraints (adapted from forecasting-tools, MIT).

    ``uniform_weight`` is the share of probability spread uniformly over the
    range, which guarantees the minimum step. forecasting-tools uses exactly
    0.01, which puts flat stretches of the CDF right at the 5e-5 limit that its
    own validator checks with a strict ``<``; 0.0102 leaves a 2% margin.
    """
    raw = np.asarray(cdf, dtype=float).copy()
    n = len(raw)
    if n < 3:
        raise DistributionError("CDF too short")
    scale_lower_to = 0.0 if open_lower else raw[0]
    scale_upper_to = 1.0 if open_upper else raw[-1]
    inbound = scale_upper_to - scale_lower_to
    if inbound <= 1e-9:
        raise DistributionError("No probability mass inside the question range")

    location = np.linspace(0.0, 1.0, n)
    rescaled = (raw - scale_lower_to) / inbound
    u = uniform_weight
    if open_lower and open_upper:  # cdf[0] >= 0.001, cdf[-1] <= 0.999
        out = (0.998 - u) * rescaled + u * location + 0.001
    elif open_lower:  # cdf[0] >= 0.001, cdf[-1] == 1
        out = (0.999 - u) * rescaled + u * location + 0.001
    elif open_upper:  # cdf[0] == 0, cdf[-1] <= 0.999
        out = (0.999 - u) * rescaled + u * location
    else:  # cdf[0] == 0, cdf[-1] == 1
        out = (1.0 - u) * rescaled + u * location

    pmf = np.diff(out, prepend=0.0, append=1.0)
    cap = max_pmf_value(n)

    def cap_pmf(scale: float) -> np.ndarray:
        return np.concatenate([pmf[:1], np.minimum(cap, scale * pmf[1:-1]), pmf[-1:]])

    def capped_sum(scale: float) -> float:
        return float(cap_pmf(scale).sum())

    lo = hi = scale = 1.0
    for _ in range(200):
        if capped_sum(hi) >= 1.0:
            break
        hi *= 1.2
    for _ in range(100):
        scale = 0.5 * (lo + hi)
        total = capped_sum(scale)
        if total < 1.0:
            lo = scale
        else:
            hi = scale
        if total == 1.0 or (hi - lo) < 2e-5:
            break
    pmf = cap_pmf(scale)
    inner = pmf[1:-1].sum()
    if inner <= 0:
        raise DistributionError("Degenerate CDF after capping")
    pmf[1:-1] *= (out[-1] - out[0]) / inner
    result = np.cumsum(pmf)[:-1]
    return np.round(result, 10)


def validate_cdf(cdf, scale: Scale) -> None:
    """Raise DistributionError if ``cdf`` would be rejected by Metaculus."""
    cdf = np.asarray(cdf, dtype=float)
    if len(cdf) != scale.cdf_size:
        raise DistributionError(f"CDF has {len(cdf)} points, expected {scale.cdf_size}")
    if not np.all(np.isfinite(cdf)):
        raise DistributionError("CDF has non-finite values")
    steps = np.diff(cdf)
    if np.any(steps < scale.min_step * 0.98):
        raise DistributionError("CDF increases too slowly somewhere (min step)")
    if np.any(steps > scale.max_step + 1e-9):
        raise DistributionError("CDF too concentrated in one bin (max step)")
    if scale.open_lower:
        if cdf[0] < 0.001 - 1e-9:
            raise DistributionError("Open lower bound needs >= 0.1% below it")
    elif abs(cdf[0]) > 1e-9:
        raise DistributionError("Closed lower bound requires cdf[0] == 0")
    if scale.open_upper:
        if cdf[-1] > 0.999 + 1e-9:
            raise DistributionError("Open upper bound needs >= 0.1% above it")
    elif abs(cdf[-1] - 1.0) > 1e-9:
        raise DistributionError("Closed upper bound requires cdf[-1] == 1")


def build_cdf(declared: list[tuple[float, float]], scale: Scale) -> np.ndarray:
    """Declared (probability, value) pairs -> standardised CDF on the grid."""
    locs, probs = sanitize(declared, scale)
    raw = evaluate_cdf(locs, probs, scale.grid_locations())
    return standardize_cdf(raw, scale.open_lower, scale.open_upper)


def cdf_quantile(cdf: np.ndarray, scale: Scale, q: float) -> float:
    """Approximate value at probability ``q`` (for logs and tests)."""
    grid = scale.grid_locations()
    cdf = np.asarray(cdf, dtype=float)
    q = float(np.clip(q, cdf[0], cdf[-1]))
    loc = float(np.interp(q, cdf, grid))
    return float(scale.from_loc(loc))
