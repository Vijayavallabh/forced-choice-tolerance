#!/usr/bin/env python3
r"""A capsule-level sign-flip test, and the interval it inverts to.

Every paired contrast in the paper is a mean over items of a per-item difference -- one
reading of the same runs minus another -- with items clustered in capsules. The percentile
cluster bootstrap reads it at a level calibrated by simulation or by the double bootstrap.
This is the check that needs neither: under the null that the contrast is $\delta$, each
capsule's summed deviation $s_c(\delta)=\sum_{i\in c}(x_i-\delta)$ is taken to be symmetric
about zero, so flipping the signs of whole capsules leaves the statistic
$T(\delta)=\sum_c s_c(\delta)$ as likely as before. The p-value is the share of sign patterns
whose $|\sum_c\epsilon_c s_c(\delta)|$ reaches $|T(\delta)|$: every pattern when there are at
most 20 capsules, else 200,000 drawn at random (the same patterns for every $\delta$). The
interval is every $\delta$ the test does not reject at 5%, found by bisection on each side;
with five capsules or fewer no pattern count reaches below 5% (the smallest p-value is
$2/2^n$), no $\delta$ is rejected, and the interval is reported unbounded (None). Exact under
the symmetry assumption for any cluster sizes; it assumes nothing about their spread.

    python3 randomization.py        # a self-check on simulated contrasts (results/randomization_selfcheck.json)
"""
import itertools

import numpy as np

EXACT_UP_TO = 20
DRAWS = 200000


def _patterns(n, seed):
    if n <= EXACT_UP_TO:
        return np.array(list(itertools.product((1.0, -1.0), repeat=n)))
    rng = np.random.default_rng(seed)
    return rng.choice((1.0, -1.0), size=(DRAWS, n))


def _clusters(per_item, capsule):
    sums, counts = {}, {}
    for q, v in per_item.items():
        if v is None:
            continue
        c = capsule[q]
        sums[c] = sums.get(c, 0.0) + v
        counts[c] = counts.get(c, 0) + 1
    keys = sorted(sums)
    return np.array([sums[k] for k in keys]), np.array([counts[k] for k in keys], dtype=float)


def _p(sums, counts, delta, eps):
    s = sums - delta * counts
    t = abs(s.sum())
    return float(np.mean(np.abs(eps @ s) >= t - 1e-12))


def signflip(per_item, capsule, alpha=0.05, seed=20260925):
    """{mean, p (against 0), lo, hi (the inverted interval), n_items, n_clusters}, in points."""
    sums, counts = _clusters(per_item, capsule)
    if len(sums) < 2:
        return None
    eps = _patterns(len(sums), seed)
    mean = sums.sum() / counts.sum()
    p0 = _p(sums, counts, 0.0, eps)
    span = max(1.0, float(np.max(np.abs(sums / counts))) * 2 + abs(mean))

    def edge(direction):
        inside, outside = mean, mean + direction * span
        while _p(sums, counts, outside, eps) > alpha:      # widen until rejected
            outside += direction * span
        for _ in range(30):
            mid = (inside + outside) / 2
            if _p(sums, counts, mid, eps) > alpha:
                inside = mid
            else:
                outside = mid
        return inside

    # far from the mean every capsule deviates the same way and only the two all-same patterns
    # reach |T|, so the test can reject somewhere only if 2 / (patterns) is below alpha
    bounded = 2.0 / len(eps) <= alpha
    return {"mean": 100.0 * mean, "p": p0, "lo": 100.0 * edge(-1.0) if bounded else None,
            "hi": 100.0 * edge(1.0) if bounded else None,
            "n_items": int(counts.sum()), "n_clusters": int(len(sums)), "exact": len(sums) <= EXACT_UP_TO}


def _selfcheck(reps=200, seed=1):
    """Coverage of the inverted interval on files shaped like the moved keys (37 items in 21
    capsules of their sizes), paired differences in {-1, 0, 1} with a true mean of 0.15."""
    rng = np.random.default_rng(seed)
    sizes = np.array([1] * 12 + [2] * 3 + [3] * 5 + [4] * 1)
    capsule = {i: c for c, n in enumerate(sizes) for i in range(int(sizes[:c].sum()), int(sizes[:c + 1].sum()))}
    hit = 0
    for _ in range(reps):
        x = rng.choice((-1.0, 0.0, 1.0), p=(0.05, 0.75, 0.20), size=int(sizes.sum()))
        r = signflip(dict(enumerate(x)), capsule, seed=int(rng.integers(1 << 30)))
        hit += r["lo"] <= 15.0 <= r["hi"]
    print(f"inverted sign-flip interval covers the true 0.15 on {hit}/{reps} simulated files")
    return {"reps": reps, "covered": int(hit), "n_items": int(sizes.sum()), "n_clusters": len(sizes),
            "true_contrast": 15.0, "seed": seed}


if __name__ == "__main__":
    import json
    import pathlib
    out = pathlib.Path(__file__).resolve().parent / "results" / "randomization_selfcheck.json"
    out.write_text(json.dumps(_selfcheck(), indent=1) + "\n")
    print(f"wrote results/{out.name}")
