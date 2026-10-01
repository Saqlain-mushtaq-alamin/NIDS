"""
src/evaluation/statistics.py — Multi-seed significance testing (PHASE_2 section 2.4, FR6).

A single cross-dataset number is an anecdote.  This module turns the
cross-dataset matrix into a claim that survives a rerun: every dataset pair is
evaluated over :data:`DEFAULT_SEEDS` seeds and the resulting per-pair F1 deltas
are summarised with a mean, a standard deviation, a 95% confidence interval and
a paired significance test.

Tests used
----------
``paired_ttest`` and ``wilcoxon``
    Non-parametric *and* parametric options are both offered, because the
    per-seed deltas within a pair are not guaranteed to be normally distributed
    (with 5 seeds that is a hard assumption to make).  ``wilcoxon`` enumerates
    the exact sign distribution for ``n <= 25`` and falls back to the normal
    approximation beyond that.  Note the resolution limit: the smallest
    attainable one-sided exact p is ``0.5 ** n``, so 4 seeds can never reject at
    ``alpha=0.05`` and 5 seeds are the minimum for the exact test to be able to.
``bootstrap_ci``
    Distribution-free 95% CI over resampled seed deltas; used as the headline
    interval because it makes no normality assumption.
``mcnemar``
    Exact binomial McNemar test for the *paired* in-domain vs cross-domain
    correctness of the same test instances.  This is the strongest evidence the
    harness can produce: it answers "on identical test rows, does the model get
    more cases right in-domain than under transfer?" without aggregating over
    seeds at all.

Bootstrap resampling is driven by ``numpy.random.default_rng(seed)`` so every
number in the report is reproducible.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

DEFAULT_SEEDS: tuple[int, ...] = (42, 1337, 2024, 7, 99, 12345, 555, 31337)
BOOTSTRAP_RESAMPLES = 10_000
CI_LEVEL = 0.95


class StatisticsError(RuntimeError):
    """Raised when a statistical test cannot be run on the supplied data."""


@dataclass
class PairedComparison:
    """Result of comparing two paired score vectors sample-by-sample."""

    name: str
    test: str
    statistic: float
    p_value: float
    n_pairs: int
    mean_difference: float
    ci_low: float
    ci_high: float
    significant: bool
    alternative: str = "greater"
    note: str = ""
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "comparison": self.name,
            "test": self.test,
            "statistic": self.statistic,
            "p_value": self.p_value,
            "n_pairs": self.n_pairs,
            "mean_difference": self.mean_difference,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "significant": self.significant,
            "alternative": self.alternative,
            "note": self.note,
        }

    def format(self) -> str:
        verdict = "SIGNIFICANT" if self.significant else "not significant"
        return (
            f"{self.name}: {self.test} p={self.p_value:.2e} "
            f"(mean delta={self.mean_difference:+.4f}, "
            f"95% CI [{self.ci_low:+.4f}, {self.ci_high:+.4f}], "
            f"n={self.n_pairs}) - {verdict}"
            + (f" [{self.note}]" if self.note else "")
        )


# --------------------------------------------------------------------------- #
# Interval estimation
# --------------------------------------------------------------------------- #
def bootstrap_ci(values: Sequence[float], n_resamples: int = BOOTSTRAP_RESAMPLES,
                 ci_level: float = CI_LEVEL, seed: int = 42) -> tuple[float, float]:
    """
    Percentile bootstrap confidence interval for the mean of ``values``.

    Args:
        values: Sample to summarise.
        n_resamples: Bootstrap replicates.
        ci_level: Confidence level in (0, 1).
        seed: RNG seed for reproducibility.

    Returns:
        ``(ci_low, ci_high)``.

    Raises:
        StatisticsError: if ``values`` is empty.
    """
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        raise StatisticsError("bootstrap_ci requires at least one finite value")

    rng = np.random.default_rng(seed)
    n = array.size
    means = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        means[i] = rng.choice(array, size=n, replace=True).mean()

    alpha = (1.0 - ci_level) / 2.0
    return (
        float(np.percentile(means, 100.0 * alpha)),
        float(np.percentile(means, 100.0 * (1.0 - alpha))),
    )


def normal_ci(values: Sequence[float], ci_level: float = CI_LEVEL) -> tuple[float, float]:
    """Mean ± z·(σ/√n) interval. Reported alongside the bootstrap CI."""
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        raise StatisticsError("normal_ci requires at least one finite value")
    mean = float(array.mean())
    if array.size < 2:
        return mean, mean
    se = float(array.std(ddof=1) / np.sqrt(array.size))
    z = {0.90: 1.6449, 0.95: 1.9600, 0.99: 2.5758}.get(round(ci_level, 2), 1.9600)
    return mean - z * se, mean + z * se


# --------------------------------------------------------------------------- #
# Paired significance tests
# --------------------------------------------------------------------------- #
def paired_ttest(before: Sequence[float], after: Sequence[float],
                 alternative: str = "greater",
                 ci_level: float = CI_LEVEL) -> PairedComparison:
    """
    Two-sided-paired t-test on the differences ``before - after``.

    ``alternative='greater'`` tests *one-sided* that ``before > after``
    (i.e. in-domain beats cross-domain), which is the hypothesis of the Phase 2
    research question.  The returned ``p_value`` is that one-sided p-value;
    ``significant`` compares it against ``1 - ci_level`` halved for the
    one-sided test.
    """
    a, b = _aligned(before, after, "paired_ttest")
    differences = a - b
    n = differences.size
    mean_diff = float(differences.mean())

    ci_low, ci_high = normal_ci(differences, ci_level=ci_level)
    if n < 2:
        return PairedComparison(
            name="paired", test="paired_ttest", statistic=float("nan"),
            p_value=float("nan"), n_pairs=n, mean_difference=mean_diff,
            ci_low=mean_diff, ci_high=mean_diff, significant=False,
            alternative=alternative, note="fewer than 2 seeds: no test possible",
        )

    std = float(differences.std(ddof=1))
    if std == 0.0:
        # Identical across every seed: the difference is a deterministic
        # constant, which is as significant as a test can report.
        p_one_sided = 0.0 if mean_diff != 0 else 1.0
        return PairedComparison(
            name="paired", test="paired_ttest", statistic=float("inf") if mean_diff > 0 else 0.0,
            p_value=p_one_sided, n_pairs=n, mean_difference=mean_diff,
            ci_low=mean_diff, ci_high=mean_diff, significant=bool(mean_diff != 0),
            alternative=alternative, note="zero variance across seeds",
        )

    t_stat = mean_diff / (std / np.sqrt(n))
    p_two_sided = _t_sf(abs(t_stat), df=n - 1) * 2.0
    p_value = p_two_sided / 2.0 if alternative == "greater" and mean_diff > 0 else p_two_sided
    if alternative == "greater" and mean_diff <= 0:
        p_value = 1.0 - p_two_sided / 2.0

    alpha = 1.0 - ci_level
    return PairedComparison(
        name="paired", test="paired_ttest", statistic=float(t_stat),
        p_value=float(p_value), n_pairs=n, mean_difference=mean_diff,
        ci_low=ci_low, ci_high=ci_high, significant=bool(p_value < alpha),
        alternative=alternative,
        details={"std_difference": std, "differences": differences.tolist()},
    )


def wilcoxon(before: Sequence[float], after: Sequence[float],
             alternative: str = "greater",
             ci_level: float = CI_LEVEL) -> PairedComparison:
    """
    Wilcoxon signed-rank test on ``before - after`` (distribution-free).

    Implemented directly (rather than via ``scipy.stats.wilcoxon``) so the
    project keeps a single, documented implementation of the exact rank
    enumeration for small samples, and degrades to the normal approximation
    for n > 25.
    """
    a, b = _aligned(before, after, "wilcoxon")
    differences = a - b
    n = differences.size
    mean_diff = float(differences.mean())
    ci_low, ci_high = bootstrap_ci(differences, seed=42, ci_level=ci_level)

    nonzero = differences[differences != 0]
    n_eff = nonzero.size
    if n_eff == 0:
        return PairedComparison(
            name="paired", test="wilcoxon", statistic=0.0, p_value=1.0, n_pairs=n,
            mean_difference=mean_diff, ci_low=mean_diff, ci_high=mean_diff,
            significant=False, alternative=alternative,
            note="all differences are exactly zero",
        )

    ranks = _rank_abs(nonzero)
    pos = float(ranks[nonzero > 0].sum())
    neg = float(ranks[nonzero < 0].sum())
    w_min = min(pos, neg)

    if n_eff <= 25:
        p_value = _wilcoxon_exact_p(w_min, n_eff)
        method = "exact"
    else:
        expected = n_eff * (n_eff + 1) / 4.0
        sd = np.sqrt(n_eff * (n_eff + 1) * (2.0 * n_eff + 1) / 24.0)
        z = (w_min - expected) / sd
        # Two-sided p doubles the *lower* tail: small rank sums are the
        # evidence against H0, so the upper tail would invert the result.
        p_value = min(1.0, 2.0 * _norm_cdf(z))
        method = "normal approximation"

    if alternative == "greater":
        p_value = p_value / 2.0 if mean_diff > 0 else 1.0 - p_value / 2.0

    return PairedComparison(
        name="paired", test="wilcoxon", statistic=float(w_min), p_value=float(p_value),
        n_pairs=n_eff, mean_difference=mean_diff, ci_low=ci_low, ci_high=ci_high,
        significant=bool(p_value < 1.0 - ci_level), alternative=alternative,
        note=f"{method}, {n_eff} non-zero differences",
        details={"positive_rank_sum": pos, "negative_rank_sum": neg},
    )


def mcnemar(y_true: Sequence, model_a_pred: Sequence, model_b_pred: Sequence,
            alternative: str = "greater",
            ci_level: float = CI_LEVEL) -> PairedComparison:
    """
    Exact McNemar test between two classifiers' predictions on the *same* rows.

    This is the most rigorous comparison available for the Phase 2 question
    because it removes seed variance entirely: both models are scored on one
    fixed test set and only the discordant pairs (one right, one wrong) carry
    information.

    Args:
        y_true: Ground-truth labels.
        model_a_pred: Predictions of the in-domain model.
        model_b_pred: Predictions of the cross-domain model.
        alternative: ``'greater'`` asserts that model A is correct more often.
        ci_level: Confidence level.

    Returns:
        A :class:`PairedComparison` whose ``mean_difference`` is the accuracy
        gap ``acc(A) - acc(B)``.
    """
    true = np.asarray(list(y_true)).astype(str)
    a = np.asarray(list(model_a_pred)).astype(str)
    b = np.asarray(list(model_b_pred)).astype(str)
    if not (true.size == a.size == b.size):
        raise StatisticsError(
            f"mcnemar requires equal lengths, got {true.size}, {a.size}, {b.size}"
        )
    if true.size == 0:
        raise StatisticsError("mcnemar requires at least one row")

    correct_a = a == true
    correct_b = b == true
    b_only = int(np.count_nonzero(correct_b & ~correct_a))  # B right, A wrong
    a_only = int(np.count_nonzero(correct_a & ~correct_b))  # A right, B wrong
    n_discordant = a_only + b_only

    acc_gap = float(correct_a.mean() - correct_b.mean())

    if n_discordant == 0:
        p_value = 1.0
        statistic, note = float("nan"), "no discordant pairs"
    else:
        statistic = float(min(a_only, b_only))
        p_two = _binom_test_two_sided(min(a_only, b_only), n_discordant)
        if alternative == "greater":
            p_value = p_two / 2.0 if a_only > b_only else 1.0 - p_two / 2.0
        else:
            p_value = p_two
        note = f"{a_only} A-only vs {b_only} B-only correct"

    ci_low, ci_high = bootstrap_ci([acc_gap], seed=0, ci_level=ci_level)
    return PairedComparison(
        name="mcnemar", test="mcnemar", statistic=statistic, p_value=float(p_value),
        n_pairs=int(true.size), mean_difference=acc_gap, ci_low=ci_low, ci_high=ci_high,
        significant=bool(p_value < 1.0 - ci_level), alternative=alternative,
        note=note,
        details={"a_only_correct": a_only, "b_only_correct": b_only,
                 "n_discordant": n_discordant},
    )


# --------------------------------------------------------------------------- #
# Aggregation over the cross-dataset matrix
# --------------------------------------------------------------------------- #
def aggregate_over_seeds(results: pd.DataFrame,
                        value_column: str = "macro_f1",
                        seed_column: str = "seed") -> pd.DataFrame:
    """
    Collapse a multi-seed results matrix to mean ± std per ordered pair.

    Returns:
        DataFrame with ``train_dataset``, ``test_dataset``, ``in_domain``,
        ``n_seeds``, ``<value>_mean``, ``<value>_std``, ``<value>_min``,
        ``<value>_max`` and the bootstrap ``ci_low`` / ``ci_high``.
    """
    if value_column not in results.columns:
        raise KeyError(f"'{value_column}' not in results columns")
    if seed_column not in results.columns:
        raise KeyError(f"'{seed_column}' not in results columns")

    records = []
    for (train_name, test_name), group in results.groupby(
        ["train_dataset", "test_dataset"], sort=False
    ):
        values = group[value_column].to_numpy(dtype=np.float64)
        ci_low, ci_high = bootstrap_ci(values, seed=42)
        records.append({
            "train_dataset": train_name,
            "test_dataset": test_name,
            "in_domain": train_name == test_name,
            "n_seeds": int(group[seed_column].nunique()),
            "value_mean": float(np.nanmean(values)),
            "value_std": float(np.nanstd(values, ddof=1)) if values.size > 1 else 0.0,
            "value_min": float(np.nanmin(values)),
            "value_max": float(np.nanmax(values)),
            "ci_low": ci_low,
            "ci_high": ci_high,
        })
    return pd.DataFrame(records)


def compare_in_domain_vs_cross_domain(results: pd.DataFrame,
                                      value_column: str = "macro_f1",
                                      seed_column: str = "seed",
                                      n_resamples: int = BOOTSTRAP_RESAMPLES) -> pd.DataFrame:
    """
    Per-dataset paired comparison of in-domain vs cross-domain performance.

    For each training dataset ``A`` the diagonal cell ``A→A`` supplies the
    in-domain score and the off-diagonal cells ``A→B`` supply the cross-domain
    score, aligned **by seed** so the pairing is real.

    Returns:
        DataFrame with one row per training dataset, containing means, the
        bootstrap CI on the paired delta, the t-test and wilcoxon p-values, and
        a boolean ``cross_worse``.
    """
    for column in (value_column, seed_column, "train_dataset", "test_dataset"):
        if column not in results.columns:
            raise KeyError(f"'{column}' not in results columns")

    records = []
    for train_name, group in results.groupby("train_dataset", sort=False):
        diagonal = group[group["test_dataset"] == train_name]
        cross = group[group["test_dataset"] != train_name]
        if diagonal.empty or cross.empty:
            continue

        for test_name, cross_group in cross.groupby("test_dataset", sort=False):
            merged = diagonal.merge(
                cross_group, on=seed_column, suffixes=("_in", "_cross")
            )
            if merged.empty:
                continue
            in_scores = merged[f"{value_column}_in"].to_numpy(dtype=np.float64)
            cross_scores = merged[f"{value_column}_cross"].to_numpy(dtype=np.float64)
            deltas = in_scores - cross_scores
            ci_low, ci_high = bootstrap_ci(deltas, n_resamples=n_resamples, seed=42)

            t_result = paired_ttest(in_scores, cross_scores, alternative="greater")
            w_result = wilcoxon(in_scores, cross_scores, alternative="greater")
            n_merged = int(merged[seed_column].nunique())
            # The exact one-sided Wilcoxon test bottoms out at p = 0.5**n when
            # every difference has the same sign, so 4 seeds can never reject at
            # alpha=0.05 (0.0625) while 5 seeds can (0.03125).
            notes = [
                note for note in (t_result.note, w_result.note) if note
            ]
            if n_merged < 2:
                notes.append("fewer than 2 seeds: no paired test is possible")
            elif n_merged < 5:
                notes.append(
                    f"n={n_merged} seeds: the smallest attainable one-sided "
                    f"Wilcoxon p is {0.5 ** n_merged:.4f}, so only the t-test "
                    "can reject at alpha=0.05"
                )

            records.append({
                "train_dataset": train_name,
                "test_dataset": test_name,
                "n_seeds": n_merged,
                "in_domain_mean": float(np.mean(in_scores)),
                "cross_domain_mean": float(np.mean(cross_scores)),
                "delta_mean": float(np.mean(deltas)),
                "delta_std": float(np.std(deltas, ddof=1)) if deltas.size > 1 else 0.0,
                "delta_ci_low": ci_low,
                "delta_ci_high": ci_high,
                "retention": (
                    float(np.mean(cross_scores) / np.mean(in_scores))
                    if np.mean(in_scores) > 0 else float("nan")
                ),
                "t_statistic": t_result.statistic,
                "t_p_value": t_result.p_value,
                "wilcoxon_statistic": w_result.statistic,
                "wilcoxon_p_value": w_result.p_value,
                "cross_worse": bool(np.mean(cross_scores) < np.mean(in_scores)),
                "significant": bool(t_result.significant or w_result.significant),
                "seeds_sufficient": n_merged >= 2,
                "note": "; ".join(notes),
            })

    return pd.DataFrame(records)


def analyze(results: pd.DataFrame, value_column: str = "macro_f1",
            seed_column: str = "seed") -> dict:
    """
    One-call analysis bundle for the multi-seed suite.

    Returns:
        Dict with ``aggregate``, ``comparisons``, ``overall_ttest``,
        ``overall_wilcoxon``, ``overall_delta_ci``, ``n_seeds``,
        ``cross_worse_pairs`` and ``significant_pairs``.
    """
    aggregate = aggregate_over_seeds(results, value_column, seed_column)
    comparisons = compare_in_domain_vs_cross_domain(results, value_column, seed_column)

    diagonal = aggregate[aggregate["in_domain"]]
    off_diagonal = aggregate[~aggregate["in_domain"]]
    in_mean = float(diagonal["value_mean"].mean()) if len(diagonal) else float("nan")
    cross_mean = float(off_diagonal["value_mean"].mean()) if len(off_diagonal) else float("nan")

    overall_t = None
    overall_w = None
    overall_ci = (float("nan"), float("nan"))
    in_mean = float("nan")
    cross_mean = float("nan")
    mean_delta = float("nan")
    if len(diagonal) and len(off_diagonal):
        in_mean = float(diagonal["value_mean"].mean())
        cross_mean = float(off_diagonal["value_mean"].mean())

    # The unit of analysis is the *ordered pair*: each cross-domain cell is
    # matched with the diagonal cell of its own training dataset, so the paired
    # sample is one delta per ordered pair (never the ragged cell sets).
    if len(comparisons):
        deltas = comparisons["delta_mean"].to_numpy(dtype=np.float64)
        in_scores = comparisons["in_domain_mean"].to_numpy(dtype=np.float64)
        cross_scores = comparisons["cross_domain_mean"].to_numpy(dtype=np.float64)
        overall_t = paired_ttest(in_scores, cross_scores, alternative="greater")
        overall_w = wilcoxon(in_scores, cross_scores, alternative="greater")
        overall_ci = bootstrap_ci(deltas, seed=42)
        mean_delta = float(np.mean(deltas))

    return {
        "aggregate": aggregate,
        "comparisons": comparisons,
        "overall_ttest": overall_t,
        "overall_wilcoxon": overall_w,
        "overall_delta_ci": overall_ci,
        "in_domain_mean": in_mean,
        "cross_domain_mean": cross_mean,
        "mean_delta": mean_delta,
        "n_pairs": int(len(comparisons)),
        "n_seeds": int(results[seed_column].nunique()),
        "cross_worse_pairs": int(comparisons["cross_worse"].sum()) if len(comparisons) else 0,
        "total_pairs": int(len(comparisons)),
        "significant_pairs": int(comparisons["significant"].sum()) if len(comparisons) else 0,
    }


# --------------------------------------------------------------------------- #
# Internal numerics (no SciPy dependency)
# --------------------------------------------------------------------------- #
def _aligned(before: Sequence[float], after: Sequence[float], caller: str):
    a = np.asarray(list(before), dtype=np.float64)
    b = np.asarray(list(after), dtype=np.float64)
    if a.size != b.size:
        raise StatisticsError(
            f"{caller} requires paired samples of equal length, got {a.size} and {b.size}"
        )
    if a.size == 0:
        raise StatisticsError(f"{caller} requires at least one sample")
    return a, b


def _rank_abs(values: np.ndarray) -> np.ndarray:
    """Average ranks of ``|values|`` (ties share the mean rank)."""
    order = np.argsort(np.abs(values), kind="stable")
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[order] = np.arange(1, values.size + 1, dtype=np.float64)

    # Average tied ranks.
    magnitudes = np.abs(values)
    for magnitude in np.unique(magnitudes):
        mask = magnitudes == magnitude
        if mask.sum() > 1:
            ranks[mask] = ranks[mask].mean()
    return ranks


def _wilcoxon_exact_p(w_min: float, n: int) -> float:
    """
    Two-sided exact Wilcoxon signed-rank p-value by full enumeration.

    Counts how many of the ``2**n`` sign assignments produce a rank sum at or
    below ``w_min``.  Ties (zero differences) are excluded upstream, so each
    remaining difference contributes a distinct rank magnitude.
    """
    if n > 20:  # 2**20 enumerations is the practical ceiling.
        raise StatisticsError("exact wilcoxon enumeration is limited to n <= 20")

    total = 1 << n
    ranks = list(range(1, n + 1))
    count = 0
    for mask in range(total):
        rank_sum = 0
        for i in range(n):
            if mask >> i & 1:
                rank_sum += ranks[i]
        if rank_sum <= w_min + 1e-9:
            count += 1
    return min(1.0, 2.0 * count / total)


def _binom_test_two_sided(k: int, n: int) -> float:
    """Two-sided exact binomial p-value under H0: p = 0.5."""
    from math import comb

    if n == 0:
        return 1.0
    observed = comb(n, k) / (2 ** n)
    total = 0.0
    for i in range(n + 1):
        probability = comb(n, i) / (2 ** n)
        if probability <= observed + 1e-15:
            total += probability
    return min(1.0, total)


def _norm_sf(z: float) -> float:
    """Upper-tail probability of the standard normal distribution."""
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def _norm_cdf(z: float) -> float:
    """Lower-tail probability of the standard normal distribution."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _t_sf(t: float, df: int) -> float:
    """
    Upper tail of Student's t distribution, via the regularised incomplete beta.

    Uses the continued-fraction evaluation of ``I_{df/(df+t^2)}(df/2, 1/2)``.
    """
    x = df / (df + t * t)
    return 0.5 * _betainc(df / 2.0, 0.5, x)


def _betainc(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta function ``I_x(a, b)``."""
    from math import exp, lgamma, log

    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0

    lbeta = lgamma(a) + lgamma(b) - lgamma(a + b)
    front = exp(a * log(x) + b * log(1.0 - x) - lbeta)

    if x < (a + 1.0) / (a + b + 2.0):
        return front * _beta_cf(a, b, x) / a
    return 1.0 - exp(b * log(1.0 - x) + a * log(x) - lbeta) * _beta_cf(b, a, 1.0 - x) / b


def _beta_cf(a: float, b: float, x: float, max_iter: int = 300,
             eps: float = 3e-16) -> float:
    """Continued fraction for the incomplete beta function (Lentz's method)."""
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d

    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c

        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


__all__ = [
    "DEFAULT_SEEDS",
    "BOOTSTRAP_RESAMPLES",
    "CI_LEVEL",
    "StatisticsError",
    "PairedComparison",
    "bootstrap_ci",
    "normal_ci",
    "paired_ttest",
    "wilcoxon",
    "mcnemar",
    "aggregate_over_seeds",
    "compare_in_domain_vs_cross_domain",
    "analyze",
]
