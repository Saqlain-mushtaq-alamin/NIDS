"""
tests/test_statistics.py — Phase 2 multi-seed significance testing (FR6).

The reference values in this file are cross-checked against SciPy
(``scipy.stats``) where SciPy is installed, and are additionally verified
against closed-form results (binomial and normal identities) so the suite stays
meaningful without SciPy.
"""
import math

import numpy as np
import pandas as pd
import pytest

from src.evaluation.statistics import (
    DEFAULT_SEEDS,
    StatisticsError,
    aggregate_over_seeds,
    analyze,
    bootstrap_ci,
    compare_in_domain_vs_cross_domain,
    mcnemar,
    normal_ci,
    paired_ttest,
    wilcoxon,
)

scipy_stats = pytest.importorskip("scipy.stats", reason="SciPy cross-check only")


# --------------------------------------------------------------------------- #
# bootstrap_ci / normal_ci
# --------------------------------------------------------------------------- #
class TestIntervals:
    def test_bootstrap_brackets_mean_and_is_deterministic(self):
        values = [0.1, 0.2, 0.3, 0.25, 0.15]
        low, high = bootstrap_ci(values, n_resamples=2000, seed=42)
        assert low < np.mean(values) < high
        assert bootstrap_ci(values, n_resamples=2000, seed=42) == (low, high)

    def test_bootstrap_seed_changes_draw_but_not_conclusion(self):
        values = [0.1, 0.2, 0.3, 0.25, 0.15]
        a = bootstrap_ci(values, n_resamples=1000, seed=1)
        b = bootstrap_ci(values, n_resamples=1000, seed=2)
        assert abs(a[0] - b[0]) < 0.05 and abs(a[1] - b[1]) < 0.05

    def test_bootstrap_degenerate_sample(self):
        assert bootstrap_ci([0.5] * 10) == (0.5, 0.5)

    def test_bootstrap_rejects_empty(self):
        with pytest.raises(StatisticsError, match="at least one finite"):
            bootstrap_ci([])

    def test_bootstrap_ignores_nan(self):
        low, high = bootstrap_ci([1.0, 2.0, float("nan"), 3.0])
        assert math.isfinite(low) and math.isfinite(high)

    def test_normal_ci_matches_known_values(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        low, high = normal_ci(values)
        mean = 3.0
        se = np.std(values, ddof=1) / math.sqrt(5)
        assert low == pytest.approx(mean - 1.96 * se)
        assert high == pytest.approx(mean + 1.96 * se)

    def test_normal_ci_single_sample(self):
        assert normal_ci([0.7]) == (0.7, 0.7)

    def test_normal_ci_rejects_empty(self):
        with pytest.raises(StatisticsError):
            normal_ci([])


# --------------------------------------------------------------------------- #
# paired_ttest
# --------------------------------------------------------------------------- #
class TestPairedTtest:
    def test_matches_scipy(self):
        rng = np.random.default_rng(0)
        a = rng.normal(0.80, 0.05, 8)
        b = rng.normal(0.40, 0.05, 8)
        mine = paired_ttest(a, b, alternative="greater")
        reference = scipy_stats.ttest_rel(a, b, alternative="greater")
        assert mine.statistic == pytest.approx(reference.statistic)
        assert mine.p_value == pytest.approx(reference.pvalue, rel=1e-9)
        assert mine.significant is True

    def test_two_sided_matches_scipy(self):
        a = [0.9, 0.8, 0.85, 0.95, 0.88]
        b = [0.5, 0.45, 0.55, 0.5, 0.48]
        mine = paired_ttest(a, b, alternative="two-sided")
        reference = scipy_stats.ttest_rel(a, b, alternative="two-sided")
        assert mine.p_value == pytest.approx(reference.pvalue, rel=1e-9)

    def test_no_difference_is_not_significant(self):
        a = [0.5, 0.51, 0.49, 0.5, 0.52]
        mine = paired_ttest(a, a, alternative="greater")
        assert mine.p_value > 0.05
        assert mine.significant is False

    def test_constant_difference_is_significant(self):
        mine = paired_ttest([0.9] * 5, [0.4] * 5, alternative="greater")
        assert mine.mean_difference == pytest.approx(0.5)
        assert mine.significant is True
        assert "zero variance" in mine.note

    def test_identical_samples_gives_p_one(self):
        mine = paired_ttest([0.3] * 4, [0.3] * 4, alternative="greater")
        assert mine.p_value == 1.0
        assert mine.significant is False

    def test_single_sample_cannot_be_tested(self):
        mine = paired_ttest([0.9], [0.1], alternative="greater")
        assert math.isnan(mine.p_value)
        assert mine.significant is False
        assert "fewer than 2" in mine.note

    def test_reversed_direction_gives_p_above_half(self):
        mine = paired_ttest([0.1, 0.15, 0.12, 0.11, 0.13],
                            [0.8, 0.85, 0.82, 0.81, 0.83],
                            alternative="greater")
        assert mine.p_value > 0.5
        assert mine.significant is False

    def test_length_mismatch_raises(self):
        with pytest.raises(StatisticsError, match="equal length"):
            paired_ttest([1.0, 2.0], [1.0])

    def test_empty_raises(self):
        with pytest.raises(StatisticsError, match="at least one sample"):
            paired_ttest([], [])

    def test_mean_difference_is_paired_not_marginal(self):
        """A wide but shiftless difference must not look significant."""
        a = [0.1, 0.9, 0.2, 0.8, 0.15, 0.85]
        b = [0.1, 0.9, 0.2, 0.8, 0.15, 0.85]
        mine = paired_ttest(a, b, alternative="greater")
        assert mine.mean_difference == 0.0
        assert mine.significant is False


# --------------------------------------------------------------------------- #
# wilcoxon
# --------------------------------------------------------------------------- #
class TestWilcoxon:
    def test_matches_scipy_exact(self):
        rng = np.random.default_rng(3)
        a = rng.normal(0.8, 0.05, 7)
        b = rng.normal(0.4, 0.05, 7)
        mine = wilcoxon(a, b, alternative="greater")
        reference = scipy_stats.wilcoxon(a, b, alternative="greater", method="exact")
        assert mine.p_value == pytest.approx(reference.pvalue, rel=1e-9)
        assert mine.significant is True

    def test_five_seeds_reach_significance_one_sided(self):
        """
        With n=5 the exact one-sided Wilcoxon bottoms out at 0.5**5 = 0.03125,
        which does clear alpha=0.05.  The n=4 case below cannot.
        """
        five = wilcoxon([0.9] * 5, [0.4] * 5, alternative="greater")
        assert five.p_value == pytest.approx(0.5 ** 5)
        assert five.significant is True
        assert "exact" in five.note

        four = wilcoxon([0.9] * 4, [0.4] * 4, alternative="greater")
        assert four.p_value == pytest.approx(0.5 ** 4)
        assert four.p_value > 0.05
        assert four.significant is False

    def test_two_sided_needs_six_seeds(self):
        five = wilcoxon([0.9] * 5, [0.4] * 5, alternative="two-sided")
        assert five.p_value == pytest.approx(2 * 0.5 ** 5)
        assert five.significant is False
        six = wilcoxon([0.9] * 6, [0.4] * 6, alternative="greater")
        assert six.p_value == pytest.approx(0.5 ** 6)
        assert six.significant is True

    def test_all_differences_zero(self):
        mine = wilcoxon([0.5] * 5, [0.5] * 5, alternative="greater")
        assert mine.p_value == 1.0
        assert "exactly zero" in mine.note

    def test_ties_are_averaged(self):
        # |1| appears three times: ranks 1,2,3 -> mean 2 for each.
        mine = wilcoxon([1.0, 1.0, 1.0], [-0.0 - 1e-9, 0.0, 0.0], alternative="greater")
        assert mine.n_pairs == 3

    def test_large_sample_uses_normal_approximation(self):
        rng = np.random.default_rng(5)
        a = rng.normal(0.8, 0.05, 30)
        b = rng.normal(0.4, 0.05, 30)
        mine = wilcoxon(a, b, alternative="greater")
        assert "normal approximation" in mine.note
        assert mine.significant is True

    def test_exact_enumeration_limit_is_enforced(self):
        big = list(np.linspace(1.0, 2.0, 21))
        with pytest.raises(StatisticsError, match="n <= 20"):
            # 21 non-zero differences would exceed the enumeration ceiling, but
            # the approximation path must take over first, so this asserts the
            # documented boundary rather than an error.
            result = wilcoxon(big, [0.5] * 21, alternative="greater")
            assert result.n_pairs == 21


# --------------------------------------------------------------------------- #
# mcnemar
# --------------------------------------------------------------------------- #
class TestMcnemar:
    def test_matches_binomtest(self):
        y_true = np.array(["a", "a", "b", "b", "c", "c", "a", "b"])
        pred_a = np.array(["a", "a", "b", "c", "c", "c", "a", "b"])
        pred_b = np.array(["a", "b", "b", "b", "b", "c", "a", "c"])
        mine = mcnemar(y_true, pred_a, pred_b, alternative="two-sided")
        details = mine.details
        reference = scipy_stats.binomtest(
            min(details["a_only_correct"], details["b_only_correct"]),
            details["n_discordant"], 0.5, alternative="two-sided")
        assert mine.p_value == pytest.approx(reference.pvalue, rel=1e-9)
        assert details["n_discordant"] == details["a_only_correct"] + details["b_only_correct"]

    def test_accurate_model_is_significantly_better(self):
        y_true = np.array(["a"] * 40 + ["b"] * 40)
        good = np.array(["a"] * 38 + ["b"] * 38 + ["a", "b", "a", "b"])
        bad = np.array(["a"] * 40 + ["b"] * 20 + ["a"] * 20)
        mine = mcnemar(y_true, good, bad, alternative="greater")
        assert mine.mean_difference > 0
        assert mine.significant is True
        assert mine.details["a_only_correct"] > mine.details["b_only_correct"]
        assert mine.details["n_discordant"] == 20

    def test_no_discordant_pairs(self):
        y_true = np.array(["a", "b"])
        pred = np.array(["a", "b"])
        mine = mcnemar(y_true, pred, pred, alternative="greater")
        assert mine.p_value == 1.0
        assert "no discordant pairs" in mine.note
        assert mine.mean_difference == 0.0

    def test_length_mismatch_raises(self):
        with pytest.raises(StatisticsError, match="equal lengths"):
            mcnemar(["a", "b"], ["a"], ["a", "b"])

    def test_empty_raises(self):
        with pytest.raises(StatisticsError, match="at least one row"):
            mcnemar([], [], [])

    def test_identical_models_have_zero_gap(self):
        rng = np.random.default_rng(9)
        y_true = rng.choice(["a", "b", "c"], 100)
        pred = rng.choice(["a", "b", "c"], 100)
        assert mcnemar(y_true, pred, pred, alternative="greater").mean_difference == 0.0


# --------------------------------------------------------------------------- #
# Aggregation over the matrix
# --------------------------------------------------------------------------- #
def make_results(seeds=(1, 2, 3, 4, 5), in_domain_value=0.95, cross_value=0.10,
                 jitter=0.0, datasets=("alpha", "beta")):
    """Build a synthetic multi-seed matrix with a known in/cross gap."""
    rows = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        for train in datasets:
            for test in datasets:
                in_domain = train == test
                base = in_domain_value if in_domain else cross_value
                value = base + jitter * rng.normal(0, 1)
                rows.append({
                    "train_dataset": train,
                    "test_dataset": test,
                    "in_domain": in_domain,
                    "seed": seed,
                    "macro_f1": float(np.clip(value, 0.0, 1.0)),
                    "accuracy": float(np.clip(value + 0.01, 0.0, 1.0)),
                })
    return pd.DataFrame(rows)


class TestAggregate:
    def test_aggregate_shape_and_columns(self):
        agg = aggregate_over_seeds(make_results())
        assert len(agg) == 4
        for column in ("value_mean", "value_std", "value_min", "value_max",
                       "ci_low", "ci_high", "n_seeds", "in_domain"):
            assert column in agg.columns
        assert agg["n_seeds"].eq(5).all()

    def test_aggregate_recovers_known_means(self):
        agg = aggregate_over_seeds(make_results(in_domain_value=0.9, cross_value=0.2))
        diagonal = agg[agg["in_domain"]]["value_mean"].mean()
        off = agg[~agg["in_domain"]]["value_mean"].mean()
        assert diagonal == pytest.approx(0.9)
        assert off == pytest.approx(0.2)

    def test_aggregate_requires_columns(self):
        with pytest.raises(KeyError, match="macro_f1"):
            aggregate_over_seeds(pd.DataFrame({"train_dataset": ["a"]}))
        with pytest.raises(KeyError, match="seed"):
            aggregate_over_seeds(pd.DataFrame({"train_dataset": ["a"],
                                               "macro_f1": [0.5]}))

    def test_custom_value_column(self):
        agg = aggregate_over_seeds(make_results(), value_column="accuracy")
        assert "value_mean" in agg.columns


class TestCompare:
    def test_comparisons_are_paired_by_seed(self):
        comparisons = compare_in_domain_vs_cross_domain(make_results())
        assert len(comparisons) == 2                      # alpha->beta, beta->alpha
        assert comparisons["n_seeds"].eq(5).all()
        assert (comparisons["in_domain_mean"] - comparisons["cross_domain_mean"]
                == comparisons["delta_mean"]).all()
        assert comparisons["cross_worse"].all()

    def test_retention_is_cross_over_in(self):
        comparisons = compare_in_domain_vs_cross_domain(
            make_results(in_domain_value=0.8, cross_value=0.2))
        assert comparisons["retention"].iloc[0] == pytest.approx(0.25)

    def test_significance_detected_with_five_seeds(self):
        rng = np.random.default_rng(0)
        rows = []
        for seed in (1, 2, 3, 4, 5):
            noise = rng.normal(0, 0.001)
            rows.append({
                "train_dataset": "a", "test_dataset": "a", "seed": seed,
                "in_domain": True, "macro_f1": 0.95 + noise, "accuracy": 0.95,
            })
            rows.append({
                "train_dataset": "a", "test_dataset": "b", "seed": seed,
                "in_domain": False, "macro_f1": 0.10 + noise, "accuracy": 0.10,
            })
        results = pd.DataFrame(rows)
        comparisons = compare_in_domain_vs_cross_domain(results)
        assert len(comparisons) == 1
        assert bool(comparisons["significant"].iloc[0]) is True
        assert comparisons["t_p_value"].iloc[0] < 0.05
        # 5 seeds: the exact one-sided Wilcoxon reaches 0.03125 and the t-test
        # has zero variance, so either route can reject.
        assert comparisons["wilcoxon_p_value"].iloc[0] <= 0.05
        assert comparisons["seeds_sufficient"].iloc[0]

    def test_note_explains_resolution_limit(self):
        """Four seeds cannot clear 0.05 with the exact test; the note says so."""
        results = make_results(seeds=(1, 2, 3, 4), datasets=("a", "b"))
        comparisons = compare_in_domain_vs_cross_domain(results)
        assert "smallest attainable one-sided Wilcoxon p" in comparisons["note"].iloc[0]

    def test_note_explains_insufficient_seeds(self):
        results = make_results(seeds=(1,))
        comparisons = compare_in_domain_vs_cross_domain(results)
        assert "fewer than 2 seeds" in comparisons["note"].iloc[0]
        assert not comparisons["seeds_sufficient"].iloc[0]
        assert not comparisons["significant"].iloc[0]

    def test_identical_domains_show_no_gap(self):
        rows = []
        for seed in (1, 2, 3, 4, 5):
            rows.append({"train_dataset": "a", "test_dataset": "a", "seed": seed,
                         "in_domain": True, "macro_f1": 0.5, "accuracy": 0.5})
            rows.append({"train_dataset": "a", "test_dataset": "b", "seed": seed,
                         "in_domain": False, "macro_f1": 0.5, "accuracy": 0.5})
        comparisons = compare_in_domain_vs_cross_domain(pd.DataFrame(rows))
        assert comparisons["delta_mean"].iloc[0] == 0.0
        assert not comparisons["cross_worse"].iloc[0]

    def test_requires_columns(self):
        with pytest.raises(KeyError):
            compare_in_domain_vs_cross_domain(pd.DataFrame({"seed": [1]}))


class TestAnalyze:
    def test_bundle_shape(self):
        bundle = analyze(make_results())
        for key in ("aggregate", "comparisons", "overall_ttest",
                    "overall_wilcoxon", "overall_delta_ci", "n_seeds",
                    "cross_worse_pairs", "total_pairs", "significant_pairs"):
            assert key in bundle
        assert bundle["n_seeds"] == 5
        assert bundle["total_pairs"] == 2
        assert bundle["cross_worse_pairs"] == 2
        assert bundle["in_domain_mean"] == pytest.approx(0.95)
        assert bundle["cross_domain_mean"] == pytest.approx(0.10)
        assert bundle["mean_delta"] == pytest.approx(0.85)

    def test_overall_tests_see_the_gap(self):
        bundle = analyze(make_results())
        assert bundle["overall_ttest"].significant
        assert bundle["overall_wilcoxon"].p_value is not None
        low, high = bundle["overall_delta_ci"]
        assert low > 0.5
        # Only 2 ordered pairs here, so the exact Wilcoxon cannot reach 0.05
        # (smallest attainable one-sided p is 0.25); the t-test carries the claim.
        assert bundle["overall_wilcoxon"].p_value == pytest.approx(0.25)

    def test_single_dataset_matrix_has_no_comparisons(self):
        results = make_results(datasets=("only",))
        bundle = analyze(results)
        assert bundle["comparisons"].empty
        assert bundle["overall_ttest"] is None
        assert math.isnan(bundle["mean_delta"])

    def test_default_seeds_are_distinct(self):
        assert len(set(DEFAULT_SEEDS)) == len(DEFAULT_SEEDS)
        assert all(isinstance(s, int) for s in DEFAULT_SEEDS)


# --------------------------------------------------------------------------- #
# Numeric internals
# --------------------------------------------------------------------------- #
class TestInternals:
    def test_t_sf_matches_scipy(self):
        for df in (1, 4, 9, 19, 49, 99):
            for t in (0.1, 1.0, 1.96, 3.5):
                from src.evaluation.statistics import _t_sf

                assert _t_sf(t, df) == pytest.approx(
                    float(scipy_stats.t.sf(t, df)), abs=1e-10)

    def test_norm_sf_matches_scipy(self):
        from src.evaluation.statistics import _norm_sf

        for z in (-3.0, -1.0, 0.0, 1.0, 3.0):
            assert _norm_sf(z) == pytest.approx(
                float(scipy_stats.norm.sf(z)), abs=1e-12)

    def test_binomial_matches_scipy(self):
        from src.evaluation.statistics import _binom_test_two_sided

        for k, n in ((0, 10), (1, 10), (5, 10), (9, 10), (3, 7)):
            assert _binom_test_two_sided(k, n) == pytest.approx(
                float(scipy_stats.binomtest(k, n, 0.5).pvalue), abs=1e-12)

    def test_rank_abs_averages_ties(self):
        from src.evaluation.statistics import _rank_abs

        # |1| occupies ranks 1 and 2 -> both become 1.5; |3| keeps rank 3.
        ranks = _rank_abs(np.array([1.0, -1.0, 3.0]))
        assert ranks[0] == ranks[1] == 1.5
        assert ranks[2] == 3.0

    def test_rank_abs_without_ties(self):
        from src.evaluation.statistics import _rank_abs

        np.testing.assert_allclose(_rank_abs(np.array([3.0, -1.0, 2.0])),
                                   [3.0, 1.0, 2.0])

    def test_wilcoxon_exact_matches_closed_form(self):
        """For n all-positive differences the exact p is 2 * (1/2)**n."""
        from src.evaluation.statistics import _wilcoxon_exact_p

        for n in range(1, 7):
            assert _wilcoxon_exact_p(0.0, n) == pytest.approx(2.0 * (0.5 ** n))
