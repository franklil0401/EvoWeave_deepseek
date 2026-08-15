"""Unit tests for the pure-Python statistics helpers used by benchmark reports."""

from __future__ import annotations

import pytest

from evoweave_ds.benchmarking.statistics import (
    compare_success_rates,
    fisher_exact_p,
    mann_whitney_u,
)


class TestFisherExact:
    def test_known_classic_case(self) -> None:
        # 8/10 vs 2/10: strongly higher dynamic rate, one-sided p should be small.
        p = fisher_exact_p(8, 10, 2, 10)
        assert 0.0 < p < 0.05

    def test_equal_rates(self) -> None:
        p = fisher_exact_p(5, 10, 5, 10)
        assert 0.0 <= p <= 1.0

    def test_perfect_separation(self) -> None:
        p = fisher_exact_p(10, 10, 0, 10)
        assert p <= 1.0 / 184756 + 1e-12  # C(20,10) denominator, single table

    def test_no_dynamic_success(self) -> None:
        # 0/10 vs 10/10: dynamic is worse; upper tail still valid but large.
        p = fisher_exact_p(0, 10, 10, 10)
        assert 0.0 <= p <= 1.0

    def test_invalid_counts(self) -> None:
        with pytest.raises(ValueError, match="成功次数超出"):
            fisher_exact_p(11, 10, 0, 10)


class TestMannWhitney:
    def test_separated_groups(self) -> None:
        u, p = mann_whitney_u((1.0, 2.0, 3.0), (4.0, 5.0, 6.0))
        assert u == 0.0
        assert 0.0 < p < 0.2  # small-sample approximation, direction clear

    def test_identical_groups(self) -> None:
        u, p = mann_whitney_u((1.0, 2.0, 3.0), (1.0, 2.0, 3.0))
        assert u >= 0.0
        assert p > 0.05  # no evidence of a difference

    def test_requires_nonempty(self) -> None:
        with pytest.raises(ValueError, match="非空样本"):
            mann_whitney_u((), (1.0, 2.0))


class TestCompareSuccessRates:
    def test_delta_sign(self) -> None:
        result = compare_success_rates(8, 10, 2, 10)
        assert result["rate_delta"] == pytest.approx(0.6)
        assert result["fisher_p"] < 0.05

    def test_zero_totals(self) -> None:
        result = compare_success_rates(0, 0, 0, 0)
        assert result["dynamic_rate"] == 0.0
        assert result["baseline_rate"] == 0.0
