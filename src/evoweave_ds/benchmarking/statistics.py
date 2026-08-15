"""Small pure-Python statistical tests for benchmark reports (no scipy dependency).

The benchmark matrix compares one dynamic strategy group against baseline
groups over repeated trials.  We implement the two tests the report needs:

- Fisher's exact test (2x2 contingency) for success-rate differences, with
  exact one-sided p-value via hypergeometric tail sums;
- Mann-Whitney U test (Wilcoxon rank-sum) for token/latency comparisons,
  using the continuity corrected normal approximation.

All functions are deterministic and pure; the report embeds the resulting
p-values as evidence for the pre-registered thresholds in 实验方案.md.
"""

from __future__ import annotations

from math import comb, erf, sqrt

__all__ = [
    "compare_success_rates",
    "fisher_exact_p",
    "mann_whitney_u",
]


def _hypergeom_tail(a: int, b: int, c: int, d: int) -> float:
    """One-sided Fisher exact p-value: P(>= observed association).

    Table layout::

              success  failure
        dyn      a        b
        base     c        d

    The p-value is the sum of hypergeometric probabilities for tables
    with at least as much association as observed (upper tail).
    """

    n1 = a + b  # dynamic trials
    n2 = c + d  # baseline trials
    n = n1 + n2
    k = a + c  # total successes
    if min(n1, n2, k, n - k) < 0:
        raise ValueError("负计数不能进入 Fisher 精确检验")
    lo = max(0, k - n2)
    hi = min(n1, k)
    total = 0.0
    for x in range(a, hi + 1):
        if x > hi or x < lo:
            continue
        ways = comb(n1, x) * comb(n2, k - x)
        total += ways
    return total / comb(n, k)


def fisher_exact_p(
    dynamic_passed: int,
    dynamic_total: int,
    baseline_passed: int,
    baseline_total: int,
) -> float:
    """One-sided p-value that dynamic success rate exceeds baseline."""

    if dynamic_total < 0 or baseline_total < 0:
        raise ValueError("试验次数不能为负")
    if not (0 <= dynamic_passed <= dynamic_total and 0 <= baseline_passed <= baseline_total):
        raise ValueError("成功次数超出试验次数")
    a = dynamic_passed
    b = dynamic_total - dynamic_passed
    c = baseline_passed
    d = baseline_total - baseline_passed
    return _hypergeom_tail(a, b, c, d)


def _rank_sum(sample: tuple[float, ...]) -> float:
    """Sum of ranks with average ranks for ties."""

    combined = sorted((value, index) for index, value in enumerate(sample))
    ranks = [0.0] * len(sample)
    i = 0
    while i < len(combined):
        j = i
        while j + 1 < len(combined) and combined[j + 1][0] == combined[i][0]:
            j += 1
        average = (i + j) / 2 + 1.0  # 1-based average rank
        for k in range(i, j + 1):
            ranks[combined[k][1]] = average
        i = j + 1
    return sum(ranks)


def _normal_two_tailed(z: float) -> float:
    return 2.0 * (1.0 - 0.5 * (1.0 + erf(abs(z) / sqrt(2.0))))


def mann_whitney_u(
    group_a: tuple[float, ...],
    group_b: tuple[float, ...],
) -> tuple[float, float]:
    """Two-sided Mann-Whitney U p-value.

    Returns (u_statistic, p_value).  Uses the continuity corrected normal
    approximation, which is adequate for the matrix sample sizes here.
    """

    na, nb = len(group_a), len(group_b)
    if na == 0 or nb == 0:
        raise ValueError("Mann-Whitney U 需要两个非空样本")
    rank_a = _rank_sum(group_a)
    u_a = rank_a - na * (na + 1) / 2.0
    u_b = na * nb - u_a
    u_stat = min(u_a, u_b)
    mu = na * nb / 2.0
    sigma = sqrt(na * nb * (na + nb + 1) / 12.0)
    if u_stat <= mu:
        z = (u_stat + 0.5 - mu) / sigma
    else:
        z = (u_stat - 0.5 - mu) / sigma
    return u_stat, _normal_two_tailed(z)


def compare_success_rates(
    dynamic_passed: int,
    dynamic_total: int,
    baseline_passed: int,
    baseline_total: int,
) -> dict[str, float]:
    """Return success-rate comparison statistics for one baseline."""

    p = fisher_exact_p(dynamic_passed, dynamic_total, baseline_passed, baseline_total)
    dynamic_rate = dynamic_passed / dynamic_total if dynamic_total else 0.0
    baseline_rate = baseline_passed / baseline_total if baseline_total else 0.0
    return {
        "dynamic_rate": dynamic_rate,
        "baseline_rate": baseline_rate,
        "rate_delta": dynamic_rate - baseline_rate,
        "fisher_p": p,
    }
