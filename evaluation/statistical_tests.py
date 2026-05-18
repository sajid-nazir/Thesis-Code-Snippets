"""Bootstrap confidence intervals and statistical tests for evaluation scores."""

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score


def bootstrap_ci(scores, n_boot=10000, ci=95):
    """Compute bootstrap confidence interval for the mean.

    Args:
        scores: array-like of scores
        n_boot: number of bootstrap resamples
        ci: confidence level (default 95%)

    Returns:
        (lower, upper) bounds of the CI
    """
    scores = np.array(scores)
    means = []
    for _ in range(n_boot):
        sample = np.random.choice(scores, size=len(scores), replace=True)
        means.append(sample.mean())
    means = sorted(means)
    lower = means[int(n_boot * (1 - ci / 100) / 2)]
    upper = means[int(n_boot * (1 + ci / 100) / 2)]
    return round(float(lower), 4), round(float(upper), 4)


def paired_bootstrap(scores_a, scores_b, n_boot=10000):
    """Paired bootstrap test: is system A significantly better than B?

    Args:
        scores_a: per-question scores for system A
        scores_b: per-question scores for system B (same questions, aligned)

    Returns:
        dict with mean_diff, ci_95, p_positive
    """
    diff = np.array(scores_a) - np.array(scores_b)
    means = []
    for _ in range(n_boot):
        sample = np.random.choice(diff, size=len(diff), replace=True)
        means.append(sample.mean())
    means = sorted(means)
    ci_lower = means[int(n_boot * 0.025)]
    ci_upper = means[int(n_boot * 0.975)]
    p_positive = sum(1 for m in means if m > 0) / n_boot

    return {
        "mean_diff": round(float(np.mean(diff)), 4),
        "ci_95": (round(float(ci_lower), 4), round(float(ci_upper), 4)),
        "p_positive": round(float(p_positive), 4),
        "significant": ci_lower > 0 or ci_upper < 0,
    }


def judge_calibration(human_scores, llm_scores):
    """Compute agreement metrics between human and LLM judge scores.

    Args:
        human_scores: list of human-assigned scores (0-3)
        llm_scores: list of LLM-assigned scores (0-3), aligned

    Returns:
        dict with kappa, spearman, exact agreement, within-1 agreement
    """
    kappa_linear = cohen_kappa_score(human_scores, llm_scores, weights='linear')
    kappa_quadratic = cohen_kappa_score(human_scores, llm_scores, weights='quadratic')
    rho, p_value = spearmanr(human_scores, llm_scores)

    exact = sum(1 for h, l in zip(human_scores, llm_scores) if h == l)
    within_one = sum(1 for h, l in zip(human_scores, llm_scores) if abs(h - l) <= 1)
    n = len(human_scores)

    return {
        "n": n,
        "kappa_linear": round(float(kappa_linear), 4),
        "kappa_quadratic": round(float(kappa_quadratic), 4),
        "spearman_rho": round(float(rho), 4),
        "spearman_p": round(float(p_value), 6),
        "exact_agreement_pct": round(exact / n * 100, 1),
        "within_one_agreement_pct": round(within_one / n * 100, 1),
    }
