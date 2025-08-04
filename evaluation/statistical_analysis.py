import numpy as np
from scipy import stats
from typing import List, Dict, Any, Tuple, Optional
import logging #statistical_analysis.py
import pandas as pd


class StatisticalAnalyzer:
    """Perform statistical significance testing on attack and defense results"""

    def __init__(self, alpha: float = 0.05):
        """
        Initialize statistical analyzer

        Args:
            alpha: Significance level for hypothesis testing
        """
        self.alpha = alpha
        self.logger = logging.getLogger(__name__)

    def paired_t_test(self, before: List[float], after: List[float]) -> Dict[str, Any]:
        """
        Perform paired t-test for before/after comparison

        Args:
            before: List of values before treatment
            after: List of values after treatment

        Returns:
            Dictionary with test results
        """
        if len(before) != len(after):
            raise ValueError("Before and after samples must have the same length")

        t_statistic, p_value = stats.ttest_rel(before, after)

        result = {
            't_statistic': t_statistic,
            'p_value': p_value,
            'significant': p_value < self.alpha,
            'effect_size': self.cohens_d_paired(before, after),
            'mean_difference': np.mean(after) - np.mean(before),
            'confidence_interval': self.paired_confidence_interval(before, after)
        }

        self.logger.info(f"Paired t-test: t={t_statistic:.3f}, p={p_value:.6f}, "
                         f"significant={result['significant']}")

        return result

    def independent_t_test(self, group1: List[float], group2: List[float],
                           equal_var: bool = False) -> Dict[str, Any]:
        """
        Perform independent samples t-test

        Args:
            group1: First group of values
            group2: Second group of values
            equal_var: Assume equal variances

        Returns:
            Dictionary with test results
        """
        t_statistic, p_value = stats.ttest_ind(group1, group2, equal_var=equal_var)

        result = {
            't_statistic': t_statistic,
            'p_value': p_value,
            'significant': p_value < self.alpha,
            'effect_size': self.cohens_d_independent(group1, group2),
            'mean_difference': np.mean(group1) - np.mean(group2),
            'group1_stats': self.descriptive_stats(group1),
            'group2_stats': self.descriptive_stats(group2)
        }

        self.logger.info(f"Independent t-test: t={t_statistic:.3f}, p={p_value:.6f}, "
                         f"significant={result['significant']}")

        return result

    def one_way_anova(self, *groups: List[float]) -> Dict[str, Any]:
        """
        Perform one-way ANOVA test

        Args:
            *groups: Variable number of groups to compare

        Returns:
            Dictionary with ANOVA results
        """
        f_statistic, p_value = stats.f_oneway(*groups)

        # Effect size (eta-squared)
        eta_squared = self.eta_squared(*groups)

        result = {
            'f_statistic': f_statistic,
            'p_value': p_value,
            'significant': p_value < self.alpha,
            'eta_squared': eta_squared,
            'num_groups': len(groups),
            'group_stats': [self.descriptive_stats(group) for group in groups]
        }

        # Post-hoc analysis if significant
        if result['significant'] and len(groups) > 2:
            result['posthoc'] = self.tukey_hsd(*groups)

        self.logger.info(f"One-way ANOVA: F={f_statistic:.3f}, p={p_value:.6f}, "
                         f"significant={result['significant']}")

        return result

    def mann_whitney_u(self, group1: List[float], group2: List[float]) -> Dict[str, Any]:
        """
        Perform Mann-Whitney U test (non-parametric)

        Args:
            group1: First group of values
            group2: Second group of values

        Returns:
            Dictionary with test results
        """
        u_statistic, p_value = stats.mannwhitneyu(group1, group2, alternative='two-sided')

        # Effect size (rank-biserial correlation)
        r = 1 - (2 * u_statistic) / (len(group1) * len(group2))

        result = {
            'u_statistic': u_statistic,
            'p_value': p_value,
            'significant': p_value < self.alpha,
            'effect_size_r': r,
            'median_difference': np.median(group1) - np.median(group2),
            'group1_median': np.median(group1),
            'group2_median': np.median(group2)
        }

        self.logger.info(f"Mann-Whitney U test: U={u_statistic:.3f}, p={p_value:.6f}, "
                         f"significant={result['significant']}")

        return result

    def wilcoxon_signed_rank(self, before: List[float], after: List[float]) -> Dict[str, Any]:
        """
        Perform Wilcoxon signed-rank test (non-parametric paired test)

        Args:
            before: Values before treatment
            after: Values after treatment

        Returns:
            Dictionary with test results
        """
        if len(before) != len(after):
            raise ValueError("Before and after samples must have the same length")

        w_statistic, p_value = stats.wilcoxon(before, after)

        # Effect size (r = Z / sqrt(N))
        z_score = stats.norm.ppf(p_value / 2)
        r = abs(z_score) / np.sqrt(len(before))

        result = {
            'w_statistic': w_statistic,
            'p_value': p_value,
            'significant': p_value < self.alpha,
            'effect_size_r': r,
            'median_difference': np.median(after) - np.median(before)
        }

        self.logger.info(f"Wilcoxon signed-rank test: W={w_statistic:.3f}, p={p_value:.6f}, "
                         f"significant={result['significant']}")

        return result

    def kruskal_wallis(self, *groups: List[float]) -> Dict[str, Any]:
        """
        Perform Kruskal-Wallis test (non-parametric ANOVA)

        Args:
            *groups: Variable number of groups to compare

        Returns:
            Dictionary with test results
        """
        h_statistic, p_value = stats.kruskal(*groups)

        # Effect size (eta-squared approximation)
        n_total = sum(len(group) for group in groups)
        eta_squared_approx = (h_statistic - len(groups) + 1) / (n_total - len(groups))

        result = {
            'h_statistic': h_statistic,
            'p_value': p_value,
            'significant': p_value < self.alpha,
            'eta_squared_approx': max(0, eta_squared_approx),  # Ensure non-negative
            'num_groups': len(groups),
            'group_medians': [np.median(group) for group in groups]
        }

        self.logger.info(f"Kruskal-Wallis test: H={h_statistic:.3f}, p={p_value:.6f}, "
                         f"significant={result['significant']}")

        return result

    def cohens_d_paired(self, before: List[float], after: List[float]) -> float:
        """Calculate Cohen's d for paired samples"""
        differences = np.array(after) - np.array(before)
        return np.mean(differences) / np.std(differences, ddof=1)

    def cohens_d_independent(self, group1: List[float], group2: List[float]) -> float:
        """Calculate Cohen's d for independent samples"""
        n1, n2 = len(group1), len(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

        # Pooled standard deviation
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

        return (np.mean(group1) - np.mean(group2)) / pooled_std

    def eta_squared(self, *groups: List[float]) -> float:
        """Calculate eta-squared effect size for ANOVA"""
        # Calculate between-group sum of squares
        grand_mean = np.mean([val for group in groups for val in group])
        ss_between = sum(len(group) * (np.mean(group) - grand_mean) ** 2 for group in groups)

        # Calculate total sum of squares
        ss_total = sum((val - grand_mean) ** 2 for group in groups for val in group)

        return ss_between / ss_total if ss_total > 0 else 0

    def paired_confidence_interval(self, before: List[float], after: List[float],
                                   confidence: float = 0.95) -> Tuple[float, float]:
        """Calculate confidence interval for paired difference"""
        differences = np.array(after) - np.array(before)
        mean_diff = np.mean(differences)
        se_diff = stats.sem(differences)

        alpha = 1 - confidence
        t_critical = stats.t.ppf(1 - alpha / 2, len(differences) - 1)

        margin_error = t_critical * se_diff
        return (mean_diff - margin_error, mean_diff + margin_error)

    def tukey_hsd(self, *groups: List[float]) -> List[Dict[str, Any]]:
        """Perform Tukey HSD post-hoc test"""
        from itertools import combinations

        results = []
        group_names = [f"Group_{i + 1}" for i in range(len(groups))]

        for i, j in combinations(range(len(groups)), 2):
            group1, group2 = groups[i], groups[j]

            # Calculate Tukey HSD
            n1, n2 = len(group1), len(group2)
            mean_diff = np.mean(group1) - np.mean(group2)

            # MSE calculation (simplified)
            all_values = [val for group in groups for val in group]
            group_means = [np.mean(group) for group in groups]
            mse = np.var(all_values, ddof=len(groups))

            se = np.sqrt(mse * (1 / n1 + 1 / n2))

            # Critical value approximation
            q_critical = 3.0  # Simplified - should use proper q-table
            hsd = q_critical * se

            significant = abs(mean_diff) > hsd

            results.append({
                'group1': group_names[i],
                'group2': group_names[j],
                'mean_difference': mean_diff,
                'hsd_threshold': hsd,
                'significant': significant
            })

        return results

    def descriptive_stats(self, data: List[float]) -> Dict[str, float]:
        """Calculate descriptive statistics"""
        data = np.array(data)
        return {
            'mean': np.mean(data),
            'median': np.median(data),
            'std': np.std(data, ddof=1),
            'min': np.min(data),
            'max': np.max(data),
            'q25': np.percentile(data, 25),
            'q75': np.percentile(data, 75),
            'n': len(data)
        }

    def multiple_comparisons_correction(self, p_values: List[float],
                                        method: str = 'bonferroni') -> List[float]:
        """Apply multiple comparisons correction"""
        p_values = np.array(p_values)

        if method == 'bonferroni':
            return np.minimum(p_values * len(p_values), 1.0).tolist()
        elif method == 'holm':
            # Holm-Bonferroni method
            sorted_indices = np.argsort(p_values)
            corrected = np.zeros_like(p_values)

            for i, idx in enumerate(sorted_indices):
                corrected[idx] = min(p_values[idx] * (len(p_values) - i), 1.0)

            return corrected.tolist()
        else:
            raise ValueError(f"Unknown correction method: {method}")

    def generate_statistical_report(self, results: Dict[str, Any]) -> str:
        """Generate a formatted statistical report"""
        report = []
        report.append("STATISTICAL ANALYSIS REPORT")
        report.append("=" * 40)
        report.append("")

        for test_name, test_results in results.items():
            report.append(f"{test_name.upper()}:")

            if 'p_value' in test_results:
                p_val = test_results['p_value']
                significant = test_results.get('significant', False)

                report.append(f"  p-value: {p_val:.6f} {'*' if significant else ''}")
                report.append(f"  Significant: {'Yes' if significant else 'No'}")

                if 'effect_size' in test_results:
                    effect_size = test_results['effect_size']
                    report.append(f"  Effect size: {effect_size:.3f}")

                if 'confidence_interval' in test_results:
                    ci = test_results['confidence_interval']
                    report.append(f"  95% CI: ({ci[0]:.3f}, {ci[1]:.3f})")

            report.append("")

        return "\n".join(report)


# Example usage
if __name__ == "__main__":
    analyzer = StatisticalAnalyzer()

    # Example data
    before_defense = [25.2, 27.1, 24.8, 26.3, 25.9]  # PSNR values
    after_defense = [18.1, 19.4, 17.8, 18.9, 18.6]  # PSNR values (lower = better defense)

    # Perform paired t-test
    result = analyzer.paired_t_test(before_defense, after_defense)
    print("Paired t-test results:")
    print(f"  p-value: {result['p_value']:.6f}")
    print(f"  Significant: {result['significant']}")
    print(f"  Effect size: {result['effect_size']:.3f}")

    # Generate report
    report = analyzer.generate_statistical_report({'paired_t_test': result})
    print("\n" + report)
