import json
import os
from datetime import datetime
from typing import Dict, Any, List, Optional #report_generator.py
import logging
import matplotlib.pyplot as plt
import numpy as np


class ReportGenerator:
    """Generate comprehensive automated reports from evaluation results"""

    def __init__(self, save_dir: str = './reports'):
        """
        Initialize report generator

        Args:
            save_dir: Directory to save generated reports
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        # Setup logging
        self.logger = logging.getLogger(__name__)

    def generate_comprehensive_report(self, attack_results: Dict[str, Any],
                                      defense_results: Dict[str, Any],
                                      statistical_results: Optional[Dict[str, Any]] = None,
                                      metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate comprehensive evaluation report

        Args:
            attack_results: Attack evaluation results
            defense_results: Defense evaluation results
            statistical_results: Statistical analysis results
            metadata: Additional metadata

        Returns:
            Path to generated report file
        """
        report_lines = []

        # Header
        report_lines.extend(self._generate_header(metadata))

        # Executive Summary
        report_lines.extend(self._generate_executive_summary(attack_results, defense_results))

        # Attack Analysis
        report_lines.extend(self._generate_attack_analysis(attack_results))

        # Defense Analysis
        report_lines.extend(self._generate_defense_analysis(defense_results))

        # Statistical Analysis
        if statistical_results:
            report_lines.extend(self._generate_statistical_analysis(statistical_results))

        # Key Findings
        report_lines.extend(self._generate_key_findings(attack_results, defense_results))

        # Recommendations
        report_lines.extend(self._generate_recommendations(defense_results))

        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = os.path.join(self.save_dir, f'comprehensive_report_{timestamp}.txt')

        with open(report_file, 'w') as f:
            f.write('\n'.join(report_lines))

        self.logger.info(f"Comprehensive report saved to {report_file}")
        return report_file

    def _generate_header(self, metadata: Optional[Dict[str, Any]]) -> List[str]:
        """Generate report header"""
        lines = []
        lines.append("GRADIENT INVERSION ATTACKS ON PEFT METHODS")
        lines.append("COMPREHENSIVE EVALUATION REPORT")
        lines.append("=" * 60)
        lines.append("")
        lines.append(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        if metadata:
            lines.append(f"Experiment ID: {metadata.get('experiment_id', 'N/A')}")
            lines.append(f"Dataset: {metadata.get('dataset', 'CIFAR-100')}")
            lines.append(f"Device: {metadata.get('device', 'CPU')}")
            lines.append(f"Evaluation Samples: {metadata.get('num_samples', 'N/A')}")

        lines.append("")
        return lines

    def _generate_executive_summary(self, attack_results: Dict[str, Any],
                                    defense_results: Dict[str, Any]) -> List[str]:
        """Generate executive summary"""
        lines = []
        lines.append("EXECUTIVE SUMMARY")
        lines.append("-" * 20)
        lines.append("")

        # Attack summary
        if 'comparative_analysis' in attack_results:
            comp = attack_results['comparative_analysis']
            summary_stats = comp.get('summary_statistics', {})

            lines.append(f"• {summary_stats.get('methods_evaluated', 0)} PEFT methods evaluated")
            lines.append(f"• Average attack success rate: {summary_stats.get('avg_success_rate', 0):.1%}")
            lines.append(f"• Average PSNR: {summary_stats.get('avg_psnr', 0):.1f} dB")

            # Most vulnerable method
            ranking = comp.get('vulnerability_ranking', [])
            if ranking:
                most_vulnerable = ranking[0]
                lines.append(f"• Most vulnerable PEFT method: {most_vulnerable[0].upper()} "
                             f"({most_vulnerable[1]:.1%} success rate)")

        # Defense summary
        if 'overall_defense_performance' in defense_results:
            overall = defense_results['overall_defense_performance']
            lines.append(f"• Average defense effectiveness: {overall.get('avg_effectiveness', 0):.1%}")
            lines.append(f"• Best overall defense: {overall.get('best_overall_defense', 'N/A')}")

        lines.append("")
        return lines

    def _generate_attack_analysis(self, attack_results: Dict[str, Any]) -> List[str]:
        """Generate attack analysis section"""
        lines = []
        lines.append("ATTACK ANALYSIS")
        lines.append("-" * 15)
        lines.append("")

        # Individual method results
        individual_results = attack_results.get('individual_results', {})

        lines.append("Attack Performance by PEFT Method:")
        lines.append("")

        for method, results in individual_results.items():
            if 'error' not in results:
                lines.append(f"{method.upper()}:")
                lines.append(f"  • Success Rate: {results.get('attack_success_rate', 0):.1%}")

                eval_metrics = results.get('evaluation_metrics', {})
                lines.append(f"  • Mean PSNR: {eval_metrics.get('psnr_mean', 0):.2f} dB")
                lines.append(f"  • Mean SSIM: {eval_metrics.get('ssim_mean', 0):.3f}")
                lines.append(f"  • Patches Recovered: {results.get('num_patches_recovered', 0)}")
                lines.append("")
            else:
                lines.append(f"{method.upper()}: FAILED - {results.get('error', 'Unknown error')}")
                lines.append("")

        # Vulnerability ranking
        if 'comparative_analysis' in attack_results:
            ranking = attack_results['comparative_analysis'].get('vulnerability_ranking', [])
            if ranking:
                lines.append("Vulnerability Ranking (Most to Least Vulnerable):")
                for i, (method, score) in enumerate(ranking, 1):
                    lines.append(f"  {i}. {method.upper()}: {score:.1%} success rate")
                lines.append("")

        return lines

    def _generate_defense_analysis(self, defense_results: Dict[str, Any]) -> List[str]:
        """Generate defense analysis section"""
        lines = []
        lines.append("DEFENSE ANALYSIS")
        lines.append("-" * 16)
        lines.append("")

        # Defense rankings
        rankings = defense_results.get('defense_rankings', [])
        if rankings:
            lines.append("Defense Effectiveness Rankings:")
            for i, (defense, score) in enumerate(rankings, 1):
                lines.append(f"  {i}. {defense.title()}: {score:.1%} effectiveness")
            lines.append("")

        # Best defense per PEFT method
        best_defenses = defense_results.get('best_defenses_by_method', {})
        if best_defenses:
            lines.append("Recommended Defense by PEFT Method:")
            for peft_method, defense_info in best_defenses.items():
                defense_name = defense_info.get('defense', 'N/A')
                effectiveness = defense_info.get('effectiveness', 0)
                lines.append(f"  • {peft_method.upper()}: {defense_name.title()} "
                             f"({effectiveness:.1%} effectiveness)")
            lines.append("")

        # Overall performance
        overall_perf = defense_results.get('overall_defense_performance', {})
        if overall_perf:
            lines.append("Overall Defense Performance:")
            lines.append(f"  • Average Effectiveness: {overall_perf.get('avg_effectiveness', 0):.1%}")
            lines.append(f"  • Standard Deviation: {overall_perf.get('std_effectiveness', 0):.1%}")
            lines.append(f"  • Most Vulnerable PEFT: {overall_perf.get('most_vulnerable_peft', 'N/A')}")
            lines.append("")

        return lines

    def _generate_statistical_analysis(self, statistical_results: Dict[str, Any]) -> List[str]:
        """Generate statistical analysis section"""
        lines = []
        lines.append("STATISTICAL ANALYSIS")
        lines.append("-" * 20)
        lines.append("")

        for test_name, test_results in statistical_results.items():
            lines.append(f"{test_name.replace('_', ' ').title()}:")

            if 'p_value' in test_results:
                p_val = test_results['p_value']
                significant = test_results.get('significant', False)

                lines.append(f"  • p-value: {p_val:.6f}")
                lines.append(f"  • Statistically significant: {'Yes' if significant else 'No'}")

                if 'effect_size' in test_results:
                    effect_size = test_results['effect_size']
                    magnitude = self._interpret_effect_size(abs(effect_size))
                    lines.append(f"  • Effect size: {effect_size:.3f} ({magnitude})")

                if 'confidence_interval' in test_results:
                    ci = test_results['confidence_interval']
                    lines.append(f"  • 95% Confidence Interval: ({ci[0]:.3f}, {ci[1]:.3f})")

            lines.append("")

        return lines

    def _generate_key_findings(self, attack_results: Dict[str, Any],
                               defense_results: Dict[str, Any]) -> List[str]:
        """Generate key findings section"""
        lines = []
        lines.append("KEY FINDINGS")
        lines.append("-" * 12)
        lines.append("")

        findings = []

        # Attack findings
        if 'comparative_analysis' in attack_results:
            comp = attack_results['comparative_analysis']
            ranking = comp.get('vulnerability_ranking', [])

            if ranking:
                most_vulnerable = ranking[0]
                least_vulnerable = ranking[-1]

                findings.append(f"1. {most_vulnerable[0].upper()} is the most vulnerable PEFT method "
                                f"with {most_vulnerable[1]:.1%} attack success rate")

                findings.append(f"2. {least_vulnerable[0].upper()} is the least vulnerable PEFT method "
                                f"with {least_vulnerable[1]:.1%} attack success rate")

        # Defense findings
        if 'defense_rankings' in defense_results:
            rankings = defense_results['defense_rankings']
            if rankings:
                best_defense = rankings[0]
                findings.append(f"3. {best_defense[0].title()} is the most effective defense "
                                f"with {best_defense[1]:.1%} average effectiveness")

        # Effectiveness matrix insights
        matrix_data = defense_results.get('defense_effectiveness_matrix', {})
        if matrix_data:
            matrix = np.array(matrix_data.get('matrix', []))
            if matrix.size > 0:
                min_effectiveness = np.min(matrix)
                max_effectiveness = np.max(matrix)

                findings.append(f"4. Defense effectiveness ranges from {min_effectiveness:.1%} "
                                f"to {max_effectiveness:.1%}")

        # Privacy-utility tradeoff
        findings.append("5. All tested defenses provide meaningful privacy protection "
                        "with varying degrees of utility impact")

        for i, finding in enumerate(findings, 1):
            lines.append(f"• {finding}")

        lines.append("")
        return lines

    def _generate_recommendations(self, defense_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations section"""
        lines = []
        lines.append("RECOMMENDATIONS")
        lines.append("-" * 15)
        lines.append("")

        recommendations = []

        # Best defense recommendations
        best_defenses = defense_results.get('best_defenses_by_method', {})
        if best_defenses:
            recommendations.append("Method-Specific Recommendations:")
            for peft_method, defense_info in best_defenses.items():
                defense_name = defense_info.get('defense', 'N/A')
                recommendations.append(f"  • For {peft_method.upper()}: Use {defense_name.title()} defense")

        recommendations.append("")
        recommendations.append("General Recommendations:")
        recommendations.append("  • Implement multiple defense layers for enhanced protection")
        recommendations.append("  • Consider privacy-utility tradeoffs based on application requirements")
        recommendations.append("  • Regularly evaluate defense effectiveness against new attack methods")
        recommendations.append("  • Use adaptive defenses that adjust based on detected threats")

        lines.extend(recommendations)
        lines.append("")

        return lines

    def _interpret_effect_size(self, effect_size: float) -> str:
        """Interpret Cohen's d effect size"""
        if effect_size < 0.2:
            return "negligible"
        elif effect_size < 0.5:
            return "small"
        elif effect_size < 0.8:
            return "medium"
        else:
            return "large"

    def generate_json_report(self, all_results: Dict[str, Any],
                             filename: str = 'evaluation_results.json') -> str:
        """
        Generate JSON report with all results

        Args:
            all_results: Dictionary containing all evaluation results
            filename: Output filename

        Returns:
            Path to saved JSON file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_file = os.path.join(self.save_dir, f'{timestamp}_{filename}')

        # Add metadata
        report_data = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'report_version': '1.0',
                'generator': 'PEFT Attack Evaluation Framework'
            },
            'results': all_results
        }

        with open(json_file, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)

        self.logger.info(f"JSON report saved to {json_file}")
        return json_file

    def generate_summary_table(self, attack_results: Dict[str, Any],
                               defense_results: Dict[str, Any]) -> str:
        """Generate summary table in markdown format"""
        lines = []
        lines.append("# Evaluation Summary Table")
        lines.append("")

        # Attack results table
        lines.append("## Attack Results")
        lines.append("")
        lines.append("| PEFT Method | Success Rate | Mean PSNR (dB) | Mean SSIM |")
        lines.append("|-------------|--------------|----------------|-----------|")

        individual_results = attack_results.get('individual_results', {})
        for method, results in individual_results.items():
            if 'error' not in results:
                success_rate = results.get('attack_success_rate', 0)
                eval_metrics = results.get('evaluation_metrics', {})
                psnr = eval_metrics.get('psnr_mean', 0)
                ssim = eval_metrics.get('ssim_mean', 0)

                lines.append(f"| {method.upper()} | {success_rate:.1%} | {psnr:.2f} | {ssim:.3f} |")

        lines.append("")

        # Defense effectiveness table
        matrix_data = defense_results.get('defense_effectiveness_matrix', {})
        if matrix_data:
            defense_names = matrix_data.get('defense_names', [])
            peft_methods = matrix_data.get('peft_methods', [])
            matrix = matrix_data.get('matrix', [])

            if matrix and defense_names and peft_methods:
                lines.append("## Defense Effectiveness")
                lines.append("")

                # Create table header
                header = "| PEFT Method |" + "|".join(f" {d.title()} " for d in defense_names) + "|"
                separator = "|-------------|" + "|".join("--------|" for _ in defense_names) + "|"

                lines.append(header)
                lines.append(separator)

                # Add data rows
                for i, peft_method in enumerate(peft_methods):
                    row_data = [f"{matrix[i][j]:.1%}" for j in range(len(defense_names))]
                    row = f"| {peft_method.upper()} |" + "|".join(f" {data} " for data in row_data) + "|"
                    lines.append(row)

        # Save markdown file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        md_file = os.path.join(self.save_dir, f'summary_table_{timestamp}.md')

        with open(md_file, 'w') as f:
            f.write('\n'.join(lines))

        self.logger.info(f"Summary table saved to {md_file}")
        return md_file


# Example usage
if __name__ == "__main__":
    generator = ReportGenerator()

    # Example data
    example_attack_results = {
        'comparative_analysis': {
            'vulnerability_ranking': [('adapter', 0.85), ('lora', 0.72), ('prefix', 0.58), ('bias', 0.41)],
            'summary_statistics': {
                'methods_evaluated': 4,
                'avg_success_rate': 0.64,
                'avg_psnr': 22.5
            }
        },
        'individual_results': {
            'adapter': {
                'attack_success_rate': 0.85,
                'evaluation_metrics': {'psnr_mean': 28.2, 'ssim_mean': 0.823},
                'num_patches_recovered': 124
            }
        }
    }

    example_defense_results = {
        'defense_rankings': [('dp', 0.92), ('instahide', 0.87), ('mixup', 0.76)],
        'overall_defense_performance': {
            'avg_effectiveness': 0.85,
            'best_overall_defense': 'dp'
        }
    }

    report_file = generator.generate_comprehensive_report(
        example_attack_results,
        example_defense_results,
        metadata={'experiment_id': 'EXP001', 'dataset': 'CIFAR-100'}
    )

    print(f"Example report generated: {report_file}")
