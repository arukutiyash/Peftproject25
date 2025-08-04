import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging #attack_evaluator.py
import json
import os
from datetime import datetime
import matplotlib.pyplot as plt


class AttackEvaluator:
    """Comprehensive evaluation of gradient inversion attacks across PEFT methods"""

    def __init__(self, device: str = 'cpu', save_dir: str = './attack_results'):
        """
        Initialize attack evaluator

        Args:
            device: Computing device
            save_dir: Directory to save evaluation results
        """
        self.device = device
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        # Import attack classes
        from attacks.adapter_attack import AdapterGradientInversionAttack
        from attacks.prefix_attack import PrefixGradientInversionAttack
        from attacks.bias_attack import BiasGradientInversionAttack
        from attacks.lora_attack import LoRAGradientInversionAttack
        from utils.metrics import MetricsCalculator
        from utils.data_processing import DataProcessor
        from utils.patch_recovery import PatchRecoverer
        from utils.visualization import ResultVisualizer

        # Initialize utility classes
        self.metrics_calculator = MetricsCalculator(device=device)
        self.data_processor = DataProcessor(device=device)
        self.patch_recoverer = PatchRecoverer(device=device)
        self.visualizer = ResultVisualizer()

        # Attack class mapping
        self.attack_classes = {
            'adapter': AdapterGradientInversionAttack,
            'prefix': PrefixGradientInversionAttack,
            'bias': BiasGradientInversionAttack,
            'lora': LoRAGradientInversionAttack
        }

        # Results storage
        self.evaluation_results = {}

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def evaluate_single_attack(self, model: nn.Module, peft_method: str,
                               data_loader, num_samples: int = 64,
                               attack_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Evaluate a single attack on one PEFT method

        Args:
            model: Model with specific PEFT method
            peft_method: Type of PEFT method
            data_loader: Data loader for evaluation
            num_samples: Number of samples to evaluate
            attack_config: Configuration for attack

        Returns:
            Evaluation results dictionary
        """
        self.logger.info(f"Starting evaluation of {peft_method} attack...")

        if peft_method not in self.attack_classes:
            raise ValueError(f"Unsupported PEFT method: {peft_method}")

        # Initialize attack
        attack_class = self.attack_classes[peft_method]
        attack_config = attack_config or {}
        attacker = attack_class(model, self.device, **attack_config)

        # Collect evaluation data
        all_original_patches = []
        all_recovered_patches = []
        all_attack_metrics = []
        sample_count = 0

        model.eval()

        for batch_idx, (images, labels) in enumerate(data_loader):
            if sample_count >= num_samples:
                break

            images = images.to(self.device)
            labels = labels.to(self.device)

            # Extract ground truth patches
            original_patches = self.data_processor.extract_patches(images.cpu())

            try:
                # Run attack
                attack_results = attacker.run_attack(images, labels, original_patches)

                recovered_patches = attack_results.get('recovered_patches', [])
                attack_metrics = attack_results.get('metrics', {})

                # Store results
                all_original_patches.extend(original_patches)
                all_recovered_patches.extend(recovered_patches)
                all_attack_metrics.append(attack_metrics)

                sample_count += images.size(0)

                if batch_idx % 10 == 0:
                    self.logger.info(f"Processed {sample_count} samples...")

            except Exception as e:
                self.logger.error(f"Attack failed on batch {batch_idx}: {e}")
                continue

        # Compute overall evaluation metrics
        evaluation_metrics = self.metrics_calculator.evaluate_reconstruction_quality(
            all_original_patches, all_recovered_patches
        )

        # Compute attack success rate
        psnr_values = []
        for orig, rec in zip(all_original_patches, all_recovered_patches):
            if orig.shape == rec.shape:
                psnr = self.metrics_calculator.compute_psnr(orig, rec)
                psnr_values.append(psnr)

        success_rate = self.metrics_calculator.compute_attack_success_rate(psnr_values)

        # Compile results
        results = {
            'peft_method': peft_method,
            'num_samples_evaluated': sample_count,
            'num_patches_recovered': len(all_recovered_patches),
            'evaluation_metrics': evaluation_metrics,
            'attack_success_rate': success_rate,
            'psnr_values': psnr_values,
            'individual_attack_metrics': all_attack_metrics,
            'timestamp': datetime.now().isoformat()
        }

        # Save individual results
        self._save_attack_results(peft_method, results, all_original_patches, all_recovered_patches)

        self.logger.info(f"{peft_method} attack evaluation completed:")
        self.logger.info(f"  Success rate: {success_rate:.3f}")
        self.logger.info(f"  Mean PSNR: {evaluation_metrics.get('psnr_mean', 0):.2f} dB")
        self.logger.info(f"  Mean SSIM: {evaluation_metrics.get('ssim_mean', 0):.3f}")

        return results

    def evaluate_all_attacks(self, models_dict: Dict[str, nn.Module],
                             data_loader, num_samples: int = 64,
                             attack_configs: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Evaluate attacks across all PEFT methods

        Args:
            models_dict: Dictionary mapping PEFT method to model
            data_loader: Data loader for evaluation
            num_samples: Number of samples per method
            attack_configs: Configurations for each attack type

        Returns:
            Complete evaluation results
        """
        self.logger.info("Starting comprehensive attack evaluation...")

        attack_configs = attack_configs or {}
        all_results = {}

        for peft_method, model in models_dict.items():
            if peft_method in self.attack_classes:
                config = attack_configs.get(peft_method, {})

                try:
                    results = self.evaluate_single_attack(
                        model, peft_method, data_loader, num_samples, config
                    )
                    all_results[peft_method] = results

                except Exception as e:
                    self.logger.error(f"Failed to evaluate {peft_method} attack: {e}")
                    all_results[peft_method] = {'error': str(e)}

        # Create comparative analysis
        comparative_results = self._create_comparative_analysis(all_results)

        # Save comprehensive results
        comprehensive_results = {
            'individual_results': all_results,
            'comparative_analysis': comparative_results,
            'evaluation_config': {
                'num_samples': num_samples,
                'device': self.device,
                'timestamp': datetime.now().isoformat()
            }
        }

        self._save_comprehensive_results(comprehensive_results)

        # Generate visualizations
        self._create_evaluation_visualizations(all_results)

        self.evaluation_results = comprehensive_results
        return comprehensive_results

    def _create_comparative_analysis(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Create comparative analysis across PEFT methods"""

        comparative_analysis = {
            'success_rates': {},
            'mean_psnr': {},
            'mean_ssim': {},
            'vulnerability_ranking': [],
            'best_performing_attack': '',
            'summary_statistics': {}
        }

        valid_results = {k: v for k, v in results.items() if 'error' not in v}

        if not valid_results:
            return comparative_analysis

        # Extract key metrics
        for method, result in valid_results.items():
            comparative_analysis['success_rates'][method] = result.get('attack_success_rate', 0.0)
            comparative_analysis['mean_psnr'][method] = result.get('evaluation_metrics', {}).get('psnr_mean', 0.0)
            comparative_analysis['mean_ssim'][method] = result.get('evaluation_metrics', {}).get('ssim_mean', 0.0)

        # Rank methods by vulnerability (higher success rate = more vulnerable)
        vulnerability_ranking = sorted(
            comparative_analysis['success_rates'].items(),
            key=lambda x: x[1],
            reverse=True
        )
        comparative_analysis['vulnerability_ranking'] = vulnerability_ranking

        # Identify best performing attack
        if vulnerability_ranking:
            comparative_analysis['best_performing_attack'] = vulnerability_ranking[0][0]

        # Summary statistics
        success_rates = list(comparative_analysis['success_rates'].values())
        psnr_values = list(comparative_analysis['mean_psnr'].values())

        comparative_analysis['summary_statistics'] = {
            'avg_success_rate': np.mean(success_rates) if success_rates else 0.0,
            'std_success_rate': np.std(success_rates) if success_rates else 0.0,
            'avg_psnr': np.mean(psnr_values) if psnr_values else 0.0,
            'std_psnr': np.std(psnr_values) if psnr_values else 0.0,
            'methods_evaluated': len(valid_results)
        }

        return comparative_analysis

    def _save_attack_results(self, peft_method: str, results: Dict[str, Any],
                             original_patches: List[torch.Tensor],
                             recovered_patches: List[torch.Tensor]):
        """Save individual attack results"""

        method_dir = os.path.join(self.save_dir, peft_method)
        os.makedirs(method_dir, exist_ok=True)

        # Save metrics as JSON
        metrics_file = os.path.join(method_dir, 'metrics.json')
        with open(metrics_file, 'w') as f:
            # Convert non-serializable items
            serializable_results = {
                k: v for k, v in results.items()
                if k not in ['individual_attack_metrics']  # Skip complex nested data
            }
            json.dump(serializable_results, f, indent=2, default=str)

        # Save patches for visualization
        if original_patches and recovered_patches:
            patches_file = os.path.join(method_dir, 'sample_patches.pt')
            torch.save({
                'original': original_patches[:16],  # Save first 16 patches
                'recovered': recovered_patches[:16]
            }, patches_file)

        # Create visualization
        viz_file = os.path.join(method_dir, 'patch_comparison.png')
        if original_patches and recovered_patches:
            self.visualizer.plot_patch_comparison(
                original_patches[:8], recovered_patches[:8],
                title=f"{peft_method.upper()} Attack Results",
                save_path=viz_file
            )

    def _save_comprehensive_results(self, results: Dict[str, Any]):
        """Save comprehensive evaluation results"""

        results_file = os.path.join(self.save_dir, 'comprehensive_results.json')
        with open(results_file, 'w') as f:
            # Create serializable version
            serializable_results = {
                'comparative_analysis': results['comparative_analysis'],
                'evaluation_config': results['evaluation_config'],
                'individual_summaries': {}
            }

            # Add simplified individual results
            for method, result in results['individual_results'].items():
                if 'error' not in result:
                    serializable_results['individual_summaries'][method] = {
                        'success_rate': result.get('attack_success_rate', 0.0),
                        'mean_psnr': result.get('evaluation_metrics', {}).get('psnr_mean', 0.0),
                        'mean_ssim': result.get('evaluation_metrics', {}).get('ssim_mean', 0.0),
                        'num_patches_recovered': result.get('num_patches_recovered', 0)
                    }
                else:
                    serializable_results['individual_summaries'][method] = result

            json.dump(serializable_results, f, indent=2, default=str)

        self.logger.info(f"Comprehensive results saved to {results_file}")

    def _create_evaluation_visualizations(self, results: Dict[str, Any]):
        """Create evaluation visualizations"""

        # Success rate comparison
        methods = []
        success_rates = []
        mean_psnr = []

        for method, result in results.items():
            if 'error' not in result:
                methods.append(method.upper())
                success_rates.append(result.get('attack_success_rate', 0.0))
                mean_psnr.append(result.get('evaluation_metrics', {}).get('psnr_mean', 0.0))

        if methods:
            # Success rate bar plot
            plt.figure(figsize=(10, 6))

            plt.subplot(1, 2, 1)
            plt.bar(methods, success_rates, alpha=0.7, color='red')
            plt.title('Attack Success Rates by PEFT Method')
            plt.xlabel('PEFT Method')
            plt.ylabel('Success Rate')
            plt.ylim(0, 1)
            plt.xticks(rotation=45)

            # PSNR comparison
            plt.subplot(1, 2, 2)
            plt.bar(methods, mean_psnr, alpha=0.7, color='blue')
            plt.title('Mean PSNR by PEFT Method')
            plt.xlabel('PEFT Method')
            plt.ylabel('PSNR (dB)')
            plt.xticks(rotation=45)

            plt.tight_layout()

            viz_file = os.path.join(self.save_dir, 'attack_comparison.png')
            plt.savefig(viz_file, dpi=300, bbox_inches='tight')
            plt.close()

    def generate_attack_report(self) -> str:
        """Generate text report of attack evaluation"""

        if not self.evaluation_results:
            return "No evaluation results available. Run evaluate_all_attacks() first."

        report = []
        report.append("GRADIENT INVERSION ATTACK EVALUATION REPORT")
        report.append("=" * 50)
        report.append("")

        # Summary
        comparative = self.evaluation_results.get('comparative_analysis', {})
        summary_stats = comparative.get('summary_statistics', {})

        report.append("SUMMARY:")
        report.append(f"  Methods evaluated: {summary_stats.get('methods_evaluated', 0)}")
        report.append(
            f"  Average success rate: {summary_stats.get('avg_success_rate', 0):.3f} ± {summary_stats.get('std_success_rate', 0):.3f}")
        report.append(
            f"  Average PSNR: {summary_stats.get('avg_psnr', 0):.2f} ± {summary_stats.get('std_psnr', 0):.2f} dB")
        report.append("")

        # Vulnerability ranking
        ranking = comparative.get('vulnerability_ranking', [])
        if ranking:
            report.append("VULNERABILITY RANKING (Most to Least Vulnerable):")
            for i, (method, success_rate) in enumerate(ranking, 1):
                report.append(f"  {i}. {method.upper()}: {success_rate:.3f} success rate")
            report.append("")

        # Individual method results
        individual = self.evaluation_results.get('individual_results', {})
        report.append("INDIVIDUAL METHOD RESULTS:")

        for method, result in individual.items():
            if 'error' not in result:
                report.append(f"\n{method.upper()}:")
                report.append(f"  Success rate: {result.get('attack_success_rate', 0):.3f}")
                metrics = result.get('evaluation_metrics', {})
                report.append(f"  Mean PSNR: {metrics.get('psnr_mean', 0):.2f} dB")
                report.append(f"  Mean SSIM: {metrics.get('ssim_mean', 0):.3f}")
                report.append(f"  Patches recovered: {result.get('num_patches_recovered', 0)}")
            else:
                report.append(f"\n{method.upper()}: FAILED - {result.get('error', 'Unknown error')}")

        report_text = "\n".join(report)

        # Save report
        report_file = os.path.join(self.save_dir, 'attack_evaluation_report.txt')
        with open(report_file, 'w') as f:
            f.write(report_text)

        return report_text


# Example usage function
def run_attack_evaluation_example():
    """Example of how to use AttackEvaluator"""

    # This would be called with actual models and data
    evaluator = AttackEvaluator(device='cpu', save_dir='./attack_evaluation_results')

    print("AttackEvaluator initialized successfully!")
    print("Key methods:")
    print("  - evaluate_single_attack(): Evaluate one PEFT method")
    print("  - evaluate_all_attacks(): Evaluate all PEFT methods")
    print("  - generate_attack_report(): Generate comprehensive report")

    return evaluator


if __name__ == "__main__":
    evaluator = run_attack_evaluation_example()
