import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging #defense_evalutor.py
import json
import os
from datetime import datetime
import matplotlib.pyplot as plt


class DefenseEvaluator:
    """Comprehensive evaluation of defense mechanisms against gradient inversion attacks"""

    def __init__(self, device: str = 'cpu', save_dir: str = './defense_results'):
        """
        Initialize defense evaluator

        Args:
            device: Computing device
            save_dir: Directory to save evaluation results
        """
        self.device = device
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        # Import required classes
        from defenses.defense_factory import DefenseFactory
        from utils.metrics import MetricsCalculator
        from utils.visualization import ResultVisualizer

        self.defense_factory = DefenseFactory()
        self.metrics_calculator = MetricsCalculator(device=device)
        self.visualizer = ResultVisualizer()

        # Results storage
        self.evaluation_results = {}

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def evaluate_defense_effectiveness(self, attack_evaluator, models_dict: Dict[str, nn.Module],
                                       defense_configs: Dict[str, Dict[str, Any]],
                                       data_loader, num_samples: int = 32) -> Dict[str, Any]:
        """
        Evaluate effectiveness of defenses against attacks

        Args:
            attack_evaluator: AttackEvaluator instance
            models_dict: Dictionary mapping PEFT method to model
            defense_configs: Configuration for each defense method
            data_loader: Data loader for evaluation
            num_samples: Number of samples to test

        Returns:
            Defense effectiveness results
        """
        self.logger.info("Starting comprehensive defense evaluation...")

        results = {}

        for defense_name, defense_config in defense_configs.items():
            self.logger.info(f"Evaluating {defense_name} defense...")

            try:
                # Create defense mechanism
                defense = self.defense_factory.create_defense(
                    defense_config.get('type', defense_name),
                    device=self.device,
                    config=defense_config.get('params', {})
                )

                defense_results = {}

                # Test defense against each PEFT method
                for peft_method, model in models_dict.items():
                    self.logger.info(f"  Testing {defense_name} against {peft_method} attack...")

                    # Apply defense and run attack
                    protected_results = self._run_protected_attack(
                        attack_evaluator, model, peft_method, defense, data_loader, num_samples
                    )

                    defense_results[peft_method] = protected_results

                results[defense_name] = defense_results

            except Exception as e:
                self.logger.error(f"Failed to evaluate {defense_name}: {e}")
                results[defense_name] = {'error': str(e)}

        # Create comprehensive analysis
        comprehensive_results = self._analyze_defense_effectiveness(results)

        # Save results
        self._save_defense_results(comprehensive_results)

        # Generate visualizations
        self._create_defense_visualizations(comprehensive_results)

        self.evaluation_results = comprehensive_results
        return comprehensive_results

    def _run_protected_attack(self, attack_evaluator, model: nn.Module, peft_method: str,
                              defense, data_loader, num_samples: int) -> Dict[str, Any]:
        """Run attack against defended model"""

        # Create a copy of the model for defense testing
        defended_model = model

        all_original_patches = []
        all_recovered_patches = []
        defense_stats = []
        sample_count = 0

        defended_model.eval()

        for batch_idx, (images, labels) in enumerate(data_loader):
            if sample_count >= num_samples:
                break

            images = images.to(self.device)
            labels = labels.to(self.device)

            # Extract original patches
            from utils.data_processing import DataProcessor
            data_processor = DataProcessor(device=self.device)
            original_patches = data_processor.extract_patches(images.cpu())

            try:
                # Run forward pass to get gradients
                defended_model.train()
                defended_model.zero_grad()

                outputs, _ = defended_model(images)
                loss = nn.CrossEntropyLoss()(outputs, labels)
                loss.backward()

                # Extract gradients
                gradients = {}
                for name, param in defended_model.named_parameters():
                    if param.grad is not None:
                        gradients[name] = param.grad.clone().detach()

                # Apply defense to gradients
                defended_gradients = defense.apply_defense_with_validation(
                    gradients, batch_data=images, model=defended_model
                )

                # Get defense statistics
                defense_summary = defense.get_defense_summary()
                defense_stats.append(defense_summary)

                # Run attack on defended gradients
                attack_class = attack_evaluator.attack_classes.get(peft_method)
                if attack_class:
                    attacker = attack_class(defended_model, self.device)

                    # Manually set defended gradients for attack
                    attacker.attack_results['gradients'] = defended_gradients
                    recovered_patches = attacker.reconstruct_patches(defended_gradients)

                    all_original_patches.extend(original_patches)
                    all_recovered_patches.extend(recovered_patches)

                sample_count += images.size(0)

            except Exception as e:
                self.logger.warning(f"Defense application failed on batch {batch_idx}: {e}")
                continue

        # Evaluate reconstruction quality
        evaluation_metrics = self.metrics_calculator.evaluate_reconstruction_quality(
            all_original_patches, all_recovered_patches
        )

        # Compute attack success rate with defense
        psnr_values = []
        for orig, rec in zip(all_original_patches, all_recovered_patches):
            if orig.shape == rec.shape:
                psnr = self.metrics_calculator.compute_psnr(orig, rec)
                psnr_values.append(psnr)

        success_rate_with_defense = self.metrics_calculator.compute_attack_success_rate(psnr_values)

        # Aggregate defense statistics
        avg_defense_stats = {}
        if defense_stats:
            numeric_keys = ['avg_utility_loss', 'privacy_cost', 'gradient_reduction_ratio']
            for key in numeric_keys:
                values = [stats.get(key, 0) for stats in defense_stats if isinstance(stats.get(key), (int, float))]
                if values:
                    avg_defense_stats[key] = np.mean(values)

        return {
            'evaluation_metrics': evaluation_metrics,
            'attack_success_rate_with_defense': success_rate_with_defense,
            'psnr_values': psnr_values,
            'defense_statistics': avg_defense_stats,
            'num_samples_tested': sample_count,
            'num_patches_recovered': len(all_recovered_patches)
        }

    def _analyze_defense_effectiveness(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze overall defense effectiveness"""

        analysis = {
            'defense_rankings': {},
            'peft_vulnerability_with_defenses': {},
            'defense_effectiveness_matrix': {},
            'best_defenses_by_method': {},
            'overall_defense_performance': {},
            'privacy_utility_tradeoff': {}
        }

        # Filter valid results
        valid_results = {k: v for k, v in results.items() if 'error' not in v}

        if not valid_results:
            return analysis

        # Create effectiveness matrix (defense × PEFT method)
        defense_names = list(valid_results.keys())
        peft_methods = set()

        for defense_results in valid_results.values():
            peft_methods.update(defense_results.keys())

        peft_methods = sorted(peft_methods)

        effectiveness_matrix = np.zeros((len(defense_names), len(peft_methods)))

        for i, defense_name in enumerate(defense_names):
            for j, peft_method in enumerate(peft_methods):
                if peft_method in valid_results[defense_name]:
                    # Use attack success rate as effectiveness measure
                    # Lower success rate = more effective defense
                    success_rate = valid_results[defense_name][peft_method].get('attack_success_rate_with_defense', 1.0)
                    effectiveness_matrix[i, j] = 1.0 - success_rate  # Higher = more effective

        analysis['defense_effectiveness_matrix'] = {
            'matrix': effectiveness_matrix.tolist(),
            'defense_names': defense_names,
            'peft_methods': peft_methods
        }

        # Rank defenses by average effectiveness
        defense_avg_effectiveness = {}
        for i, defense_name in enumerate(defense_names):
            avg_eff = np.mean(effectiveness_matrix[i, :])
            defense_avg_effectiveness[defense_name] = avg_eff

        analysis['defense_rankings'] = sorted(
            defense_avg_effectiveness.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # Find best defense for each PEFT method
        for j, peft_method in enumerate(peft_methods):
            method_effectiveness = effectiveness_matrix[:, j]
            best_defense_idx = np.argmax(method_effectiveness)
            analysis['best_defenses_by_method'][peft_method] = {
                'defense': defense_names[best_defense_idx],
                'effectiveness': method_effectiveness[best_defense_idx]
            }

        # Overall performance statistics
        analysis['overall_defense_performance'] = {
            'avg_effectiveness': np.mean(effectiveness_matrix),
            'std_effectiveness': np.std(effectiveness_matrix),
            'best_overall_defense': analysis['defense_rankings'][0][0] if analysis['defense_rankings'] else None,
            'most_vulnerable_peft': peft_methods[
                np.argmin(np.mean(effectiveness_matrix, axis=0))] if peft_methods else None
        }

        return analysis

    def _save_defense_results(self, results: Dict[str, Any]):
        """Save defense evaluation results"""

        results_file = os.path.join(self.save_dir, 'defense_evaluation_results.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        self.logger.info(f"Defense results saved to {results_file}")

    def _create_defense_visualizations(self, results: Dict[str, Any]):
        """Create defense effectiveness visualizations"""

        effectiveness_data = results.get('defense_effectiveness_matrix', {})
        matrix = np.array(effectiveness_data.get('matrix', []))
        defense_names = effectiveness_data.get('defense_names', [])
        peft_methods = effectiveness_data.get('peft_methods', [])

        if matrix.size > 0:
            # Create effectiveness heatmap
            plt.figure(figsize=(12, 8))

            from matplotlib.colors import LinearSegmentedColormap
            colors = ['red', 'yellow', 'green']
            cmap = LinearSegmentedColormap.from_list('effectiveness', colors, N=100)

            im = plt.imshow(matrix, cmap=cmap, aspect='auto', vmin=0, vmax=1)

            # Add labels
            plt.xticks(range(len(peft_methods)), [m.upper() for m in peft_methods], rotation=45)
            plt.yticks(range(len(defense_names)), defense_names)

            # Add text annotations
            for i in range(len(defense_names)):
                for j in range(len(peft_methods)):
                    plt.text(j, i, f'{matrix[i, j]:.2f}', ha='center', va='center',
                             color='black' if matrix[i, j] < 0.5 else 'white')

            plt.colorbar(im, label='Defense Effectiveness (0=Ineffective, 1=Perfect)')
            plt.title('Defense Effectiveness Against Different PEFT Methods')
            plt.xlabel('PEFT Method')
            plt.ylabel('Defense Mechanism')
            plt.tight_layout()

            viz_file = os.path.join(self.save_dir, 'defense_effectiveness_heatmap.png')
            plt.savefig(viz_file, dpi=300, bbox_inches='tight')
            plt.close()

        # Defense ranking bar plot
        rankings = results.get('defense_rankings', [])
        if rankings:
            plt.figure(figsize=(10, 6))

            defense_names = [r[0] for r in rankings]
            effectiveness_scores = [r[1] for r in rankings]

            plt.bar(defense_names, effectiveness_scores, alpha=0.7, color='green')
            plt.title('Overall Defense Effectiveness Rankings')
            plt.xlabel('Defense Mechanism')
            plt.ylabel('Average Effectiveness Score')
            plt.xticks(rotation=45)
            plt.ylim(0, 1)

            plt.tight_layout()

            ranking_viz_file = os.path.join(self.save_dir, 'defense_rankings.png')
            plt.savefig(ranking_viz_file, dpi=300, bbox_inches='tight')
            plt.close()

    def generate_defense_report(self) -> str:
        """Generate comprehensive defense evaluation report"""

        if not self.evaluation_results:
            return "No evaluation results available. Run evaluate_defense_effectiveness() first."

        report = []
        report.append("DEFENSE MECHANISM EVALUATION REPORT")
        report.append("=" * 50)
        report.append("")

        # Overall performance
        overall_perf = self.evaluation_results.get('overall_defense_performance', {})
        report.append("OVERALL DEFENSE PERFORMANCE:")
        report.append(f"  Average effectiveness: {overall_perf.get('avg_effectiveness', 0):.3f}")
        report.append(f"  Standard deviation: {overall_perf.get('std_effectiveness', 0):.3f}")
        report.append(f"  Best overall defense: {overall_perf.get('best_overall_defense', 'N/A')}")
        report.append(f"  Most vulnerable PEFT: {overall_perf.get('most_vulnerable_peft', 'N/A')}")
        report.append("")

        # Defense rankings
        rankings = self.evaluation_results.get('defense_rankings', [])
        if rankings:
            report.append("DEFENSE EFFECTIVENESS RANKINGS:")
            for i, (defense, score) in enumerate(rankings, 1):
                report.append(f"  {i}. {defense}: {score:.3f} effectiveness score")
            report.append("")

        # Best defenses by PEFT method
        best_defenses = self.evaluation_results.get('best_defenses_by_method', {})
        if best_defenses:
            report.append("BEST DEFENSE FOR EACH PEFT METHOD:")
            for peft_method, defense_info in best_defenses.items():
                defense_name = defense_info.get('defense', 'N/A')
                effectiveness = defense_info.get('effectiveness', 0)
                report.append(f"  {peft_method.upper()}: {defense_name} (effectiveness: {effectiveness:.3f})")
            report.append("")

        report_text = "\n".join(report)

        # Save report
        report_file = os.path.join(self.save_dir, 'defense_evaluation_report.txt')
        with open(report_file, 'w') as f:
            f.write(report_text)

        return report_text


# Example usage function
def run_defense_evaluation_example():
    """Example of how to use DefenseEvaluator"""

    evaluator = DefenseEvaluator(device='cpu', save_dir='./defense_evaluation_results')

    print("DefenseEvaluator initialized successfully!")
    print("Key methods:")
    print("  - evaluate_defense_effectiveness(): Evaluate defenses against attacks")
    print("  - generate_defense_report(): Generate comprehensive report")

    return evaluator


if __name__ == "__main__":
    evaluator = run_defense_evaluation_example()
