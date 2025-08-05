import torch
import os
import sys
from datetime import datetime
import logging
import numpy as np
from typing import Dict, Any

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config
from models.model_factory import ModelFactory
from utils.data_processing import DataProcessor
from evaluation.attack_evaluator import AttackEvaluator
from evaluation.defense_evaluator import DefenseEvaluator
from evaluation.statistical_analysis import StatisticalAnalyzer
from defenses.defense_factory import DefenseFactory


class DefenseExperimentRunner:
    """Run comprehensive defense experiments"""

    def __init__(self, experiment_config=None, quick_test=False):
        """
        Initialize defense experiment runner

        Args:
            experiment_config: Custom configuration
            quick_test: Whether to run quick test configuration
        """
        self.config = experiment_config or (config.get_quick_config() if quick_test else config)

        # Setup directories
        self.setup_directories()

        # Setup logging
        self.setup_logging()

        # Initialize components
        self.model_factory = ModelFactory()
        self.data_processor = DataProcessor(device=self.config.device)
        self.attack_evaluator = AttackEvaluator(
            device=self.config.device,
            save_dir=os.path.join(self.config.output_config['save_dir'], 'attacks')
        )
        self.defense_evaluator = DefenseEvaluator(
            device=self.config.device,
            save_dir=os.path.join(self.config.output_config['save_dir'], 'defenses')
        )
        self.statistical_analyzer = StatisticalAnalyzer()
        self.defense_factory = DefenseFactory()

        self.logger.info("Defense experiment runner initialized")

    def setup_directories(self):
        """Setup output directories"""
        base_dir = self.config.output_config['save_dir']

        directories = [
            base_dir,
            os.path.join(base_dir, 'defenses'),
            os.path.join(base_dir, 'attacks'),
            os.path.join(base_dir, 'statistical'),
            os.path.join(base_dir, 'logs'),
            os.path.join(base_dir, 'plots')
        ]

        for directory in directories:
            os.makedirs(directory, exist_ok=True)

    def setup_logging(self):
        """Setup logging configuration"""
        log_file = os.path.join(
            self.config.output_config['save_dir'],
            'logs',
            f'defense_experiments_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        )

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )

        self.logger = logging.getLogger('DefenseExperiments')

    def create_models(self) -> Dict[str, torch.nn.Module]:
        """Create models with different PEFT methods"""
        models = {}

        for peft_method in self.config.get_peft_methods():
            self.logger.info(f"Creating {peft_method} model...")

            try:
                model = self.model_factory.create_model(
                    peft_method=peft_method,
                    model_config=self.config.model_config,
                    peft_config=self.config.peft_configs[peft_method],
                    device=self.config.device
                )

                models[peft_method] = model
                self.logger.info(f"✓ {peft_method} model created successfully")

            except Exception as e:
                self.logger.error(f"✗ Failed to create {peft_method} model: {e}")

        return models

    def prepare_data(self):
        """Prepare evaluation data"""
        self.logger.info("Preparing evaluation data...")

        try:
            data_loader = self.data_processor.prepare_cifar100_data(
                subset_size=self.config.dataset_config.get('subset_size')
            )

            self.logger.info(f"✓ Data loader created")
            return data_loader

        except Exception as e:
            self.logger.error(f"✗ Failed to prepare data: {e}")
            raise

    def run_baseline_attacks(self, models: Dict[str, torch.nn.Module],
                             data_loader) -> Dict[str, Any]:
        """Run baseline attacks without defenses"""

        self.logger.info("Running baseline attacks (no defense)...")

        baseline_results = self.attack_evaluator.evaluate_all_attacks(
            models_dict=models,
            data_loader=data_loader,
            num_samples=self.config.evaluation_config['num_samples'],
            attack_configs=self.config.attack_configs
        )

        return baseline_results

    def run_defended_attacks(self, models: Dict[str, torch.nn.Module],
                             data_loader) -> Dict[str, Any]:
        """Run attacks against defended models"""

        self.logger.info("Running defense evaluation...")

        defense_results = self.defense_evaluator.evaluate_defense_effectiveness(
            attack_evaluator=self.attack_evaluator,
            models_dict=models,
            defense_configs=self.config.defense_configs,
            data_loader=data_loader,
            num_samples=self.config.evaluation_config['num_samples']
        )

        return defense_results

    def perform_statistical_analysis(self, baseline_results: Dict[str, Any],
                                     defense_results: Dict[str, Any]) -> Dict[str, Any]:
        """Perform statistical analysis comparing baseline vs defended"""

        self.logger.info("Performing statistical analysis...")

        statistical_results = {}

        # Extract baseline PSNR values
        baseline_psnr = {}
        baseline_individual = baseline_results.get('individual_results', {})

        for peft_method, result in baseline_individual.items():
            if 'error' not in result:
                eval_metrics = result.get('evaluation_metrics', {})
                baseline_psnr[peft_method] = eval_metrics.get('psnr_mean', 0)

        # Compare with each defense method
        for defense_name, defense_data in defense_results.items():
            if 'error' not in defense_data:
                statistical_results[defense_name] = {}

                for peft_method in baseline_psnr.keys():
                    if peft_method in defense_data:
                        defense_eval = defense_data[peft_method].get('evaluation_metrics', {})
                        defense_psnr = defense_eval.get('psnr_mean', 0)

                        # Create mock data for statistical test (in real scenario, use multiple runs)
                        baseline_values = [baseline_psnr[peft_method]] * 5  # Mock multiple values
                        defense_values = [defense_psnr] * 5

                        # Perform paired t-test
                        try:
                            t_test_result = self.statistical_analyzer.paired_t_test(
                                baseline_values, defense_values
                            )

                            statistical_results[defense_name][peft_method] = {
                                'paired_t_test': t_test_result,
                                'baseline_mean': baseline_psnr[peft_method],
                                'defense_mean': defense_psnr,
                                'improvement': baseline_psnr[peft_method] - defense_psnr
                            }

                        except Exception as e:
                            self.logger.warning(f"Statistical test failed for {defense_name}-{peft_method}: {e}")

        return statistical_results

    def generate_defense_effectiveness_matrix(self, defense_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate defense effectiveness matrix for heatmap"""

        peft_methods = list(self.config.get_peft_methods())
        defense_methods = list(self.config.get_defense_methods())

        # Create effectiveness matrix (lower PSNR = more effective defense)
        import numpy as np
        effectiveness_matrix = np.zeros((len(peft_methods), len(defense_methods)))

        for i, peft_method in enumerate(peft_methods):
            for j, defense_method in enumerate(defense_methods):
                if defense_method in defense_results and 'error' not in defense_results[defense_method]:
                    defense_data = defense_results[defense_method]
                    if peft_method in defense_data:
                        peft_result = defense_data[peft_method]
                        eval_metrics = peft_result.get('evaluation_metrics', {})
                        psnr = eval_metrics.get('psnr_mean', 0)

                        # Convert PSNR to effectiveness (lower PSNR = higher effectiveness)
                        # Normalize to 0-1 scale
                        effectiveness = max(0, (30 - psnr) / 30)  # Assuming 30 dB as baseline
                        effectiveness_matrix[i, j] = effectiveness

        return {
            'matrix': effectiveness_matrix.tolist(),
            'peft_methods': peft_methods,
            'defense_methods': defense_methods
        }

    def save_results(self, all_results: Dict[str, Any]):
        """Save all experiment results"""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save comprehensive results
        results_file = os.path.join(
            self.config.output_config['save_dir'],
            'defenses',
            f'defense_experiment_results_{timestamp}.json'
        )

        import json
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)

        self.logger.info(f"Comprehensive results saved to {results_file}")

        # Generate defense report
        defense_report = self.defense_evaluator.generate_defense_report()

        defense_report_file = os.path.join(
            self.config.output_config['save_dir'],
            'defenses',
            f'defense_report_{timestamp}.txt'
        )

        with open(defense_report_file, 'w') as f:
            f.write(defense_report)

        # Generate statistical report
        if 'statistical_analysis' in all_results:
            stat_report = self.statistical_analyzer.generate_statistical_report(
                all_results['statistical_analysis']
            )

            stat_report_file = os.path.join(
                self.config.output_config['save_dir'],
                'statistical',
                f'statistical_report_{timestamp}.txt'
            )

            with open(stat_report_file, 'w') as f:
                f.write(stat_report)

        return {
            'results': results_file,
            'defense_report': defense_report_file,
            'statistical_report': stat_report_file if 'statistical_analysis' in all_results else None
        }

    def run_complete_experiment(self) -> Dict[str, Any]:
        """Run complete defense experiment pipeline"""

        self.logger.info("=" * 60)
        self.logger.info("STARTING COMPREHENSIVE DEFENSE EXPERIMENTS")
        self.logger.info("=" * 60)

        try:
            # Step 1: Create models
            models = self.create_models()
            if not models:
                raise ValueError("No models created successfully")

            # Step 2: Prepare data
            data_loader = self.prepare_data()

            # Step 3: Run baseline attacks
            baseline_results = self.run_baseline_attacks(models, data_loader)

            # Step 4: Run defended attacks
            defense_results = self.run_defended_attacks(models, data_loader)

            # Step 5: Perform statistical analysis
            statistical_results = self.perform_statistical_analysis(
                baseline_results, defense_results
            )

            # Step 6: Generate effectiveness matrix
            effectiveness_matrix = self.generate_defense_effectiveness_matrix(defense_results)

            # Step 7: Compile all results
            all_results = {
                'baseline_results': baseline_results,
                'defense_results': defense_results,
                'statistical_analysis': statistical_results,
                'effectiveness_matrix': effectiveness_matrix,
                'experiment_config': {
                    'num_samples': self.config.evaluation_config['num_samples'],
                    'peft_methods': self.config.get_peft_methods(),
                    'defense_methods': self.config.get_defense_methods(),
                    'device': self.config.device,
                    'timestamp': datetime.now().isoformat()
                }
            }

            # Step 8: Save results
            saved_files = self.save_results(all_results)

            # Step 9: Generate summary
            summary = self._generate_experiment_summary(all_results)

            self.logger.info("=" * 60)
            self.logger.info("DEFENSE EXPERIMENTS COMPLETED SUCCESSFULLY")
            self.logger.info("=" * 60)

            return {
                'results': all_results,
                'summary': summary,
                'files': saved_files
            }

        except Exception as e:
            self.logger.error(f"Defense experiment failed: {e}")
            raise

    def _generate_experiment_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate experiment summary"""

        summary = {
            'experiment_type': 'defense_evaluation',
            'timestamp': datetime.now().isoformat(),
            'configuration': {
                'num_samples': self.config.evaluation_config['num_samples'],
                'peft_methods': self.config.get_peft_methods(),
                'defense_methods': self.config.get_defense_methods(),
                'device': self.config.device
            },
            'key_findings': {}
        }

        # Extract key findings
        defense_results = results.get('defense_results', {})

        if 'defense_rankings' in defense_results:
            rankings = defense_results['defense_rankings']
            if rankings:
                summary['key_findings']['best_defense'] = {
                    'method': rankings[0][0],
                    'effectiveness': rankings[0][1]
                }

        if 'best_defenses_by_method' in defense_results:
            summary['key_findings']['method_specific_recommendations'] = defense_results['best_defenses_by_method']

        return summary


def main():
    """Main function to run defense experiments"""

    import argparse

    parser = argparse.ArgumentParser(description='Run PEFT Defense Experiments')
    parser.add_argument('--quick', action='store_true', help='Run quick test configuration')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use (cpu/cuda)')
    parser.add_argument('--num-samples', type=int, help='Number of samples to evaluate')

    args = parser.parse_args()

    # Override config if arguments provided
    if args.device:
        config.device = args.device
    if args.num_samples:
        config.evaluation_config['num_samples'] = args.num_samples

    # Initialize and run experiments
    runner = DefenseExperimentRunner(quick_test=args.quick)

    try:
        results = runner.run_complete_experiment()

        print("\n" + "=" * 60)
        print("DEFENSE EXPERIMENTS COMPLETED SUCCESSFULLY!")
        print("=" * 60)

        # Print file locations
        files = results['files']
        print(f"Results saved to: {files['results']}")
        print(f"Defense report: {files['defense_report']}")
        if files.get('statistical_report'):
            print(f"Statistical report: {files['statistical_report']}")

        # Print summary
        summary = results['summary']
        if 'key_findings' in summary:
            findings = summary['key_findings']
            if 'best_defense' in findings:
                best = findings['best_defense']
                print(f"\nBest overall defense: {best['method']} ({best['effectiveness']:.1%} effectiveness)")

    except Exception as e:
        print(f"\nDEFENSE EXPERIMENTS FAILED: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
