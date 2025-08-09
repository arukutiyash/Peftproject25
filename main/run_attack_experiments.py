import torch
import os
import sys
import numpy as np
from datetime import datetime
import logging
from typing import Dict, Any, List  # ✅ Added List import

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config
from models.model_factory import ModelFactory
from utils.data_processing import DataProcessor
from evaluation.attack_evaluator import AttackEvaluator
from utils.visualization import ResultVisualizer

class AttackExperimentRunner:
    """Run comprehensive attack experiments across all PEFT methods"""

    def __init__(self, experiment_config=None, quick_test=False):
        """
        Initialize attack experiment runner

        Args:
            experiment_config: Custom configuration, uses global config if None
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
        self.visualizer = ResultVisualizer()

        self.logger.info("Attack experiment runner initialized")

    def setup_directories(self):
        """Setup output directories"""
        base_dir = self.config.output_config['save_dir']

        directories = [
            base_dir,
            os.path.join(base_dir, 'attacks'),
            os.path.join(base_dir, 'models'),
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
            f'attack_experiments_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        )

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )

        self.logger = logging.getLogger('AttackExperiments')

    def create_models(self) -> Dict[str, torch.nn.Module]:
        """Create models with different PEFT methods"""
        models = {}

        for peft_method in self.config.get_peft_methods():
            self.logger.info(f"Creating {peft_method} model...")

            try:
                # ✅ FIXED: Correct ModelFactory parameters
                model = self.model_factory.create_model(
                    model_size="base",
                    peft_method=peft_method,
                    num_classes=100,
                    peft_config=self.config.peft_configs[peft_method],
                    malicious=True
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

            self.logger.info(f"✓ Data loader created with subset_size={self.config.dataset_config.get('subset_size')}")
            return data_loader

        except Exception as e:
            self.logger.error(f"✗ Failed to prepare data: {e}")
            raise

    def run_single_attack_experiment(self, models: Dict[str, torch.nn.Module],
                                     data_loader) -> Dict[str, Any]:
        """Run single attack experiment across all PEFT methods"""

        self.logger.info("Starting single attack experiment...")

        # Run attack evaluation
        attack_results = self.attack_evaluator.evaluate_all_attacks(
            models_dict=models,
            data_loader=data_loader,
            num_samples=self.config.evaluation_config['num_samples'],
            attack_configs=self.config.attack_configs
        )

        return attack_results

    def run_multiple_runs(self, models: Dict[str, torch.nn.Module],
                          data_loader) -> Dict[str, Any]:
        """Run multiple experimental runs for statistical significance"""

        num_runs = self.config.evaluation_config['num_runs']
        self.logger.info(f"Starting {num_runs} experimental runs...")

        all_runs_results = []

        for run_idx in range(num_runs):
            self.logger.info(f"=== Run {run_idx + 1}/{num_runs} ===")

            # Run single experiment
            run_results = self.run_single_attack_experiment(models, data_loader)

            # Add run metadata
            run_results['run_id'] = run_idx
            run_results['timestamp'] = datetime.now().isoformat()

            all_runs_results.append(run_results)

            self.logger.info(f"Run {run_idx + 1} completed")

        # Aggregate results across runs
        aggregated_results = self._aggregate_multiple_runs(all_runs_results)

        return {
            'individual_runs': all_runs_results,
            'aggregated_results': aggregated_results,
            'num_runs': num_runs
        }

    def _aggregate_multiple_runs(self, all_runs: List[Dict[str, Any]]) -> Dict[str, Any]:  # ✅ FIXED: List type hint
        """Aggregate results across multiple runs"""

        if not all_runs:
            return {}

        aggregated = {
            'peft_methods': {},
            'overall_statistics': {}
        }

        # Get PEFT methods from first run
        first_run = all_runs[0]
        individual_results = first_run.get('individual_results', {})

        for peft_method in individual_results.keys():
            if 'error' not in individual_results[peft_method]:
                # Collect metrics across all runs
                success_rates = []
                psnr_values = []
                ssim_values = []

                for run in all_runs:
                    method_result = run.get('individual_results', {}).get(peft_method, {})
                    if 'error' not in method_result:
                        success_rates.append(method_result.get('attack_success_rate', 0))
                        eval_metrics = method_result.get('evaluation_metrics', {})
                        psnr_values.append(eval_metrics.get('psnr_mean', 0))
                        ssim_values.append(eval_metrics.get('ssim_mean', 0))

                # Calculate statistics
                aggregated['peft_methods'][peft_method] = {
                    'success_rate_mean': np.mean(success_rates) if success_rates else 0,
                    'success_rate_std': np.std(success_rates) if success_rates else 0,
                    'psnr_mean': np.mean(psnr_values) if psnr_values else 0,
                    'psnr_std': np.std(psnr_values) if psnr_values else 0,
                    'ssim_mean': np.mean(ssim_values) if ssim_values else 0,
                    'ssim_std': np.std(ssim_values) if ssim_values else 0,
                    'num_runs': len(success_rates)
                }

        return aggregated

    def save_results(self, results: Dict[str, Any]):
        """Save experiment results"""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save detailed results
        results_file = os.path.join(
            self.config.output_config['save_dir'],
            'attacks',
            f'attack_experiment_results_{timestamp}.json'
        )

        import json
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        self.logger.info(f"Results saved to {results_file}")

        # Generate report
        report = self.attack_evaluator.generate_attack_report()

        report_file = os.path.join(
            self.config.output_config['save_dir'],
            'attacks',
            f'attack_experiment_report_{timestamp}.txt'
        )

        with open(report_file, 'w') as f:
            f.write(report)

        self.logger.info(f"Report saved to {report_file}")

        return results_file, report_file

    def run_complete_experiment(self) -> Dict[str, Any]:
        """Run complete attack experiment pipeline"""

        self.logger.info("=" * 60)
        self.logger.info("STARTING COMPREHENSIVE ATTACK EXPERIMENTS")
        self.logger.info("=" * 60)

        try:
            # Step 1: Create models
            models = self.create_models()
            if not models:
                raise ValueError("No models created successfully")

            # Step 2: Prepare data
            data_loader = self.prepare_data()

            # Step 3: Run experiments
            if self.config.evaluation_config['num_runs'] > 1:
                results = self.run_multiple_runs(models, data_loader)
            else:
                single_result = self.run_single_attack_experiment(models, data_loader)
                results = {
                    'individual_runs': [single_result],
                    'aggregated_results': {'single_run': single_result},
                    'num_runs': 1
                }

            # Step 4: Save results
            results_file, report_file = self.save_results(results)

            # Step 5: Generate summary
            summary = self._generate_experiment_summary(results)

            self.logger.info("=" * 60)
            self.logger.info("ATTACK EXPERIMENTS COMPLETED SUCCESSFULLY")
            self.logger.info("=" * 60)

            return {
                'results': results,
                'summary': summary,
                'files': {
                    'results': results_file,
                    'report': report_file
                }
            }

        except Exception as e:
            self.logger.error(f"Experiment failed: {e}")
            raise

    def _generate_experiment_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate experiment summary"""

        summary = {
            'experiment_type': 'attack_evaluation',
            'timestamp': datetime.now().isoformat(),
            'configuration': {
                'num_runs': results.get('num_runs', 1),
                'num_samples': self.config.evaluation_config['num_samples'],
                'peft_methods': list(self.config.get_peft_methods()),
                'device': self.config.device
            },
            'key_findings': {}
        }

        # Extract key findings from aggregated results
        if 'aggregated_results' in results:
            agg_results = results['aggregated_results']

            if 'peft_methods' in agg_results:
                peft_data = agg_results['peft_methods']

                # Find most/least vulnerable methods
                success_rates = {method: data.get('success_rate_mean', 0)
                                 for method, data in peft_data.items()}

                if success_rates:
                    most_vulnerable = max(success_rates, key=success_rates.get)
                    least_vulnerable = min(success_rates, key=success_rates.get)

                    summary['key_findings'] = {
                        'most_vulnerable_peft': {
                            'method': most_vulnerable,
                            'success_rate': success_rates[most_vulnerable]
                        },
                        'least_vulnerable_peft': {
                            'method': least_vulnerable,
                            'success_rate': success_rates[least_vulnerable]
                        },
                        'average_success_rate': np.mean(list(success_rates.values())),
                        'methods_evaluated': len(success_rates)
                    }

        return summary


def main():
    """Main function to run attack experiments"""

    import argparse

    parser = argparse.ArgumentParser(description='Run PEFT Attack Experiments')
    parser.add_argument('--quick', action='store_true', help='Run quick test configuration')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use (cpu/cuda)')
    parser.add_argument('--num-samples', type=int, help='Number of samples to evaluate')
    parser.add_argument('--num-runs', type=int, help='Number of experimental runs')

    args = parser.parse_args()

    # Override config if arguments provided
    if args.device:
        config.device = args.device
    if args.num_samples:
        config.evaluation_config['num_samples'] = args.num_samples
    if args.num_runs:
        config.evaluation_config['num_runs'] = args.num_runs

    # Initialize and run experiments
    runner = AttackExperimentRunner(quick_test=args.quick)

    try:
        results = runner.run_complete_experiment()

        print("\n" + "=" * 60)
        print("EXPERIMENT COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"Results saved to: {results['files']['results']}")
        print(f"Report saved to: {results['files']['report']}")

        # Print summary
        summary = results['summary']
        if 'key_findings' in summary:
            findings = summary['key_findings']
            print(f"\nKey Findings:")
            print(f"  Most vulnerable PEFT: {findings.get('most_vulnerable_peft', {}).get('method', 'N/A')}")
            print(f"  Least vulnerable PEFT: {findings.get('least_vulnerable_peft', {}).get('method', 'N/A')}")
            print(f"  Average success rate: {findings.get('average_success_rate', 0):.1%}")
            print(f"  Methods evaluated: {findings.get('methods_evaluated', 0)}")

    except Exception as e:
        print(f"\nEXPERIMENT FAILED: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
