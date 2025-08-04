import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np  # heatmap_generator.py
from typing import List, Dict, Optional, Any
import os
import logging


class HeatmapGenerator:
    """Generate comprehensive heatmaps for attack and defense evaluation results"""

    def __init__(self, save_dir: str = './heatmaps'):
        """
        Initialize heatmap generator

        Args:
            save_dir: Directory to save generated heatmaps
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        # Setup logging
        self.logger = logging.getLogger(__name__)

        # Set style
        plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
        sns.set_palette("husl")

    def generate_psnr_heatmap(self, psnr_matrix: np.ndarray,
                              peft_methods: List[str],
                              defense_methods: List[str],
                              title: str = "PSNR Heatmap: Defense Effectiveness",
                              save_path: Optional[str] = None) -> str:
        """
        Generate PSNR heatmap showing defense effectiveness

        Args:
            psnr_matrix: 2D numpy array (rows: PEFT methods, cols: defense methods)
            peft_methods: List of PEFT method names
            defense_methods: List of defense method names
            title: Heatmap title
            save_path: Optional custom save path

        Returns:
            Path to saved heatmap
        """
        plt.figure(figsize=(len(defense_methods) * 1.5, len(peft_methods) * 1.2))

        # Create heatmap
        heatmap = sns.heatmap(
            psnr_matrix,
            xticklabels=[method.title() for method in defense_methods],
            yticklabels=[method.upper() for method in peft_methods],
            annot=True,
            fmt='.2f',
            cmap='RdYlBu_r',  # Red = high PSNR = less effective defense
            center=20.0,  # Center around 20 dB PSNR
            cbar_kws={'label': 'PSNR (dB)'},
            linewidths=0.5
        )

        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('Defense Method', fontsize=14)
        plt.ylabel('PEFT Method', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)

        # Add text annotations for interpretation
        plt.figtext(0.02, 0.02, 'Higher PSNR = Less Effective Defense',
                    fontsize=10, style='italic', color='gray')

        plt.tight_layout()

        # Save heatmap
        if save_path is None:
            save_path = os.path.join(self.save_dir, 'psnr_heatmap.png')

        plt.savefig(save_path, dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.close()

        self.logger.info(f"PSNR heatmap saved to {save_path}")
        return save_path

    def generate_attack_success_heatmap(self, success_matrix: np.ndarray,
                                        peft_methods: List[str],
                                        attack_scenarios: List[str],
                                        title: str = "Attack Success Rate Heatmap",
                                        save_path: Optional[str] = None) -> str:
        """
        Generate attack success rate heatmap

        Args:
            success_matrix: 2D numpy array of success rates
            peft_methods: List of PEFT method names
            attack_scenarios: List of attack scenario names
            title: Heatmap title
            save_path: Optional custom save path

        Returns:
            Path to saved heatmap
        """
        plt.figure(figsize=(len(attack_scenarios) * 1.5, len(peft_methods) * 1.2))

        # Create heatmap
        heatmap = sns.heatmap(
            success_matrix,
            xticklabels=[scenario.title() for scenario in attack_scenarios],
            yticklabels=[method.upper() for method in peft_methods],
            annot=True,
            fmt='.3f',
            cmap='Reds',
            vmin=0.0,
            vmax=1.0,
            cbar_kws={'label': 'Attack Success Rate'},
            linewidths=0.5
        )

        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('Attack Scenario', fontsize=14)
        plt.ylabel('PEFT Method', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)

        plt.tight_layout()

        # Save heatmap
        if save_path is None:
            save_path = os.path.join(self.save_dir, 'attack_success_heatmap.png')

        plt.savefig(save_path, dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.close()

        self.logger.info(f"Attack success heatmap saved to {save_path}")
        return save_path

    def generate_comprehensive_heatmap(self, results_dict: Dict[str, Any],
                                       peft_methods: List[str],
                                       defense_methods: List[str],
                                       metric: str = 'psnr',
                                       save_path: Optional[str] = None) -> str:
        """
        Generate comprehensive heatmap from results dictionary

        Updated to handle combined defenses and complex nested structures

        Args:
            results_dict: Dictionary with evaluation results
            peft_methods: List of PEFT methods
            defense_methods: List of defense methods
            metric: Metric to visualize ('psnr', 'ssim', 'success_rate')
            save_path: Optional custom save path

        Returns:
            Path to saved heatmap
        """
        # Create results matrix
        results_matrix = np.zeros((len(peft_methods), len(defense_methods)))

        # Debug information
        missing_combinations = []
        found_combinations = []

        for i, peft_method in enumerate(peft_methods):
            for j, defense_method in enumerate(defense_methods):
                value_found = False
                extracted_value = 0.0

                # Try multiple key formats for different result structures
                possible_keys = [
                    f"{peft_method}_{defense_method}",
                    f"{defense_method}_{peft_method}",
                    defense_method,
                    peft_method
                ]

                # Strategy 1: Direct key lookup
                for key in possible_keys:
                    if key in results_dict:
                        result_data = results_dict[key]

                        # Handle different nesting structures
                        if isinstance(result_data, dict):
                            # Case 1: defense -> peft -> metrics
                            if peft_method in result_data:
                                peft_data = result_data[peft_method]
                                if isinstance(peft_data, dict):
                                    extracted_value = self._extract_metric_value(peft_data, metric)
                                    value_found = True
                                    break

                            # Case 2: Direct metrics in result_data
                            else:
                                extracted_value = self._extract_metric_value(result_data, metric)
                                if extracted_value > 0:  # Only accept if we got a meaningful value
                                    value_found = True
                                    break

                # Strategy 2: Look for defense method directly in results_dict
                if not value_found and defense_method in results_dict:
                    defense_data = results_dict[defense_method]
                    if isinstance(defense_data, dict) and peft_method in defense_data:
                        peft_data = defense_data[peft_method]
                        if isinstance(peft_data, dict):
                            extracted_value = self._extract_metric_value(peft_data, metric)
                            value_found = True

                # Strategy 3: Handle nested defense results structure
                if not value_found:
                    for key, value in results_dict.items():
                        if isinstance(value, dict):
                            # Check if this might be a defense-specific result
                            if defense_method in key.lower() or any(
                                    defense_method in subkey.lower() for subkey in value.keys()):
                                if peft_method in value:
                                    peft_data = value[peft_method]
                                    if isinstance(peft_data, dict):
                                        extracted_value = self._extract_metric_value(peft_data, metric)
                                        if extracted_value > 0:
                                            value_found = True
                                            break

                # Store the result
                if value_found:
                    results_matrix[i, j] = extracted_value
                    found_combinations.append(f"{peft_method}+{defense_method}")
                else:
                    missing_combinations.append(f"{peft_method}+{defense_method}")

        # Log debug information
        if missing_combinations:
            self.logger.warning(f"Missing data for combinations: {missing_combinations}")

        self.logger.info(f"Found data for {len(found_combinations)} combinations: {found_combinations}")

        # Generate appropriate heatmap based on metric
        if metric == 'psnr':
            return self.generate_psnr_heatmap(
                results_matrix, peft_methods, defense_methods,
                title=f"Defense Effectiveness: {metric.upper()} Analysis",
                save_path=save_path
            )
        else:
            return self.generate_attack_success_heatmap(
                results_matrix, peft_methods, defense_methods,
                title=f"Attack Analysis: {metric.upper()} Results",
                save_path=save_path
            )

    def _extract_metric_value(self, data_dict: Dict[str, Any], metric: str) -> float:
        """
        Extract metric value from nested dictionary structure

        Args:
            data_dict: Dictionary containing metrics
            metric: Metric name to extract

        Returns:
            Extracted metric value
        """
        if not isinstance(data_dict, dict):
            return 0.0

        # Define possible metric key variations
        metric_keys = {
            'psnr': [
                'avg_psnr', 'psnr_mean', 'mean_psnr', 'psnr',
                'evaluation_metrics.psnr_mean'
            ],
            'ssim': [
                'avg_ssim', 'ssim_mean', 'mean_ssim', 'ssim',
                'evaluation_metrics.ssim_mean'
            ],
            'success_rate': [
                'attack_success_rate_with_defense', 'success_rate',
                'attack_success_rate', 'success_rate_with_defense'
            ]
        }

        possible_keys = metric_keys.get(metric, [metric])

        # Try direct key lookup
        for key in possible_keys:
            if key in data_dict:
                value = data_dict[key]
                if isinstance(value, (int, float)):
                    return float(value)

        # Try nested lookup in evaluation_metrics
        if 'evaluation_metrics' in data_dict:
            eval_metrics = data_dict['evaluation_metrics']
            if isinstance(eval_metrics, dict):
                for key in possible_keys:
                    # Remove 'evaluation_metrics.' prefix if present
                    clean_key = key.replace('evaluation_metrics.', '')
                    if clean_key in eval_metrics:
                        value = eval_metrics[clean_key]
                        if isinstance(value, (int, float)):
                            return float(value)

        return 0.0

    def debug_results_structure(self, results_dict: Dict[str, Any]):
        """
        Debug function to understand the structure of results_dict

        Args:
            results_dict: Results dictionary to analyze
        """
        print("=== RESULTS STRUCTURE DEBUG ===")
        print(f"Total keys in results_dict: {len(results_dict)}")

        for key, value in results_dict.items():
            print(f"\nKey: '{key}'")
            print(f"  Type: {type(value)}")

            if isinstance(value, dict):
                if any(peft in value for peft in ['adapter', 'prefix', 'bias', 'lora']):
                    print(
                        f"  Contains PEFT methods: {[k for k in value.keys() if k in ['adapter', 'prefix', 'bias', 'lora']]}")
                    # Show structure for first PEFT method found
                    for peft_method in ['adapter', 'prefix', 'bias', 'lora']:
                        if peft_method in value:
                            peft_data = value[peft_method]
                            if isinstance(peft_data, dict):
                                print(f"    {peft_method} keys: {list(peft_data.keys())}")
                            break
                else:
                    print(f"  Keys: {list(value.keys())[:5]}{'...' if len(value) > 5 else ''}")
            elif isinstance(value, (list, tuple)):
                print(f"  Length: {len(value)}")
            else:
                print(f"  Value: {str(value)[:50]}{'...' if len(str(value)) > 50 else ''}")

        print("=== END DEBUG ===")

    def create_comparison_heatmaps(self, before_results: Dict[str, Any],
                                   after_results: Dict[str, Any],
                                   peft_methods: List[str],
                                   defense_methods: List[str]) -> List[str]:
        """
        Create before/after comparison heatmaps

        Args:
            before_results: Results before defense application
            after_results: Results after defense application
            peft_methods: List of PEFT methods
            defense_methods: List of defense methods

        Returns:
            List of paths to saved heatmaps
        """
        saved_paths = []

        # Generate before heatmap
        before_path = self.generate_comprehensive_heatmap(
            before_results, peft_methods, defense_methods,
            metric='psnr',
            save_path=os.path.join(self.save_dir, 'before_defense_heatmap.png')
        )
        saved_paths.append(before_path)

        # Generate after heatmap
        after_path = self.generate_comprehensive_heatmap(
            after_results, peft_methods, defense_methods,
            metric='psnr',
            save_path=os.path.join(self.save_dir, 'after_defense_heatmap.png')
        )
        saved_paths.append(after_path)

        # Generate improvement heatmap
        improvement_matrix = np.zeros((len(peft_methods), len(defense_methods)))

        for i, peft_method in enumerate(peft_methods):
            for j, defense_method in enumerate(defense_methods):
                # Use the same extraction logic as the main function
                before_val = self._extract_value_from_results(before_results, peft_method, defense_method, 'psnr')
                after_val = self._extract_value_from_results(after_results, peft_method, defense_method, 'psnr')
                improvement_matrix[i, j] = before_val - after_val  # Positive = defense worked

        plt.figure(figsize=(len(defense_methods) * 1.5, len(peft_methods) * 1.2))

        heatmap = sns.heatmap(
            improvement_matrix,
            xticklabels=[method.title() for method in defense_methods],
            yticklabels=[method.upper() for method in peft_methods],
            annot=True,
            fmt='.2f',
            cmap='RdYlGn',  # Green = good defense (positive improvement)
            center=0.0,
            cbar_kws={'label': 'PSNR Reduction (dB)'},
            linewidths=0.5
        )

        plt.title('Defense Improvement: PSNR Reduction', fontsize=16, fontweight='bold')
        plt.xlabel('Defense Method', fontsize=14)
        plt.ylabel('PEFT Method', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)

        plt.figtext(0.02, 0.02, 'Positive values = Effective defense',
                    fontsize=10, style='italic', color='gray')

        plt.tight_layout()

        improvement_path = os.path.join(self.save_dir, 'defense_improvement_heatmap.png')
        plt.savefig(improvement_path, dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.close()

        saved_paths.append(improvement_path)

        self.logger.info(f"Comparison heatmaps saved: {len(saved_paths)} files")
        return saved_paths

    def _extract_value_from_results(self, results_dict: Dict[str, Any],
                                    peft_method: str, defense_method: str, metric: str) -> float:
        """
        Helper function to extract value using the same logic as generate_comprehensive_heatmap
        """
        # Try multiple key formats
        possible_keys = [
            f"{peft_method}_{defense_method}",
            f"{defense_method}_{peft_method}",
            defense_method,
            peft_method
        ]

        for key in possible_keys:
            if key in results_dict:
                result_data = results_dict[key]
                if isinstance(result_data, dict):
                    if peft_method in result_data:
                        peft_data = result_data[peft_method]
                        if isinstance(peft_data, dict):
                            return self._extract_metric_value(peft_data, metric)
                    else:
                        return self._extract_metric_value(result_data, metric)

        # Try defense method directly
        if defense_method in results_dict:
            defense_data = results_dict[defense_method]
            if isinstance(defense_data, dict) and peft_method in defense_data:
                peft_data = defense_data[peft_method]
                if isinstance(peft_data, dict):
                    return self._extract_metric_value(peft_data, metric)

        return 0.0


# Example usage
if __name__ == "__main__":
    generator = HeatmapGenerator()

    # Example data including combined defenses
    peft_methods = ['adapter', 'prefix', 'bias', 'lora']
    defense_methods = ['mixup', 'instahide', 'dp', 'gradprune', 'combined', 'adaptive_comprehensive']

    # Example PSNR matrix with combined defenses
    psnr_matrix = np.array([
        [15.2, 18.5, 22.1, 25.3, 19.0, 17.5],  # adapter
        [12.8, 16.2, 19.7, 23.1, 16.5, 15.2],  # prefix
        [18.9, 21.4, 24.6, 27.2, 21.0, 19.8],  # bias
        [14.1, 17.8, 20.9, 24.5, 18.2, 16.9]  # lora
    ])

    generator.generate_psnr_heatmap(psnr_matrix, peft_methods, defense_methods)
    print("Example heatmap with combined defenses generated successfully!")

    # Test the debug function
    example_results = {
        'mixup': {
            'adapter': {'avg_psnr': 15.2, 'evaluation_metrics': {'psnr_mean': 15.2}},
            'prefix': {'avg_psnr': 12.8}
        },
        'comprehensive': {
            'adapter': {'evaluation_metrics': {'psnr_mean': 19.0}},
            'prefix': {'evaluation_metrics': {'psnr_mean': 16.5}}
        }
    }

    generator.debug_results_structure(example_results)
