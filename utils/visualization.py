import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Optional, Dict #visualization.py
import torch


class ResultVisualizer:
    """Visualization utilities for attack and defense results"""

    def __init__(self, style: str = 'seaborn'):
        plt.style.use(style if style in plt.style.available else 'default')

    def plot_patch_comparison(self, original_patches: List[torch.Tensor],
                              recovered_patches: List[torch.Tensor],
                              title: str = "Patch Recovery Results",
                              max_patches: int = 16,
                              save_path: Optional[str] = None):
        """
        Plot side-by-side comparison of original and recovered patches
        """
        num_patches = min(len(original_patches), len(recovered_patches), max_patches)

        fig, axes = plt.subplots(2, num_patches, figsize=(num_patches * 1.5, 3))

        if num_patches == 1:
            axes = axes.reshape(2, 1)

        for i in range(num_patches):
            # Original patches
            orig_patch = original_patches[i].permute(1, 2, 0).numpy()
            axes[0, i].imshow(np.clip(orig_patch, 0, 1))
            axes[0, i].set_title(f'Orig {i + 1}', fontsize=8)
            axes[0, i].axis('off')

            # Recovered patches
            rec_patch = recovered_patches[i].permute(1, 2, 0).numpy()
            axes[1, i].imshow(np.clip(rec_patch, 0, 1))
            axes[1, i].set_title(f'Rec {i + 1}', fontsize=8)
            axes[1, i].axis('off')

        plt.suptitle(title, fontsize=14)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

    def plot_defense_effectiveness_heatmap(self, results_matrix: np.ndarray,
                                           peft_methods: List[str],
                                           defense_methods: List[str],
                                           metric_name: str = "PSNR",
                                           title: Optional[str] = None,
                                           save_path: Optional[str] = None):
        """
        Plot heatmap showing defense effectiveness across PEFT methods
        """
        plt.figure(figsize=(len(defense_methods) * 1.2, len(peft_methods) * 0.8))

        # Create heatmap
        sns.heatmap(results_matrix,
                    xticklabels=defense_methods,
                    yticklabels=peft_methods,
                    annot=True,
                    fmt='.2f',
                    cmap='RdYlBu_r',  # Red = high values (less effective defense)
                    center=20 if metric_name == "PSNR" else 0.5,
                    cbar_kws={'label': f'{metric_name} Value'})

        if title is None:
            title = f'Defense Effectiveness: {metric_name} Across PEFT Methods'

        plt.title(title, fontsize=14)
        plt.xlabel('Defense Mechanism', fontsize=12)
        plt.ylabel('PEFT Method', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

    def plot_attack_success_trends(self, round_results: List[Dict],
                                   peft_methods: List[str],
                                   save_path: Optional[str] = None):
        """
        Plot attack success rates over federated learning rounds
        """
        fig, axes = plt.subplots(1, len(peft_methods), figsize=(len(peft_methods) * 4, 4))

        if len(peft_methods) == 1:
            axes = [axes]

        for i, method in enumerate(peft_methods):
            rounds = []
            success_rates = []

            for round_result in round_results:
                if method in round_result:
                    rounds.append(round_result['round'])
                    success_rates.append(round_result[method].get('success_rate', 0))

            axes[i].plot(rounds, success_rates, 'o-', linewidth=2, markersize=6)
            axes[i].set_title(f'{method.upper()} Attack Success', fontsize=12)
            axes[i].set_xlabel('FL Round')
            axes[i].set_ylabel('Success Rate')
            axes[i].set_ylim(0, 1)
            axes[i].grid(True, alpha=0.3)

        plt.suptitle('Attack Success Rates Over FL Rounds', fontsize=14)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

    def plot_metrics_distribution(self, metrics_dict: Dict[str, List[float]],
                                  title: str = "Metrics Distribution",
                                  save_path: Optional[str] = None):
        """
        Plot distribution of different metrics
        """
        num_metrics = len(metrics_dict)
        fig, axes = plt.subplots(1, num_metrics, figsize=(num_metrics * 4, 4))

        if num_metrics == 1:
            axes = [axes]

        for i, (metric_name, values) in enumerate(metrics_dict.items()):
            axes[i].hist(values, bins=20, alpha=0.7, edgecolor='black')
            axes[i].set_title(f'{metric_name} Distribution')
            axes[i].set_xlabel(metric_name)
            axes[i].set_ylabel('Frequency')
            axes[i].grid(True, alpha=0.3)

            # Add statistics text
            mean_val = np.mean(values)
            std_val = np.std(values)
            axes[i].axvline(mean_val, color='red', linestyle='--',
                            label=f'Mean: {mean_val:.2f}±{std_val:.2f}')
            axes[i].legend()

        plt.suptitle(title, fontsize=14)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

    def create_comprehensive_report_figure(self, attack_results: Dict,
                                           defense_results: Dict,
                                           save_path: Optional[str] = None):
        """
        Create a comprehensive figure with multiple subplots for a complete report
        """
        fig = plt.figure(figsize=(16, 12))

        # Subplot 1: Defense effectiveness heatmap
        plt.subplot(2, 3, 1)
        # Implementation depends on data structure
        plt.title('Defense Effectiveness')

        # Subplot 2: Attack success by PEFT method
        plt.subplot(2, 3, 2)
        # Implementation depends on data structure
        plt.title('Attack Success by PEFT')

        # Subplot 3: Metrics distribution
        plt.subplot(2, 3, 3)
        # Implementation depends on data structure
        plt.title('Quality Metrics')

        # Subplot 4: Sample recovered patches
        plt.subplot(2, 3, 4)
        plt.title('Sample Recovery')

        # Subplot 5: Privacy vs Utility tradeoff
        plt.subplot(2, 3, 5)
        plt.title('Privacy-Utility Tradeoff')

        # Subplot 6: Summary statistics
        plt.subplot(2, 3, 6)
        plt.title('Summary Statistics')

        plt.suptitle('Comprehensive Attack & Defense Analysis', fontsize=16)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()
