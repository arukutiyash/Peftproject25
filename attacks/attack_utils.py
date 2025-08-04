import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
import matplotlib.pyplot as plt #attack_utils.py
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import torchvision.transforms as transforms
from PIL import Image
import os


class AttackUtils:
    """Utility functions for gradient inversion attacks"""

    @staticmethod
    def create_cifar100_dataloader(batch_size: int = 32,
                                   subset_size: Optional[int] = None) -> torch.utils.data.DataLoader:
        """Create CIFAR-100 dataloader for attack testing"""
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])

        dataset = torchvision.datasets.CIFAR100(
            root='./data', train=True, download=True, transform=transform
        )

        if subset_size:
            # Create subset for faster testing
            indices = torch.randperm(len(dataset))[:subset_size]
            dataset = torch.utils.data.Subset(dataset, indices)

        return torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=2
        )

    @staticmethod
    def extract_ground_truth_patches(images: torch.Tensor, patch_size: int = 4) -> List[torch.Tensor]:
        """Extract ground truth patches from images for evaluation"""
        patches = []
        batch_size, channels, height, width = images.shape

        for img in images:
            # Extract patches
            img_patches = img.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
            img_patches = img_patches.contiguous().view(channels, -1, patch_size, patch_size)
            img_patches = img_patches.permute(1, 0, 2, 3)  # [num_patches, channels, h, w]

            for patch in img_patches:
                patches.append(patch)

        return patches

    @staticmethod
    def visualize_patch_comparison(original_patches: List[torch.Tensor],
                                   recovered_patches: List[torch.Tensor],
                                   save_path: Optional[str] = None,
                                   max_patches: int = 16) -> None:
        """Visualize comparison between original and recovered patches"""

        num_patches = min(len(original_patches), len(recovered_patches), max_patches)

        fig, axes = plt.subplots(2, num_patches, figsize=(num_patches * 2, 4))

        for i in range(num_patches):
            # Original patch
            orig_patch = original_patches[i].detach().cpu().numpy()
            if orig_patch.shape[0] == 3:  # CHW format
                orig_patch = np.transpose(orig_patch, (1, 2, 0))

            axes[0, i].imshow(np.clip(orig_patch, 0, 1))
            axes[0, i].set_title(f'Original {i + 1}')
            axes[0, i].axis('off')

            # Recovered patch
            rec_patch = recovered_patches[i].detach().cpu().numpy()
            if rec_patch.shape[0] == 3:  # CHW format
                rec_patch = np.transpose(rec_patch, (1, 2, 0))

            axes[1, i].imshow(np.clip(rec_patch, 0, 1))
            axes[1, i].set_title(f'Recovered {i + 1}')
            axes[1, i].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    @staticmethod
    def compute_attack_metrics(original_patches: List[torch.Tensor],
                               recovered_patches: List[torch.Tensor]) -> Dict[str, float]:
        """Compute comprehensive attack success metrics"""
        if not original_patches or not recovered_patches:
            return {'psnr': 0.0, 'ssim': 0.0, 'mse': float('inf'), 'lpips': 1.0}

        metrics = {'psnr': [], 'ssim': [], 'mse': [], 'cosine_sim': []}

        min_len = min(len(original_patches), len(recovered_patches))

        for i in range(min_len):
            orig = original_patches[i]
            rec = recovered_patches[i]

            if orig.shape != rec.shape:
                continue

            # PSNR
            mse = torch.mean((orig - rec) ** 2)
            if mse > 0:
                psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
                metrics['psnr'].append(psnr.item())

            # SSIM (simplified)
            ssim = AttackUtils._compute_ssim(orig, rec)
            metrics['ssim'].append(ssim)

            # MSE
            metrics['mse'].append(mse.item())

            # Cosine similarity
            orig_flat = orig.flatten()
            rec_flat = rec.flatten()
            cosine_sim = torch.cosine_similarity(orig_flat.unsqueeze(0), rec_flat.unsqueeze(0))
            metrics['cosine_sim'].append(cosine_sim.item())

        # Average metrics
        final_metrics = {}
        for metric_name, values in metrics.items():
            if values:
                final_metrics[f'avg_{metric_name}'] = np.mean(values)
                final_metrics[f'std_{metric_name}'] = np.std(values)
                final_metrics[f'max_{metric_name}'] = np.max(values)
            else:
                final_metrics[f'avg_{metric_name}'] = 0.0
                final_metrics[f'std_{metric_name}'] = 0.0
                final_metrics[f'max_{metric_name}'] = 0.0

        # Success rate (patches with PSNR > 20 dB)
        good_patches = len([p for p in metrics['psnr'] if p > 20])
        final_metrics['success_rate'] = good_patches / len(metrics['psnr']) if metrics['psnr'] else 0.0

        return final_metrics

    @staticmethod
    def _compute_ssim(img1: torch.Tensor, img2: torch.Tensor) -> float:
        """Compute SSIM between two images"""
        mu1 = torch.mean(img1)
        mu2 = torch.mean(img2)

        sigma1_sq = torch.var(img1)
        sigma2_sq = torch.var(img2)
        sigma12 = torch.mean((img1 - mu1) * (img2 - mu2))

        c1 = 0.01 ** 2
        c2 = 0.03 ** 2

        numerator = (2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)
        denominator = (mu1 ** 2 + mu2 ** 2 + c1) * (sigma1_sq + sigma2_sq + c2)

        ssim = numerator / denominator
        return ssim.item()

    @staticmethod
    def cluster_patches(patches: List[torch.Tensor], num_clusters: Optional[int] = None) -> Dict[int, List[int]]:
        """Cluster patches to group similar ones together"""
        if not patches:
            return {}

        # Convert patches to feature vectors
        patch_vectors = []
        for patch in patches:
            patch_vector = patch.flatten().numpy()
            patch_vectors.append(patch_vector)

        patch_vectors = np.array(patch_vectors)

        # Determine number of clusters
        if num_clusters is None:
            num_clusters = min(8, len(patches))

        # Perform clustering
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(patch_vectors)

        # Group patches by cluster
        clusters = {}
        for i, label in enumerate(cluster_labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(i)

        return clusters

    @staticmethod
    def analyze_gradient_patterns(gradients: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Analyze gradient patterns for insights"""
        analysis = {
            'gradient_norms': {},
            'gradient_sparsity': {},
            'gradient_statistics': {},
            'dominant_components': {}
        }

        for name, grad in gradients.items():
            if isinstance(grad, torch.Tensor):
                # Gradient norm
                analysis['gradient_norms'][name] = torch.norm(grad).item()

                # Sparsity (percentage of near-zero values)
                threshold = 1e-6
                near_zero = torch.abs(grad) < threshold
                sparsity = near_zero.float().mean().item()
                analysis['gradient_sparsity'][name] = sparsity

                # Basic statistics
                grad_flat = grad.flatten()
                analysis['gradient_statistics'][name] = {
                    'mean': torch.mean(grad_flat).item(),
                    'std': torch.std(grad_flat).item(),
                    'min': torch.min(grad_flat).item(),
                    'max': torch.max(grad_flat).item(),
                    'median': torch.median(grad_flat).item()
                }

                # Dominant components (if 2D)
                if grad.dim() == 2:
                    try:
                        U, S, V = torch.svd(grad)
                        # Store top 3 singular values
                        top_svs = S[:3].tolist() if len(S) >= 3 else S.tolist()
                        analysis['dominant_components'][name] = {
                            'top_singular_values': top_svs,
                            'rank_estimate': torch.sum(S > S[0] * 0.1).item()  # 10% threshold
                        }
                    except:
                        analysis['dominant_components'][name] = {'error': 'SVD failed'}

        return analysis

    @staticmethod
    def create_attack_comparison_plot(attack_results: Dict[str, Dict[str, Any]],
                                      save_path: Optional[str] = None) -> None:
        """Create comparison plot of attack results across PEFT methods"""

        peft_methods = list(attack_results.keys())
        metrics = ['avg_psnr', 'avg_ssim', 'success_rate']

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        for i, metric in enumerate(metrics):
            values = []
            for method in peft_methods:
                if metric in attack_results[method]:
                    values.append(attack_results[method][metric])
                else:
                    values.append(0.0)

            axes[i].bar(peft_methods, values)
            axes[i].set_title(f'{metric.replace("_", " ").title()}')
            axes[i].set_ylabel(metric)

            # Add value labels on bars
            for j, v in enumerate(values):
                axes[i].text(j, v + 0.01, f'{v:.3f}', ha='center', va='bottom')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    @staticmethod
    def save_attack_report(attack_results: Dict[str, Any], filepath: str) -> None:
        """Save detailed attack report to text file"""

        with open(filepath, 'w') as f:
            f.write("GRADIENT INVERSION ATTACK REPORT\n")
            f.write("=" * 40 + "\n\n")

            # Basic info
            f.write(f"Attack Type: {attack_results.get('attack_type', 'Unknown')}\n")
            f.write(f"PEFT Method: {attack_results.get('peft_method', 'Unknown')}\n")
            f.write(f"Timestamp: {attack_results.get('timestamp', 'Unknown')}\n\n")

            # Configuration
            if 'config' in attack_results:
                f.write("Configuration:\n")
                for key, value in attack_results['config'].items():
                    f.write(f"  {key}: {value}\n")
                f.write("\n")

            # Metrics
            if 'metrics' in attack_results:
                f.write("Attack Metrics:\n")
                metrics = attack_results['metrics']
                for metric_name, value in metrics.items():
                    if isinstance(value, float):
                        f.write(f"  {metric_name}: {value:.6f}\n")
                    else:
                        f.write(f"  {metric_name}: {value}\n")
                f.write("\n")

            # Multi-round results
            if 'round_metrics' in attack_results:
                f.write("Round-by-Round Results:\n")
                for i, round_metrics in enumerate(attack_results['round_metrics']):
                    f.write(f"  Round {i + 1}:\n")
                    for metric_name, value in round_metrics.items():
                        if isinstance(value, float):
                            f.write(f"    {metric_name}: {value:.6f}\n")
                        else:
                            f.write(f"    {metric_name}: {value}\n")
                f.write("\n")

            # Summary
            f.write("Summary:\n")
            f.write(f"  Total patches recovered: {len(attack_results.get('recovered_patches', []))}\n")
            f.write(f"  Attack successful: {'Yes' if attack_results.get('success_rate', 0) > 0.5 else 'No'}\n")

        print(f"Attack report saved to {filepath}")

    @staticmethod
    def create_heatmap_data(results: Dict[str, Dict[str, float]],
                            peft_methods: List[str],
                            defense_methods: List[str]) -> np.ndarray:
        """Create heatmap data matrix for visualization"""

        heatmap_data = np.zeros((len(peft_methods), len(defense_methods)))

        for i, peft_method in enumerate(peft_methods):
            for j, defense_method in enumerate(defense_methods):
                key = f"{peft_method}_{defense_method}"
                if key in results:
                    # Use PSNR as the primary metric (higher = less effective defense)
                    heatmap_data[i, j] = results[key].get('avg_psnr', 0.0)

        return heatmap_data

    @staticmethod
    def plot_defense_effectiveness_heatmap(heatmap_data: np.ndarray,
                                           peft_methods: List[str],
                                           defense_methods: List[str],
                                           save_path: Optional[str] = None) -> None:
        """Plot heatmap showing defense effectiveness across PEFT methods"""

        plt.figure(figsize=(12, 8))

        # Create heatmap
        sns.heatmap(heatmap_data,
                    xticklabels=defense_methods,
                    yticklabels=peft_methods,
                    annot=True,
                    fmt='.2f',
                    cmap='RdYlBu_r',  # Red = high PSNR = less effective defense
                    center=20,  # Center around 20 dB PSNR
                    cbar_kws={'label': 'PSNR (dB)'})

        plt.title('Defense Effectiveness Against Gradient Inversion Attacks\n(Higher PSNR = Less Effective Defense)',
                  fontsize=14)
        plt.xlabel('Defense Method', fontsize=12)
        plt.ylabel('PEFT Method', fontsize=12)

        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


# Example usage functions
def run_comprehensive_attack_test():
    """Example function showing how to use the attack framework"""
