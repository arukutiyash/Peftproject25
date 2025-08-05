import torch
import torch.nn as nn
import numpy as np #attck_base.py
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import torchvision.transforms as transforms
from PIL import Image


class BaseGradientInversionAttack(ABC):
    """Base class for gradient inversion attacks on PEFT methods"""

    def __init__(self, model: nn.Module, device: str = 'cuda', recovery_method:
                 str = "direct", patch_size: int = 4, img_size: int = 32, num_patches: int = 64, **kwargs):
        """
        Initialize base attack

        Args:
            model: The target model with PEFT method
            device: Computing device
            patch_size: Size of image patches
            img_size: Image size (assumed square)
            num_patches: Number of patches per image
        """
        self.model = model.to(device)
        self.device = device
        self.patch_size = patch_size
        self.img_size = img_size
        self.num_patches = num_patches

        # Attack configuration
        self.attack_config = {
            'max_iterations': 1000,
            'learning_rate': 0.01,
            'convergence_threshold': 1e-6,
            'noise_scale': 0.1,
            'regularization_weight': 0.01
        }

        # Results storage
        self.attack_results = {
            'recovered_patches': [],
            'original_patches': [],
            'gradients': {},
            'loss_history': [],
            'success_rate': 0.0,
            'psnr': 0.0,
            'ssim': 0.0
        }

    @abstractmethod
    def setup_malicious_parameters(self):
        """Setup malicious model parameters for the attack"""
        pass

    @abstractmethod
    def extract_target_gradients(self, data_batch: torch.Tensor, labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract gradients specific to the PEFT method"""
        pass

    @abstractmethod
    def reconstruct_patches(self, gradients: Dict[str, torch.Tensor]) -> List[torch.Tensor]:
        """Reconstruct patches from extracted gradients"""
        pass

    def compute_loss(self, pred_logits: torch.Tensor, true_labels: torch.Tensor) -> torch.Tensor:
        """Compute loss for gradient computation"""
        return nn.CrossEntropyLoss()(pred_logits, true_labels)

    def get_patch_embeddings(self, images: torch.Tensor) -> torch.Tensor:
        """Convert images to patch embeddings"""
        batch_size, channels, height, width = images.shape

        # Reshape to patches
        patches = images.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches = patches.contiguous().view(batch_size, channels, -1, self.patch_size, self.patch_size)
        patches = patches.permute(0, 2, 1, 3, 4).contiguous().view(batch_size, -1,
                                                                   channels * self.patch_size * self.patch_size)

        return patches

    def patches_to_image(self, patches: torch.Tensor) -> torch.Tensor:
        """Convert patches back to image"""
        batch_size, num_patches, patch_dim = patches.shape
        channels = 3

        # Reshape patches
        patches = patches.view(batch_size, num_patches, channels, self.patch_size, self.patch_size)
        patches = patches.permute(0, 2, 1, 3, 4)

        # Calculate grid dimensions
        patches_per_side = int(np.sqrt(num_patches))

        # Reconstruct image
        image = patches.view(batch_size, channels, patches_per_side, patches_per_side, self.patch_size, self.patch_size)
        image = image.permute(0, 1, 2, 4, 3, 5).contiguous()
        image = image.view(batch_size, channels, self.img_size, self.img_size)

        return image

    def compute_psnr(self, original: torch.Tensor, reconstructed: torch.Tensor) -> float:
        """Compute Peak Signal-to-Noise Ratio"""
        mse = torch.mean((original - reconstructed) ** 2)
        if mse == 0:
            return float('inf')
        max_pixel = 1.0
        psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
        return psnr.item()

    def compute_ssim(self, original: torch.Tensor, reconstructed: torch.Tensor) -> float:
        """Simplified SSIM computation"""
        # This is a simplified version - for full SSIM, use torchmetrics or similar
        mu1 = torch.mean(original)
        mu2 = torch.mean(reconstructed)

        sigma1_sq = torch.var(original)
        sigma2_sq = torch.var(reconstructed)
        sigma12 = torch.mean((original - mu1) * (reconstructed - mu2))

        c1 = 0.01 ** 2
        c2 = 0.03 ** 2

        numerator = (2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)
        denominator = (mu1 ** 2 + mu2 ** 2 + c1) * (sigma1_sq + sigma2_sq + c2)

        ssim = numerator / denominator
        return ssim.item()

    def visualize_attack_results(self, save_path: Optional[str] = None):
        """Visualize original vs reconstructed patches"""
        if not self.attack_results['recovered_patches'] or not self.attack_results['original_patches']:
            print("No attack results to visualize")
            return

        fig, axes = plt.subplots(2, min(8, len(self.attack_results['recovered_patches'])),
                                 figsize=(16, 4))

        for i in range(min(8, len(self.attack_results['recovered_patches']))):
            # Original patches
            if len(self.attack_results['original_patches']) > i:
                original_patch = self.attack_results['original_patches'][i].detach().cpu().numpy()
                if original_patch.shape[0] == 3:  # CHW format
                    original_patch = np.transpose(original_patch, (1, 2, 0))
                axes[0, i].imshow(np.clip(original_patch, 0, 1))
                axes[0, i].set_title(f'Original {i + 1}')
                axes[0, i].axis('off')

            # Reconstructed patches
            reconstructed_patch = self.attack_results['recovered_patches'][i].detach().cpu().numpy()
            if reconstructed_patch.shape[0] == 3:  # CHW format
                reconstructed_patch = np.transpose(reconstructed_patch, (1, 2, 0))
            axes[1, i].imshow(np.clip(reconstructed_patch, 0, 1))
            axes[1, i].set_title(f'Reconstructed {i + 1}')
            axes[1, i].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def evaluate_attack_success(self, original_patches: List[torch.Tensor],
                                reconstructed_patches: List[torch.Tensor]) -> Dict[str, float]:
        """Evaluate attack success metrics"""
        if not original_patches or not reconstructed_patches:
            return {'psnr': 0.0, 'ssim': 0.0, 'mse': float('inf')}

        psnr_values = []
        ssim_values = []
        mse_values = []

        min_len = min(len(original_patches), len(reconstructed_patches))

        for i in range(min_len):
            orig = original_patches[i]
            recon = reconstructed_patches[i]

            # Ensure same shape
            if orig.shape != recon.shape:
                continue

            psnr = self.compute_psnr(orig, recon)
            ssim = self.compute_ssim(orig, recon)
            mse = torch.mean((orig - recon) ** 2).item()

            psnr_values.append(psnr)
            ssim_values.append(ssim)
            mse_values.append(mse)

        return {
            'psnr': np.mean(psnr_values) if psnr_values else 0.0,
            'ssim': np.mean(ssim_values) if ssim_values else 0.0,
            'mse': np.mean(mse_values) if mse_values else float('inf'),
            'success_rate': len([p for p in psnr_values if p > 20]) / len(psnr_values) if psnr_values else 0.0
        }

    def run_attack(self, data_batch: torch.Tensor, labels: torch.Tensor,
                   target_patches: Optional[List[torch.Tensor]] = None) -> Dict[str, Any]:
        """
        Run the complete gradient inversion attack

        Args:
            data_batch: Input images
            labels: True labels
            target_patches: Ground truth patches for evaluation

        Returns:
            Attack results dictionary
        """
        print(f"Running {self.__class__.__name__} attack...")

        # Setup malicious parameters
        self.setup_malicious_parameters()

        # Extract gradients
        gradients = self.extract_target_gradients(data_batch, labels)
        self.attack_results['gradients'] = gradients

        # Reconstruct patches
        recovered_patches = self.reconstruct_patches(gradients)
        self.attack_results['recovered_patches'] = recovered_patches

        # Evaluate if target patches provided
        if target_patches:
            self.attack_results['original_patches'] = target_patches
            metrics = self.evaluate_attack_success(target_patches, recovered_patches)
            self.attack_results.update(metrics)

            print(f"Attack Results:")
            print(f"  PSNR: {metrics['psnr']:.2f} dB")
            print(f"  SSIM: {metrics['ssim']:.4f}")
            print(f"  MSE: {metrics['mse']:.6f}")
            print(f"  Success Rate: {metrics['success_rate']:.2%}")

        print(f"Recovered {len(recovered_patches)} patches")

        return self.attack_results

    def save_attack_results(self, filepath: str):
        """Save attack results to file"""
        results_to_save = {
            'attack_type': self.__class__.__name__,
            'config': self.attack_config,
            'metrics': {
                'psnr': self.attack_results.get('psnr', 0.0),
                'ssim': self.attack_results.get('ssim', 0.0),
                'mse': self.attack_results.get('mse', 0.0),
                'success_rate': self.attack_results.get('success_rate', 0.0)
            },
            'num_recovered_patches': len(self.attack_results.get('recovered_patches', []))
        }

        torch.save(results_to_save, filepath)
        print(f"Attack results saved to {filepath}")
