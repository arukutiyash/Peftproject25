import torch
import numpy as np
from typing import List, Dict, Union
import math #metrics.py


class MetricsCalculator:
    """Calculate image quality metrics for attack evaluation"""

    def __init__(self, device: str = 'cpu'):
        self.device = device

    def compute_psnr(self, img1: torch.Tensor, img2: torch.Tensor, max_val: float = 1.0) -> float:
        """
        Compute Peak Signal-to-Noise Ratio
        Args:
            img1: First image tensor
            img2: Second image tensor
            max_val: Maximum possible pixel value
        Returns:
            PSNR value in dB
        """
        mse = torch.mean((img1 - img2) ** 2).item()

        if mse == 0:
            return float('inf')

        psnr = 20 * math.log10(max_val / math.sqrt(mse))
        return psnr

    def compute_ssim(self, img1: torch.Tensor, img2: torch.Tensor) -> float:
        """
        Compute Structural Similarity Index (simplified implementation)
        Args:
            img1: First image tensor
            img2: Second image tensor
        Returns:
            SSIM value between 0 and 1
        """
        # Constants for stability
        C1 = (0.01) ** 2
        C2 = (0.03) ** 2

        # Compute means
        mu1 = torch.mean(img1)
        mu2 = torch.mean(img2)

        # Compute variances and covariance
        sigma1_sq = torch.var(img1)
        sigma2_sq = torch.var(img2)
        sigma12 = torch.mean((img1 - mu1) * (img2 - mu2))

        # Compute SSIM
        numerator = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
        denominator = (mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2)

        ssim = numerator / denominator
        return ssim.item()

    def compute_lpips_placeholder(self, img1: torch.Tensor, img2: torch.Tensor) -> float:
        """
        Placeholder for LPIPS calculation (requires external model)
        Args:
            img1: First image tensor
            img2: Second image tensor
        Returns:
            Placeholder LPIPS value
        """
        # This is a placeholder - actual LPIPS requires pretrained VGG/AlexNet
        # For now, return a heuristic based on MSE
        mse = torch.mean((img1 - img2) ** 2).item()
        lpips_approx = min(1.0, mse * 10)  # Rough approximation
        return lpips_approx

    def evaluate_reconstruction_quality(self, original_images: List[torch.Tensor],
                                        reconstructed_images: List[torch.Tensor]) -> Dict[str, float]:
        """
        Comprehensive evaluation of reconstruction quality
        Args:
            original_images: List of original image tensors
            reconstructed_images: List of reconstructed image tensors
        Returns:
            Dictionary of metric values
        """
        psnr_values = []
        ssim_values = []
        lpips_values = []

        min_len = min(len(original_images), len(reconstructed_images))

        for i in range(min_len):
            orig = original_images[i]
            recon = reconstructed_images[i]

            # Ensure same dimensions
            if orig.shape != recon.shape:
                continue

            # Compute metrics
            psnr_val = self.compute_psnr(orig, recon)
            ssim_val = self.compute_ssim(orig, recon)
            lpips_val = self.compute_lpips_placeholder(orig, recon)

            psnr_values.append(psnr_val)
            ssim_values.append(ssim_val)
            lpips_values.append(lpips_val)

        # Compute statistics
        results = {
            'psnr_mean': np.mean(psnr_values) if psnr_values else 0.0,
            'psnr_std': np.std(psnr_values) if psnr_values else 0.0,
            'psnr_max': np.max(psnr_values) if psnr_values else 0.0,
            'ssim_mean': np.mean(ssim_values) if ssim_values else 0.0,
            'ssim_std': np.std(ssim_values) if ssim_values else 0.0,
            'lpips_mean': np.mean(lpips_values) if lpips_values else 0.0,
            'lpips_std': np.std(lpips_values) if lpips_values else 0.0,
            'num_evaluated': len(psnr_values)
        }

        return results

    def compute_attack_success_rate(self, psnr_values: List[float],
                                    threshold: float = 20.0) -> float:
        """
        Compute attack success rate based on PSNR threshold
        Args:
            psnr_values: List of PSNR values
            threshold: PSNR threshold for successful attack
        Returns:
            Success rate between 0 and 1
        """
        if not psnr_values:
            return 0.0

        successful_attacks = sum(1 for psnr in psnr_values if psnr >= threshold)
        return successful_attacks / len(psnr_values)
