import torch
import numpy as np
from torchvision import datasets, transforms #data_processing.py
from typing import List, Tuple, Dict, Optional
import torch.nn.functional as F


class DataProcessor:
    """Enhanced data processing for patch extraction and statistical analysis"""

    def __init__(self, patch_size: int = 4, device: str = 'cpu'):
        self.patch_size = patch_size
        self.device = device

    def extract_patches(self, images: torch.Tensor) -> List[torch.Tensor]:
        """
        Extract non-overlapping patches from images
        Args:
            images: tensor of shape (B, C, H, W)
        Returns:
            List of patches as tensors (C, patch_size, patch_size)
        """
        B, C, H, W = images.shape
        patches = []

        for i in range(B):
            img = images[i]
            for y in range(0, H, self.patch_size):
                for x in range(0, W, self.patch_size):
                    if y + self.patch_size <= H and x + self.patch_size <= W:
                        patch = img[:, y:y + self.patch_size, x:x + self.patch_size]
                        patches.append(patch.cpu())

        return patches

    def compute_patch_statistics(self, patches: List[torch.Tensor]) -> np.ndarray:
        """
        Compute statistics (mean intensity) for each patch
        Args:
            patches: List of patch tensors
        Returns:
            Numpy array of statistics
        """
        stats = []
        for patch in patches:
            # Compute mean pixel intensity across all channels
            mean_intensity = patch.mean().item()
            stats.append(mean_intensity)

        return np.array(stats)

    def create_bias_intervals(self, stats: np.ndarray, num_intervals: int = 64) -> List[Tuple[float, float]]:
        """
        Create intervals for targeting specific patch statistics
        Args:
            stats: Array of patch statistics
            num_intervals: Number of intervals to create
        Returns:
            List of (low, high) tuples representing intervals
        """
        min_val = np.min(stats)
        max_val = np.max(stats)

        # Create evenly spaced intervals
        bins = np.linspace(min_val, max_val, num_intervals + 1)
        intervals = []

        for i in range(num_intervals):
            intervals.append((bins[i], bins[i + 1]))

        return intervals

    def filter_patches_by_stats(self, patches: List[torch.Tensor],
                                target_interval: Tuple[float, float]) -> List[torch.Tensor]:
        """
        Filter patches that fall within a specific statistical interval
        Args:
            patches: List of patches
            target_interval: (low, high) tuple
        Returns:
            Filtered list of patches
        """
        filtered_patches = []
        low, high = target_interval

        for patch in patches:
            mean_intensity = patch.mean().item()
            if low <= mean_intensity < high:
                filtered_patches.append(patch)

        return filtered_patches

    def prepare_cifar100_data(self, subset_size: Optional[int] = None) -> torch.utils.data.DataLoader:
        """
        Prepare CIFAR-100 dataset for processing
        Args:
            subset_size: Optional size of subset to use
        Returns:
            DataLoader for the dataset
        """
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])

        dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)

        if subset_size:
            indices = torch.randperm(len(dataset))[:subset_size]
            dataset = torch.utils.data.Subset(dataset, indices)

        return torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2)
