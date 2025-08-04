import torch
import numpy as np
from typing import List, Dict, Optional
import matplotlib.pyplot as plt #patch_recovery.py


class PatchRecoverer:
    """Enhanced patch recovery from gradients using multiple methods"""

    def __init__(self, patch_size: int = 4, device: str = 'cpu'):
        self.patch_size = patch_size
        self.device = device

    def recover_from_adapter_gradients(self, gradients: Dict[str, torch.Tensor]) -> List[torch.Tensor]:
        """
        Recover patches from adapter gradients using difference method
        Args:
            gradients: Dictionary containing weight and bias gradients
        Returns:
            List of recovered patches
        """
        recovered_patches = []

        # Extract weight and bias gradients
        weight_grads = []
        bias_grads = []

        for name, grad in gradients.items():
            if 'weight' in name and 'down_proj' in name:
                weight_grads.append(grad)
            elif 'bias' in name and 'down_proj' in name:
                bias_grads.append(grad)

        # Apply recovery formula: (W_k - W_k+1) / (b_k - b_k+1)
        for w_grad, b_grad in zip(weight_grads, bias_grads):
            patches = self._apply_difference_formula(w_grad, b_grad)
            recovered_patches.extend(patches)

        return recovered_patches

    def _apply_difference_formula(self, weight_grad: torch.Tensor,
                                  bias_grad: torch.Tensor) -> List[torch.Tensor]:
        """Apply the core PEFTLeak recovery formula"""
        patches = []

        try:
            # Ensure gradients have compatible dimensions
            min_dim = min(weight_grad.shape[0] - 1, bias_grad.shape[0] - 1)

            for k in range(min_dim):
                # Compute differences
                w_diff = weight_grad[k] - weight_grad[k + 1]
                b_diff = bias_grad[k] - bias_grad[k + 1]

                # Avoid division by zero
                if torch.abs(b_diff).max() > 1e-8:
                    recovered_vector = w_diff / b_diff

                    # Reshape to patch format if possible
                    patch_elements = 3 * self.patch_size * self.patch_size
                    if recovered_vector.numel() >= patch_elements:
                        patch_data = recovered_vector.flatten()[:patch_elements]
                        patch = patch_data.view(3, self.patch_size, self.patch_size)

                        # Normalize to valid image range
                        patch = torch.clamp(torch.tanh(patch) * 0.5 + 0.5, 0, 1)
                        patches.append(patch.cpu())

        except Exception as e:
            print(f"Error in patch recovery: {e}")

        return patches

    def recover_from_prefix_gradients(self, gradients: Dict[str, torch.Tensor]) -> List[torch.Tensor]:
        """
        Recover patches from prefix-tuning gradients
        Args:
            gradients: Dictionary containing prefix gradients
        Returns:
            List of recovered patches
        """
        recovered_patches = []

        for name, grad in gradients.items():
            if 'prefix_embeddings' in name:
                # Convert prefix embeddings to patches
                patches = self._embedding_to_patches(grad)
                recovered_patches.extend(patches)

        return recovered_patches

    def _embedding_to_patches(self, embedding_grad: torch.Tensor) -> List[torch.Tensor]:
        """Convert embedding gradients to patches"""
        patches = []
        patch_elements = 3 * self.patch_size * self.patch_size

        # Process each embedding vector
        for i in range(embedding_grad.shape[0]):
            emb_vec = embedding_grad[i]

            if emb_vec.numel() >= patch_elements:
                patch_data = emb_vec[:patch_elements]
                patch = patch_data.view(3, self.patch_size, self.patch_size)
                patch = torch.clamp(torch.tanh(patch) * 0.5 + 0.5, 0, 1)
                patches.append(patch.cpu())

        return patches

    def visualize_recovery_results(self, original_patches: List[torch.Tensor],
                                   recovered_patches: List[torch.Tensor],
                                   max_display: int = 8):
        """
        Visualize comparison between original and recovered patches
        Args:
            original_patches: List of original patches
            recovered_patches: List of recovered patches
            max_display: Maximum number of patches to display
        """
        num_patches = min(len(original_patches), len(recovered_patches), max_display)

        fig, axes = plt.subplots(2, num_patches, figsize=(num_patches * 2, 4))

        for i in range(num_patches):
            # Original patches
            if len(original_patches) > i:
                orig_patch = original_patches[i].permute(1, 2, 0).numpy()
                axes[0, i].imshow(np.clip(orig_patch, 0, 1))
                axes[0, i].set_title(f'Original {i + 1}')
                axes[0, i].axis('off')

            # Recovered patches
            rec_patch = recovered_patches[i].permute(1, 2, 0).numpy()
            axes[1, i].imshow(np.clip(rec_patch, 0, 1))
            axes[1, i].set_title(f'Recovered {i + 1}')
            axes[1, i].axis('off')

        plt.tight_layout()
        plt.show()
