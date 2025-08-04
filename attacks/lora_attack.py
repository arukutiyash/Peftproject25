import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional #lora_attack.py
from attacks.attack_base import BaseGradientInversionAttack


class LoRAGradientInversionAttack(BaseGradientInversionAttack):
    """Gradient inversion attack specifically for LoRA PEFT"""

    def __init__(self, model: nn.Module, device: str = 'cuda',
                 lora_rank: int = 16, lora_alpha: float = 16, **kwargs):
        super().__init__(model, device, **kwargs)
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / lora_rank

        # LoRA-specific configuration
        self.attack_config.update({
            'lora_rank': lora_rank,
            'lora_alpha': lora_alpha,
            'scaling': self.scaling,
            'recovery_method': 'low_rank_decomposition',
            'svd_threshold': 0.01
        })

    def setup_malicious_parameters(self):
        """Setup malicious LoRA parameters for gradient inversion"""
        print("Setting up malicious LoRA parameters...")

        with torch.no_grad():
            for name, module in self.model.named_modules():
                if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                    # Initialize LoRA matrices for attack

                    # Matrix A: Create patterns that will reveal input structure
                    lora_A = module.lora_A  # Shape: [rank, in_features]
                    lora_B = module.lora_B  # Shape: [out_features, rank]

                    # Set A matrix to extract specific input patterns
                    nn.init.normal_(lora_A, mean=0, std=0.02)

                    # Create structured patterns in A matrix
                    for r in range(min(self.lora_rank, lora_A.shape[0])):
                        # Each row of A focuses on different input aspects
                        pattern_idx = r % (3 * self.patch_size * self.patch_size)
                        if pattern_idx < lora_A.shape[1]:
                            lora_A[r, pattern_idx] = 1.0  # Strong signal for specific positions

                    # Set B matrix for identity-like transformation when combined with A
                    nn.init.zeros_(lora_B)
                    min_dim = min(lora_B.shape[0], lora_B.shape[1], lora_A.shape[0])

                    # Create structured transformation in B
                    for i in range(min_dim):
                        lora_B[i % lora_B.shape[0], i] = 0.1  # Weak identity mapping

    def extract_target_gradients(self, data_batch: torch.Tensor, labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract LoRA gradients from model"""
        self.model.train()
        self.model.zero_grad()

        # Forward pass
        outputs, _ = self.model(data_batch)
        loss = self.compute_loss(outputs, labels)

        # Backward pass
        loss.backward()

        # Extract LoRA gradients
        lora_gradients = {}
        lora_weights = {}

        for name, param in self.model.named_parameters():
            if 'lora_' in name and param.grad is not None:
                lora_gradients[name] = param.grad.clone().detach()
                lora_weights[name] = param.data.clone().detach()

        return {'gradients': lora_gradients, 'weights': lora_weights}

    def reconstruct_patches(self, gradients: Dict[str, torch.Tensor]) -> List[torch.Tensor]:
        """Reconstruct patches from LoRA gradients using low-rank decomposition"""
        recovered_patches = []

        lora_grads = gradients.get('gradients', {})
        lora_weights = gradients.get('weights', {})

        # Group LoRA A and B gradients
        a_gradients = {k: v for k, v in lora_grads.items() if 'lora_A' in k}
        b_gradients = {k: v for k, v in lora_grads.items() if 'lora_B' in k}

        # Reconstruct using LoRA matrix analysis
        patches_from_decomposition = self._recover_from_lora_decomposition(a_gradients, b_gradients)
        recovered_patches.extend(patches_from_decomposition)

        # Reconstruct using delta weight analysis
        patches_from_delta = self._recover_from_delta_weights(lora_weights)
        recovered_patches.extend(patches_from_delta)

        return recovered_patches[:self.num_patches]

    def _recover_from_lora_decomposition(self, a_grads: Dict[str, torch.Tensor],
                                         b_grads: Dict[str, torch.Tensor]) -> List[torch.Tensor]:
        """Recover patches from LoRA A and B gradient decomposition"""
        patches = []

        # Match A and B gradients from same layers
        for a_name, a_grad in a_grads.items():
            # Find corresponding B gradient
            layer_id = self._extract_lora_layer_id(a_name)
            b_name = a_name.replace('lora_A', 'lora_B')

            if b_name in b_grads:
                b_grad = b_grads[b_name]

                # Reconstruct using LoRA formula: ΔW = B @ A
                # Gradient of ΔW gives us information about input patterns
                delta_w_grad = self._compute_delta_weight_gradient(a_grad, b_grad, layer_id)

                if delta_w_grad is not None:
                    # Decompose delta weight gradient to extract patches
                    layer_patches = self._extract_patches_from_delta_grad(delta_w_grad)
                    patches.extend(layer_patches)

        return patches

    def _compute_delta_weight_gradient(self, a_grad: torch.Tensor, b_grad: torch.Tensor,
                                       layer_id: str) -> Optional[torch.Tensor]:
        """Compute gradient of delta weight ΔW = B @ A"""
        try:
            # Get corresponding A and B weights (not gradients)
            # This is a simplification - in practice, you'd need to track the actual weights
            # For now, use gradients to approximate the delta weight pattern

            # a_grad: [rank, in_features]
            # b_grad: [out_features, rank]

            if a_grad.shape[0] == b_grad.shape[1]:  # rank dimensions match
                # Approximate delta weight using outer product of gradients
                delta_w_grad = torch.outer(b_grad.mean(dim=0), a_grad.mean(dim=0))
                return delta_w_grad * self.scaling

            return None

        except Exception as e:
            print(f"Error computing delta weight gradient for {layer_id}: {e}")
            return None

    def _extract_patches_from_delta_grad(self, delta_grad: torch.Tensor) -> List[torch.Tensor]:
        """Extract patches from delta weight gradient"""
        patches = []

        try:
            # Apply SVD to decompose the delta gradient
            U, S, V = torch.svd(delta_grad)

            # Use top singular components
            significant_components = S > (S.max() * self.attack_config['svd_threshold'])
            num_components = significant_components.sum().item()

            for i in range(min(num_components, 4)):  # Limit to top 4 components
                # Reconstruct component
                component = S[i] * torch.outer(U[:, i], V[:, i])

                # Convert component to patch
                patch = self._matrix_to_patch(component)
                if patch is not None:
                    patches.append(patch)

            # Also try direct conversion of delta gradient
            direct_patch = self._matrix_to_patch(delta_grad)
            if direct_patch is not None:
                patches.append(direct_patch)

        except Exception as e:
            print(f"Error extracting patches from delta gradient: {e}")

        return patches

    def _recover_from_delta_weights(self, lora_weights: Dict[str, torch.Tensor]) -> List[torch.Tensor]:
        """Recover patches from LoRA delta weights ΔW = BA"""
        patches = []

        # Group A and B weights
        a_weights = {k: v for k, v in lora_weights.items() if 'lora_A' in k}
        b_weights = {k: v for k, v in lora_weights.items() if 'lora_B' in k}

        for a_name, a_weight in a_weights.items():
            b_name = a_name.replace('lora_A', 'lora_B')

            if b_name in b_weights:
                b_weight = b_weights[b_name]

                # Compute actual delta weight: ΔW = B @ A
                try:
                    delta_w = torch.matmul(b_weight, a_weight) * self.scaling

                    # Extract patches from delta weight matrix
                    delta_patches = self._extract_patches_from_delta_weight(delta_w, a_name)
                    patches.extend(delta_patches)

                except Exception as e:
                    print(f"Error computing delta weight for {a_name}: {e}")

        return patches

    def _extract_patches_from_delta_weight(self, delta_w: torch.Tensor, layer_name: str) -> List[torch.Tensor]:
        """Extract patches from delta weight matrix"""
        patches = []

        try:
            # Different strategies based on layer type
            if 'qkv' in layer_name:
                # QKV projection - contains query, key, value info
                patches.extend(self._extract_qkv_patches(delta_w))
            elif 'proj' in layer_name or 'fc' in layer_name:
                # Standard linear projection
                patches.extend(self._extract_linear_patches(delta_w))

        except Exception as e:
            print(f"Error extracting patches from delta weight {layer_name}: {e}")

        return patches

    def _extract_qkv_patches(self, delta_w: torch.Tensor) -> List[torch.Tensor]:
        """Extract patches from QKV delta weight matrix"""
        patches = []

        try:
            # QKV matrix is typically [3*embed_dim, embed_dim]
            # Split into Q, K, V components
            out_dim, in_dim = delta_w.shape

            if out_dim % 3 == 0:
                qkv_dim = out_dim // 3

                q_weights = delta_w[:qkv_dim, :]
                k_weights = delta_w[qkv_dim:2 * qkv_dim, :]
                v_weights = delta_w[2 * qkv_dim:, :]

                # Extract patches from each component
                for component_name, component_weights in [('Q', q_weights), ('K', k_weights), ('V', v_weights)]:
                    component_patches = self._matrix_to_patches(component_weights)
                    patches.extend(component_patches[:2])  # Limit patches per component

        except Exception as e:
            print(f"Error extracting QKV patches: {e}")

        return patches

    def _extract_linear_patches(self, delta_w: torch.Tensor) -> List[torch.Tensor]:
        """Extract patches from linear layer delta weight matrix"""
        patches = []

        try:
            # Use SVD decomposition for linear layers
            U, S, V = torch.svd(delta_w)

            # Extract patches from top singular vectors
            significant_indices = S > (S.max() * 0.1)  # 10% threshold
            num_significant = significant_indices.sum().item()

            for i in range(min(num_significant, 3)):
                # Create patch from singular vector combination
                patch_matrix = S[i] * torch.outer(U[:, i], V[:, i])
                patch = self._matrix_to_patch(patch_matrix)
                if patch is not None:
                    patches.append(patch)

        except Exception as e:
            print(f"Error extracting linear patches: {e}")

        return patches

    def _matrix_to_patches(self, matrix: torch.Tensor) -> List[torch.Tensor]:
        """Convert matrix to multiple patches"""
        patches = []

        try:
            # Flatten matrix and create multiple patches
            flattened = matrix.flatten()
            patch_elements = 3 * self.patch_size * self.patch_size

            num_possible_patches = len(flattened) // patch_elements

            for i in range(min(num_possible_patches, 2)):  # Max 2 patches per matrix
                start_idx = i * patch_elements
                end_idx = start_idx + patch_elements

                patch_data = flattened[start_idx:end_idx]
                patch = patch_data.view(3, self.patch_size, self.patch_size)

                # Normalize patch
                patch = torch.tanh(patch) * 0.5 + 0.5
                patches.append(patch)

        except Exception as e:
            print(f"Error converting matrix to patches: {e}")

        return patches

    def _matrix_to_patch(self, matrix: torch.Tensor) -> Optional[torch.Tensor]:
        """Convert 2D matrix to single image patch"""
        try:
            flattened = matrix.flatten()
            patch_elements = 3 * self.patch_size * self.patch_size

            if len(flattened) >= patch_elements:
                patch_data = flattened[:patch_elements]
            else:
                # Pad with zeros if matrix too small
                patch_data = torch.cat([flattened,
                                        torch.zeros(patch_elements - len(flattened),
                                                    device=flattened.device)])

            patch = patch_data.view(3, self.patch_size, self.patch_size)

            # Normalize to [0, 1]
            patch = torch.tanh(patch) * 0.5 + 0.5

            return patch

        except Exception as e:
            print(f"Error converting matrix to patch: {e}")
            return None

    def _extract_lora_layer_id(self, param_name: str) -> str:
        """Extract layer identifier from LoRA parameter name"""
        if 'blocks' in param_name:
            parts = param_name.split('.')
            for i, part in enumerate(parts):
                if part == 'blocks' and i + 1 < len(parts):
                    return f"block_{parts[i + 1]}"
        return param_name

    def analyze_lora_rank_impact(self, data_batch: torch.Tensor) -> Dict[str, float]:
        """Analyze impact of different rank dimensions on reconstruction"""
        impact_analysis = {}

        # Original output
        original_output, _ = self.model(data_batch)

        # Test impact of zeroing different rank dimensions
        for name, module in self.model.named_modules():
            if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                for rank_idx in range(min(self.lora_rank, 4)):  # Test first 4 ranks
                    # Zero out specific rank dimension
                    original_a = module.lora_A.data[rank_idx].clone()
                    original_b = module.lora_B.data[:, rank_idx].clone()

                    module.lora_A.data[rank_idx].zero_()
                    module.lora_B.data[:, rank_idx].zero_()

                    # Forward pass
                    modified_output, _ = self.model(data_batch)
                    impact = torch.norm(modified_output - original_output).item()

                    impact_analysis[f"{name}_rank_{rank_idx}"] = impact

                    # Restore original values
                    module.lora_A.data[rank_idx] = original_a
                    module.lora_B.data[:, rank_idx] = original_b

        return impact_analysis
