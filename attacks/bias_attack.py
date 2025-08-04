import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional #bias_attack.py
from attacks.attack_base import BaseGradientInversionAttack


class BiasGradientInversionAttack(BaseGradientInversionAttack):
    """Gradient inversion attack specifically for Bias-tuning PEFT"""

    def __init__(self, model: nn.Module, device: str = 'cuda', **kwargs):
        super().__init__(model, device, **kwargs)

        # Bias-specific configuration
        self.attack_config.update({
            'recovery_method': 'direct_bias_inversion',
            'activation_threshold': 0.05,
            'bias_scaling_factor': 1.0
        })

    def setup_malicious_parameters(self):
        """Setup malicious bias parameters for gradient inversion"""
        print("Setting up malicious bias parameters...")

        with torch.no_grad():
            for i, block in enumerate(self.model.blocks):
                # Set biases to position-specific values for targeted recovery
                pos_encoding = self.model.pos_embed[0, i % self.model.pos_embed.size(1)] if hasattr(self.model,
                                                                                                    'pos_embed') else None

                # Configure attention biases
                if hasattr(block, 'attn_qkv_bias'):
                    if pos_encoding is not None:
                        bias_dim = block.attn_qkv_bias.bias.shape[0]
                        pos_dim = min(bias_dim, len(pos_encoding))
                        block.attn_qkv_bias.bias[:pos_dim] = pos_encoding[:pos_dim] * 0.1

                if hasattr(block, 'attn_proj_bias'):
                    if pos_encoding is not None:
                        bias_dim = block.attn_proj_bias.bias.shape[0]
                        pos_dim = min(bias_dim, len(pos_encoding))
                        block.attn_proj_bias.bias[:pos_dim] = pos_encoding[:pos_dim] * 0.1

                # Configure MLP biases
                if hasattr(block, 'mlp_fc1_bias'):
                    if pos_encoding is not None:
                        bias_dim = block.mlp_fc1_bias.bias.shape[0]
                        pos_dim = min(bias_dim, len(pos_encoding))
                        block.mlp_fc1_bias.bias[:pos_dim] = pos_encoding[:pos_dim] * 0.1

                if hasattr(block, 'mlp_fc2_bias'):
                    if pos_encoding is not None:
                        bias_dim = block.mlp_fc2_bias.bias.shape[0]
                        pos_dim = min(bias_dim, len(pos_encoding))
                        block.mlp_fc2_bias.bias[:pos_dim] = pos_encoding[:pos_dim] * 0.1

                # Configure layer norm biases
                if hasattr(block, 'norm1_bias'):
                    if pos_encoding is not None:
                        block.norm1_bias.bias[:] = pos_encoding * 0.05

                if hasattr(block, 'norm2_bias'):
                    if pos_encoding is not None:
                        block.norm2_bias.bias[:] = pos_encoding * 0.05

    def extract_target_gradients(self, data_batch: torch.Tensor, labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract bias gradients from model"""
        self.model.train()
        self.model.zero_grad()

        # Store intermediate activations for analysis
        activations = {}

        def save_activation(name):
            def hook(module, input, output):
                activations[name] = output.clone().detach() if isinstance(output, torch.Tensor) else output[
                    0].clone().detach()

            return hook

        # Register hooks to capture activations
        hooks = []
        for i, block in enumerate(self.model.blocks):
            hook = block.register_forward_hook(save_activation(f'block_{i}'))
            hooks.append(hook)

        # Forward pass
        outputs, _ = self.model(data_batch)
        loss = self.compute_loss(outputs, labels)

        # Backward pass
        loss.backward()

        # Remove hooks
        for hook in hooks:
            hook.remove()

        # Extract bias gradients
        bias_gradients = {}
        for name, param in self.model.named_parameters():
            if any(bias_name in name for bias_name in ['attn_qkv_bias', 'attn_proj_bias',
                                                       'mlp_fc1_bias', 'mlp_fc2_bias',
                                                       'norm1_bias', 'norm2_bias']) and param.grad is not None:
                bias_gradients[name] = param.grad.clone().detach()

        # Include activations for patch reconstruction
        return {**bias_gradients, 'activations': activations}

    def reconstruct_patches(self, gradients: Dict[str, torch.Tensor]) -> List[torch.Tensor]:
        """Reconstruct patches from bias gradients"""
        recovered_patches = []

        # Separate bias gradients from activations
        bias_grads = {k: v for k, v in gradients.items() if k != 'activations'}
        activations = gradients.get('activations', {})

        # Method 1: Direct bias gradient inversion
        patches_from_bias = self._recover_from_bias_gradients(bias_grads)
        recovered_patches.extend(patches_from_bias)

        # Method 2: Activation pattern analysis
        patches_from_activations = self._recover_from_activation_patterns(activations, bias_grads)
        recovered_patches.extend(patches_from_activations)

        return recovered_patches[:self.num_patches]

    def _recover_from_bias_gradients(self, bias_grads: Dict[str, torch.Tensor]) -> List[torch.Tensor]:
        """Recover patches directly from bias gradients"""
        patches = []

        for name, grad in bias_grads.items():
            # Bias gradients directly relate to input activations
            if grad.dim() == 1:  # 1D bias vector
                # Convert bias gradient to patches
                patch_candidates = self._bias_gradient_to_patches(grad, name)
                patches.extend(patch_candidates)
            elif grad.dim() == 2:  # 2D bias (if batch-dependent)
                # Process each batch item
                for batch_idx in range(grad.shape[0]):
                    batch_grad = grad[batch_idx]
                    patch_candidates = self._bias_gradient_to_patches(batch_grad, name)
                    patches.extend(patch_candidates)

        return patches

    def _bias_gradient_to_patches(self, bias_grad: torch.Tensor, bias_name: str) -> List[torch.Tensor]:
        """Convert bias gradient vector to image patches"""
        patches = []

        try:
            bias_dim = bias_grad.shape[0]
            patch_elements = 3 * self.patch_size * self.patch_size

            # Different strategies based on bias type
            if 'attn' in bias_name:
                # Attention biases often contain spatial information
                num_patches_possible = bias_dim // patch_elements

                for p in range(min(num_patches_possible, 4)):
                    start_idx = p * patch_elements
                    end_idx = start_idx + patch_elements

                    if end_idx <= bias_dim:
                        patch_data = bias_grad[start_idx:end_idx]
                        patch = patch_data.view(3, self.patch_size, self.patch_size)

                        # Normalize to valid image range
                        patch = self._normalize_patch(patch)
                        patches.append(patch)

            elif 'mlp' in bias_name:
                # MLP biases may contain feature-level information
                if bias_dim >= patch_elements:
                    # Use first portion as patch
                    patch_data = bias_grad[:patch_elements]
                    patch = patch_data.view(3, self.patch_size, self.patch_size)
                    patch = self._normalize_patch(patch)
                    patches.append(patch)

                    # Use gradient magnitude pattern as additional patch
                    if bias_dim >= 2 * patch_elements:
                        patch_data2 = bias_grad[patch_elements:2 * patch_elements]
                        patch2 = patch_data2.view(3, self.patch_size, self.patch_size)
                        patch2 = self._normalize_patch(patch2)
                        patches.append(patch2)

            elif 'norm' in bias_name:
                # Layer norm biases contain overall activation patterns
                # Repeat pattern to create patch
                if bias_dim == self.model.embed_dim:
                    # Embed_dim should match expected size
                    repeated_pattern = bias_grad.repeat((patch_elements + bias_dim - 1) // bias_dim)
                    patch_data = repeated_pattern[:patch_elements]
                    patch = patch_data.view(3, self.patch_size, self.patch_size)
                    patch = self._normalize_patch(patch)
                    patches.append(patch)

        except Exception as e:
            print(f"Error converting bias gradient {bias_name} to patch: {e}")

        return patches

    def _recover_from_activation_patterns(self, activations: Dict[str, torch.Tensor],
                                          bias_grads: Dict[str, torch.Tensor]) -> List[torch.Tensor]:
        """Recover patches from activation patterns combined with bias information"""
        patches = []

        for activation_name, activation in activations.items():
            try:
                if activation.dim() >= 3:  # [batch, seq_len, embed_dim] or similar
                    batch_size = activation.shape[0]

                    # Focus on patch tokens (excluding CLS token)
                    if activation.shape[1] > 1:  # Has more than just CLS token
                        patch_activations = activation[:, 1:, :]  # Skip CLS token

                        # Combine with corresponding bias gradients
                        corresponding_bias = self._find_corresponding_bias_grad(activation_name, bias_grads)

                        if corresponding_bias is not None:
                            # Use bias gradient as a mask/filter for activations
                            filtered_activations = self._apply_bias_filter(patch_activations, corresponding_bias)

                            # Convert filtered activations to patches
                            activation_patches = self._activations_to_patches(filtered_activations)
                            patches.extend(activation_patches)

            except Exception as e:
                print(f"Error processing activations {activation_name}: {e}")

        return patches

    def _find_corresponding_bias_grad(self, activation_name: str, bias_grads: Dict[str, torch.Tensor]) -> Optional[
        torch.Tensor]:
        """Find bias gradient corresponding to activation"""
        # Extract layer number from activation name
        if 'block_' in activation_name:
            layer_num = activation_name.split('block_')[1].split('_')[0]

            # Look for bias gradients from the same layer
            for bias_name, bias_grad in bias_grads.items():
                if layer_num in bias_name:
                    return bias_grad

        return None

    def _apply_bias_filter(self, activations: torch.Tensor, bias_grad: torch.Tensor) -> torch.Tensor:
        """Apply bias gradient as filter to activations"""
        try:
            # activations: [batch, num_patches, embed_dim]
            # bias_grad: [bias_dim]

            batch_size, num_patches, embed_dim = activations.shape
            bias_dim = bias_grad.shape[0]

            # Match dimensions
            if bias_dim == embed_dim:
                # Direct element-wise filtering
                filtered = activations * bias_grad.unsqueeze(0).unsqueeze(0)
            elif bias_dim > embed_dim:
                # Use first embed_dim elements of bias
                bias_subset = bias_grad[:embed_dim]
                filtered = activations * bias_subset.unsqueeze(0).unsqueeze(0)
            else:
                # Repeat bias to match embed_dim
                bias_repeated = bias_grad.repeat((embed_dim + bias_dim - 1) // bias_dim)[:embed_dim]
                filtered = activations * bias_repeated.unsqueeze(0).unsqueeze(0)

            return filtered

        except Exception as e:
            print(f"Error applying bias filter: {e}")
            return activations

    def _activations_to_patches(self, activations: torch.Tensor) -> List[torch.Tensor]:
        """Convert filtered activations to image patches"""
        patches = []

        try:
            # activations: [batch, num_patches, embed_dim]
            batch_size, num_patches, embed_dim = activations.shape
            patch_elements = 3 * self.patch_size * self.patch_size

            for batch_idx in range(min(batch_size, 1)):  # Process first batch item
                for patch_idx in range(min(num_patches, 8)):  # Limit number of patches
                    patch_embedding = activations[batch_idx, patch_idx, :]  # [embed_dim]

                    # Convert embedding to patch format
                    if embed_dim >= patch_elements:
                        patch_data = patch_embedding[:patch_elements]
                        patch = patch_data.view(3, self.patch_size, self.patch_size)
                        patch = self._normalize_patch(patch)
                        patches.append(patch)

        except Exception as e:
            print(f"Error converting activations to patches: {e}")

        return patches

    def _normalize_patch(self, patch: torch.Tensor) -> torch.Tensor:
        """Normalize patch to valid image range [0, 1]"""
        # Apply tanh activation and scale to [0, 1]
        normalized = torch.tanh(patch) * 0.5 + 0.5

        # Ensure values are in valid range
        normalized = torch.clamp(normalized, 0.0, 1.0)

        return normalized

    def analyze_bias_sensitivity(self, data_batch: torch.Tensor) -> Dict[str, float]:
        """Analyze sensitivity of different bias components"""
        sensitivity_analysis = {}

        # Original output
        original_output, _ = self.model(data_batch)

        # Test each bias component
        bias_modules = []
        for name, module in self.model.named_modules():
            if any(bias_name in name for bias_name in ['attn_qkv_bias', 'attn_proj_bias',
                                                       'mlp_fc1_bias', 'mlp_fc2_bias',
                                                       'norm1_bias', 'norm2_bias']):
                bias_modules.append((name, module))

        for name, module in bias_modules:
            # Perturb bias
            if hasattr(module, 'bias'):
                original_bias = module.bias.data.clone()
                perturbation = torch.randn_like(original_bias) * 0.01
                module.bias.data += perturbation

                # Measure output change
                perturbed_output, _ = self.model(data_batch)
                sensitivity = torch.norm(perturbed_output - original_output).item()
                sensitivity_analysis[name] = sensitivity

                # Restore original bias
                module.bias.data = original_bias

        return sensitivity_analysis
