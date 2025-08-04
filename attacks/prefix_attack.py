import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional #prefix_attack.py
from attacks.attack_base import BaseGradientInversionAttack


class PrefixGradientInversionAttack(BaseGradientInversionAttack):
    """Gradient inversion attack specifically for Prefix-tuning PEFT"""

    def __init__(self, model: nn.Module, device: str = 'cuda',
                 prefix_length: int = 10, **kwargs):
        super().__init__(model, device, **kwargs)
        self.prefix_length = prefix_length

        # Prefix-specific configuration
        self.attack_config.update({
            'prefix_length': prefix_length,
            'recovery_method': 'attention_analysis',
            'attention_threshold': 0.1
        })

    def setup_malicious_parameters(self):
        """Setup malicious prefix parameters for gradient inversion"""
        print("Setting up malicious prefix parameters...")

        with torch.no_grad():
            for i, block in enumerate(self.model.blocks):
                if hasattr(block, 'attn') and hasattr(block.attn, 'prefix_encoder'):
                    prefix_encoder = block.attn.prefix_encoder

                    # Set prefix embeddings to position-specific patterns
                    pos_encodings = self.model.pos_embed[0, :self.prefix_length] if hasattr(self.model,
                                                                                            'pos_embed') else None

                    if pos_encodings is not None:
                        prefix_encoder.prefix_embeddings.data = pos_encodings.clone()
                    else:
                        # Create synthetic position encodings
                        for j in range(self.prefix_length):
                            encoding = torch.zeros(self.model.embed_dim)
                            # Simple sinusoidal encoding
                            for d in range(0, self.model.embed_dim, 2):
                                encoding[d] = np.sin(j / (10000 ** (2 * d / self.model.embed_dim)))
                                if d + 1 < self.model.embed_dim:
                                    encoding[d + 1] = np.cos(j / (10000 ** (2 * (d + 1) / self.model.embed_dim)))
                            prefix_encoder.prefix_embeddings.data[j] = encoding

                    # Set MLP layers to preserve information
                    for layer in prefix_encoder.prefix_mlp:
                        if isinstance(layer, nn.Linear):
                            # Initialize as near-identity transformation
                            nn.init.eye_(layer.weight[:min(layer.in_features, layer.out_features),
                                         :min(layer.in_features, layer.out_features)])
                            if layer.bias is not None:
                                nn.init.zeros_(layer.bias)

    def extract_target_gradients(self, data_batch: torch.Tensor, labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract prefix gradients from model"""
        self.model.train()
        self.model.zero_grad()

        # Forward pass
        outputs, attention_weights = self.model(data_batch)
        loss = self.compute_loss(outputs, labels)

        # Backward pass
        loss.backward()

        # Extract prefix gradients
        prefix_gradients = {}
        attention_patterns = {}

        for name, param in self.model.named_parameters():
            if 'prefix' in name.lower() and param.grad is not None:
                prefix_gradients[name] = param.grad.clone().detach()

        # Also store attention patterns for analysis
        for i, attn_weight in enumerate(attention_weights):
            attention_patterns[f'layer_{i}_attention'] = attn_weight.clone().detach()

        return {**prefix_gradients, **attention_patterns}

    def reconstruct_patches(self, gradients: Dict[str, torch.Tensor]) -> List[torch.Tensor]:
        """Reconstruct patches from prefix gradients using attention patterns"""
        recovered_patches = []

        # Separate prefix gradients from attention patterns
        prefix_grads = {k: v for k, v in gradients.items() if 'prefix' in k.lower()}
        attention_patterns = {k: v for k, v in gradients.items() if 'attention' in k.lower()}

        # Reconstruct patches using prefix embeddings and attention
        recovered_patches.extend(self._recover_from_prefix_gradients(prefix_grads))
        recovered_patches.extend(self._recover_from_attention_patterns(attention_patterns))

        return recovered_patches[:self.num_patches]

    def _recover_from_prefix_gradients(self, prefix_grads: Dict[str, torch.Tensor]) -> List[torch.Tensor]:
        """Recover patches from prefix embedding gradients"""
        patches = []

        for name, grad in prefix_grads.items():
            if 'prefix_embeddings' in name:
                # Each prefix embedding gradient potentially contains patch information
                for i in range(min(grad.shape[0], self.prefix_length)):
                    embedding_grad = grad[i]  # Shape: [embed_dim]

                    # Convert embedding gradient to patch
                    patch = self._embedding_to_patch(embedding_grad)
                    if patch is not None:
                        patches.append(patch)

            elif 'prefix_mlp' in name and 'weight' in name:
                # MLP weight gradients contain mixed patch information
                weight_grad = grad

                # Decompose weight gradients to recover patches
                patches_from_weights = self._decompose_weight_gradients(weight_grad)
                patches.extend(patches_from_weights)

        return patches

    def _recover_from_attention_patterns(self, attention_patterns: Dict[str, torch.Tensor]) -> List[torch.Tensor]:
        """Recover patches from attention patterns"""
        patches = []

        for name, attn_weights in attention_patterns.items():
            # Attention weights shape: [batch, heads, seq_len, seq_len]
            if attn_weights.dim() == 4:
                batch_size, num_heads, seq_len, _ = attn_weights.shape

                # Focus on attention from tokens to prefix
                if seq_len > self.prefix_length:
                    # Extract attention from image patches to prefix tokens
                    patch_to_prefix_attn = attn_weights[:, :, self.prefix_length:, :self.prefix_length]

                    # Analyze strong attention connections
                    strong_connections = (patch_to_prefix_attn > self.attack_config['attention_threshold'])

                    # Convert attention patterns to patches
                    for head in range(min(num_heads, 4)):  # Limit to first 4 heads
                        head_attn = patch_to_prefix_attn[0, head]  # First batch item
                        patch = self._attention_to_patch(head_attn)
                        if patch is not None:
                            patches.append(patch)

        return patches

    def _embedding_to_patch(self, embedding: torch.Tensor) -> Optional[torch.Tensor]:
        """Convert embedding gradient to image patch"""
        try:
            # Reshape embedding to patch dimensions if possible
            embed_dim = embedding.shape[0]
            patch_elements = 3 * self.patch_size * self.patch_size

            if embed_dim >= patch_elements:
                # Take first patch_elements and reshape
                patch_data = embedding[:patch_elements]
                patch = patch_data.view(3, self.patch_size, self.patch_size)

                # Normalize to valid image range
                patch = torch.tanh(patch) * 0.5 + 0.5  # Map to [0, 1]
                return patch
            else:
                # Pad with zeros if embedding is too small
                padded = torch.cat([embedding, torch.zeros(patch_elements - embed_dim, device=embedding.device)])
                patch = padded.view(3, self.patch_size, self.patch_size)
                patch = torch.tanh(patch) * 0.5 + 0.5
                return patch

        except Exception as e:
            print(f"Error converting embedding to patch: {e}")
            return None

    def _decompose_weight_gradients(self, weight_grad: torch.Tensor) -> List[torch.Tensor]:
        """Decompose MLP weight gradients to recover patch information"""
        patches = []

        try:
            # Apply SVD decomposition to extract principal components
            if weight_grad.dim() >= 2:
                U, S, V = torch.svd(weight_grad)

                # Use top singular vectors as patch candidates
                num_components = min(4, len(S))
                for i in range(num_components):
                    # Combine U and V vectors weighted by singular values
                    component = S[i] * torch.outer(U[:, i], V[:, i])

                    # Convert component to patch format
                    patch = self._matrix_to_patch(component)
                    if patch is not None:
                        patches.append(patch)

        except Exception as e:
            print(f"Error in weight gradient decomposition: {e}")

        return patches

    def _attention_to_patch(self, attention_weights: torch.Tensor) -> Optional[torch.Tensor]:
        """Convert attention pattern to image patch"""
        try:
            # attention_weights shape: [num_patches, prefix_length]
            num_image_patches, prefix_len = attention_weights.shape

            # Create patch based on attention distribution
            patch_size_sq = self.patch_size * self.patch_size

            if num_image_patches >= patch_size_sq:
                # Take spatial pattern from attention
                spatial_pattern = attention_weights[:patch_size_sq, :].mean(dim=1)  # Average over prefix

                # Reshape to spatial dimensions
                spatial_2d = spatial_pattern.view(self.patch_size, self.patch_size)

                # Expand to RGB channels
                patch = spatial_2d.unsqueeze(0).repeat(3, 1, 1)

                # Normalize
                patch = (patch - patch.min()) / (patch.max() - patch.min() + 1e-8)
                return patch
            else:
                # Create synthetic patch from available attention
                patch = torch.zeros(3, self.patch_size, self.patch_size, device=attention_weights.device)
                available_vals = attention_weights.flatten()[:patch_size_sq * 3]
                patch.flatten()[:len(available_vals)] = available_vals
                return patch

        except Exception as e:
            print(f"Error converting attention to patch: {e}")
            return None

    def _matrix_to_patch(self, matrix: torch.Tensor) -> Optional[torch.Tensor]:
        """Convert 2D matrix to image patch"""
        try:
            # Flatten and take required elements
            flattened = matrix.flatten()
            patch_elements = 3 * self.patch_size * self.patch_size

            if len(flattened) >= patch_elements:
                patch_data = flattened[:patch_elements]
            else:
                # Pad with zeros
                patch_data = torch.cat(
                    [flattened, torch.zeros(patch_elements - len(flattened), device=flattened.device)])

            # Reshape and normalize
            patch = patch_data.view(3, self.patch_size, self.patch_size)
            patch = torch.tanh(patch) * 0.5 + 0.5  # Normalize to [0, 1]

            return patch

        except Exception as e:
            print(f"Error converting matrix to patch: {e}")
            return None

    def analyze_prefix_impact(self, data_batch: torch.Tensor) -> Dict[str, float]:
        """Analyze the impact of different prefix tokens on reconstruction"""
        impact_analysis = {}

        # Original output
        original_output, original_attention = self.model(data_batch)

        # Test impact of each prefix position
        for prefix_pos in range(self.prefix_length):
            # Temporarily zero out specific prefix position
            for block in self.model.blocks:
                if hasattr(block, 'attn') and hasattr(block.attn, 'prefix_encoder'):
                    original_embedding = block.attn.prefix_encoder.prefix_embeddings.data[prefix_pos].clone()
                    block.attn.prefix_encoder.prefix_embeddings.data[prefix_pos].zero_()

            # Forward pass with modified prefix
            modified_output, modified_attention = self.model(data_batch)

            # Measure impact
            output_diff = torch.norm(modified_output - original_output).item()
            attention_diff = 0.0
            for orig_attn, mod_attn in zip(original_attention, modified_attention):
                attention_diff += torch.norm(orig_attn - mod_attn).item()

            impact_analysis[f'prefix_pos_{prefix_pos}'] = {
                'output_impact': output_diff,
                'attention_impact': attention_diff / len(original_attention)
            }

            # Restore original embedding
            for block in self.model.blocks:
                if hasattr(block, 'attn') and hasattr(block.attn, 'prefix_encoder'):
                    block.attn.prefix_encoder.prefix_embeddings.data[prefix_pos] = original_embedding

        return impact_analysis
