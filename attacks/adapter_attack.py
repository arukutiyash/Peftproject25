import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
from attacks.attack_base import BaseGradientInversionAttack


class AdapterGradientInversionAttack(BaseGradientInversionAttack):
    """Gradient inversion attack specifically for Adapter-based PEFT"""

    def __init__(self, model: nn.Module, device: str = 'cuda',
                 bottleneck_dim: int = 64, **kwargs):
        super().__init__(model, device, **kwargs)
        self.bottleneck_dim = bottleneck_dim

        # Adapter-specific configuration
        self.attack_config.update({
            'bottleneck_dim': bottleneck_dim,
            'recovery_method': 'gradient_difference',
            'bias_intervals': self._compute_bias_intervals()
        })

    def _compute_bias_intervals(self) -> List[Tuple[float, float]]:
        """Compute bias intervals for patch statistics (simplified version)"""
        # This is a simplified version - in practice, use statistics from public dataset
        intervals = []
        for i in range(self.bottleneck_dim):
            # Create intervals based on expected patch intensity distributions
            lower_bound = -2.0 + (i * 4.0 / self.bottleneck_dim)
            upper_bound = lower_bound + (4.0 / self.bottleneck_dim)
            intervals.append((lower_bound, upper_bound))
        return intervals

    def setup_malicious_parameters(self):
        """Setup malicious adapter parameters for gradient inversion"""
        print("Setting up malicious adapter parameters...")

        with torch.no_grad():
            for i, block in enumerate(self.model.blocks):
                # Configure attention adapter
                if hasattr(block, 'attn_peft') and block.attn_peft is not None:
                    adapter = block.attn_peft

                    # Set down projection to specific patterns
                    nn.init.zeros_(adapter.down_proj.weight)
                    nn.init.zeros_(adapter.down_proj.bias)

                    # Set up projection weights for identity mapping
                    nn.init.zeros_(adapter.up_proj.weight)
                    min_dim = min(adapter.up_proj.weight.shape[0], adapter.down_proj.weight.shape[0])
                    adapter.up_proj.weight[:min_dim, :min_dim] = torch.eye(min_dim) * 0.1
                    nn.init.zeros_(adapter.up_proj.bias)

                    # Set bias intervals for targeted recovery
                    if hasattr(adapter, 'down_proj') and adapter.down_proj.bias is not None:
                        bias_intervals = self.attack_config['bias_intervals']
                        for j in range(min(len(bias_intervals), adapter.down_proj.bias.shape[0])):
                            adapter.down_proj.bias[j] = bias_intervals[j][0]  # Use lower bound

                # Configure MLP adapter similarly
                if hasattr(block, 'mlp_peft') and block.mlp_peft is not None:
                    adapter = block.mlp_peft

                    nn.init.zeros_(adapter.down_proj.weight)
                    nn.init.zeros_(adapter.down_proj.bias)
                    nn.init.zeros_(adapter.up_proj.weight)
                    min_dim = min(adapter.up_proj.weight.shape[0], adapter.down_proj.weight.shape[0])
                    adapter.up_proj.weight[:min_dim, :min_dim] = torch.eye(min_dim) * 0.1
                    nn.init.zeros_(adapter.up_proj.bias)

                    if hasattr(adapter, 'down_proj') and adapter.down_proj.bias is not None:
                        bias_intervals = self.attack_config['bias_intervals']
                        for j in range(min(len(bias_intervals), adapter.down_proj.bias.shape[0])):
                            adapter.down_proj.bias[j] = bias_intervals[j][0]

    def extract_target_gradients(self, data_batch: torch.Tensor, labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract adapter gradients from model"""
        self.model.train()
        self.model.zero_grad()

        # Forward pass
        outputs, _ = self.model(data_batch)
        loss = self.compute_loss(outputs, labels)

        # Backward pass
        loss.backward()

        # Extract adapter gradients
        adapter_gradients = {}
        for name, param in self.model.named_parameters():
            if 'peft' in name.lower() and param.grad is not None:
                adapter_gradients[name] = param.grad.clone().detach()

        return adapter_gradients

    def reconstruct_patches(self, gradients: Dict[str, torch.Tensor]) -> List[torch.Tensor]:
        """Reconstruct patches from adapter gradients using the PEFTLeak method"""
        recovered_patches = []

        # Group gradients by layer and type
        weight_grads = {}
        bias_grads = {}

        for name, grad in gradients.items():
            if 'weight' in name:
                layer_id = self._extract_layer_id(name)
                weight_grads[layer_id] = grad
            elif 'bias' in name:
                layer_id = self._extract_layer_id(name)
                bias_grads[layer_id] = grad

        # Reconstruct patches using gradient differences
        for layer_id in weight_grads:
            if layer_id in bias_grads:
                weight_grad = weight_grads[layer_id]
                bias_grad = bias_grads[layer_id]

                # Apply PEFTLeak recovery formula
                patches = self._apply_recovery_formula(weight_grad, bias_grad)
                recovered_patches.extend(patches)

        return recovered_patches

    def _extract_layer_id(self, param_name: str) -> str:
        """Extract layer identifier from parameter name"""
        # Extract block number and adapter type
        if 'blocks' in param_name:
            parts = param_name.split('.')
            block_idx = None
            adapter_type = None

            for i, part in enumerate(parts):
                if part == 'blocks' and i + 1 < len(parts):
                    block_idx = parts[i + 1]
                elif 'peft' in part:
                    adapter_type = part

            return f"block_{block_idx}_{adapter_type}" if block_idx and adapter_type else param_name

        return param_name

    def _apply_recovery_formula(self, weight_grad: torch.Tensor, bias_grad: torch.Tensor) -> List[torch.Tensor]:
        """Apply the PEFTLeak recovery formula to reconstruct patches"""
        patches = []

        # Apply the core PEFTLeak formula: (W_grad[k] - W_grad[k+1]) / (b_grad[k] - b_grad[k+1])
        try:
            if weight_grad.dim() >= 2 and bias_grad.dim() >= 1:
                batch_size = bias_grad.shape[0] if bias_grad.dim() > 1 else 1

                for k in range(min(weight_grad.shape[0] - 1, bias_grad.shape[0] - 1)):
                    # Extract consecutive gradients
                    w_k = weight_grad[k]
                    w_k1 = weight_grad[k + 1] if k + 1 < weight_grad.shape[0] else weight_grad[k]

                    b_k = bias_grad[k]
                    b_k1 = bias_grad[k + 1] if k + 1 < bias_grad.shape[0] else bias_grad[k]

                    # Apply recovery formula
                    denominator = b_k - b_k1
                    if abs(denominator) > 1e-8:  # Avoid division by zero
                        numerator = w_k - w_k1
                        recovered_patch = numerator / denominator

                        # Reshape to patch format if needed
                        if recovered_patch.numel() == (3 * self.patch_size * self.patch_size):
                            recovered_patch = recovered_patch.view(3, self.patch_size, self.patch_size)
                            patches.append(recovered_patch)
                        elif recovered_patch.dim() == 1 and recovered_patch.shape[0] >= 3:
                            # Take first patch_size^2 * 3 elements
                            patch_elements = 3 * self.patch_size * self.patch_size
                            if recovered_patch.shape[0] >= patch_elements:
                                patch_data = recovered_patch[:patch_elements]
                                recovered_patch = patch_data.view(3, self.patch_size, self.patch_size)
                                patches.append(recovered_patch)

        except Exception as e:
            print(f"Error in recovery formula: {e}")
            # Create dummy patches if recovery fails
            for _ in range(min(8, self.num_patches // 8)):
                dummy_patch = torch.randn(3, self.patch_size, self.patch_size) * 0.1
                patches.append(dummy_patch)

        return patches[:min(len(patches), self.num_patches)]

    def analyze_adapter_sensitivity(self, data_batch: torch.Tensor) -> Dict[str, float]:
        """Analyze sensitivity of different adapter components to input changes"""
        sensitivity_analysis = {}

        # Original forward pass
        original_output, _ = self.model(data_batch)

        # Test sensitivity to different adapter components
        for name, module in self.model.named_modules():
            if 'peft' in name.lower() and hasattr(module, 'down_proj'):
                # Perturb adapter weights slightly
                with torch.no_grad():
                    original_weight = module.down_proj.weight.clone()
                    perturbation = torch.randn_like(original_weight) * 0.01
                    module.down_proj.weight.data += perturbation

                # Forward pass with perturbed weights
                perturbed_output, _ = self.model(data_batch)

                # Compute sensitivity as output difference
                sensitivity = torch.norm(perturbed_output - original_output).item()
                sensitivity_analysis[name] = sensitivity

                # Restore original weights
                with torch.no_grad():
                    module.down_proj.weight.data = original_weight

        return sensitivity_analysis

    def evaluate_attack_success(self, original_patches: List[torch.Tensor],
                               reconstructed_patches: List[torch.Tensor]) -> Dict[str, float]:
        """Evaluate attack success metrics"""
        if not original_patches or not reconstructed_patches:
            return {'psnr': 0.0, 'ssim': 0.0, 'mse': float('inf'), 'success_rate': 0.0}

        psnr_values = []
        ssim_values = []
        mse_values = []

        min_len = min(len(original_patches), len(reconstructed_patches))

        for i in range(min_len):
            orig = original_patches[i]
            recon = reconstructed_patches[i]

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

    def compute_psnr(self, original: torch.Tensor, reconstructed: torch.Tensor) -> float:
        """Compute PSNR between original and reconstructed patches"""
        mse = torch.mean((original - reconstructed) ** 2)
        if mse == 0:
            return float('inf')
        psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
        return psnr.item()

    def compute_ssim(self, original: torch.Tensor, reconstructed: torch.Tensor) -> float:
        """Compute SSIM between original and reconstructed patches"""
        try:
            from skimage.metrics import structural_similarity as ssim
            
            # Convert tensors to numpy
            orig_np = original.detach().cpu().numpy()
            recon_np = reconstructed.detach().cpu().numpy()
            
            # Ensure proper format (H, W, C) for SSIM
            if orig_np.shape[0] == 3:  # CHW format
                orig_np = np.transpose(orig_np, (1, 2, 0))
                recon_np = np.transpose(recon_np, (1, 2, 0))
            
            return ssim(orig_np, recon_np, multichannel=True, data_range=1.0)
            
        except ImportError:
            # Fallback to simple correlation coefficient
            orig_flat = original.flatten()
            recon_flat = reconstructed.flatten()
            correlation = torch.corrcoef(torch.stack([orig_flat, recon_flat]))[0, 1]
            return correlation.item() if not torch.isnan(correlation) else 0.0
