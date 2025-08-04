import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from defenses.defense_base import BaseDefense #grad_prune_defense.py


class GradientPruningDefense(BaseDefense):
    """Gradient pruning defense to reduce information leakage"""

    def __init__(self, device: str = 'cuda', pruning_ratio: float = 0.5,
                 pruning_strategy: str = 'magnitude', adaptive: bool = False, **kwargs):
        """
        Initialize gradient pruning defense

        Args:
            device: Computing device
            pruning_ratio: Fraction of gradients to prune (0-1)
            pruning_strategy: Strategy for pruning ('magnitude', 'random', 'structured', 'topk')
            adaptive: Whether to adapt pruning ratio based on gradient patterns
            **kwargs: Additional defense parameters
        """
        super().__init__(device, **kwargs)

        self.pruning_ratio = pruning_ratio
        self.pruning_strategy = pruning_strategy
        self.adaptive = adaptive

        # Update defense config
        self.defense_config.update({
            'pruning_ratio': pruning_ratio,
            'pruning_strategy': pruning_strategy,
            'adaptive': adaptive,
            'defense_type': 'gradient_compression'
        })

        # Adaptive pruning parameters
        self.gradient_history = []
        self.sparsity_history = []
        self.adaptation_rate = 0.1

    def apply_defense(self, gradients: Dict[str, torch.Tensor], **kwargs) -> Dict[str, torch.Tensor]:
        """Apply gradient pruning defense"""

        if self.adaptive:
            self._adapt_pruning_ratio(gradients)

        if self.pruning_strategy == 'magnitude':
            return self._magnitude_based_pruning(gradients)
        elif self.pruning_strategy == 'random':
            return self._random_pruning(gradients)
        elif self.pruning_strategy == 'structured':
            return self._structured_pruning(gradients)
        elif self.pruning_strategy == 'topk':
            return self._topk_pruning(gradients)
        else:
            self.logger.error(f"Unknown pruning strategy: {self.pruning_strategy}")
            return gradients

    def _magnitude_based_pruning(self, gradients: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Prune gradients based on magnitude"""
        pruned_gradients = {}

        for name, grad in gradients.items():
            if isinstance(grad, torch.Tensor):
                # Flatten gradient for easier processing
                grad_flat = grad.flatten()

                # Calculate threshold for pruning
                grad_abs = torch.abs(grad_flat)
                k = int(len(grad_flat) * self.pruning_ratio)

                if k > 0:
                    # Find threshold value (k-th smallest absolute value)
                    threshold = torch.kthvalue(grad_abs, k)[0]

                    # Create mask for values above threshold
                    mask = grad_abs >= threshold

                    # Apply mask and reshape back
                    pruned_flat = grad_flat * mask.float()
                    pruned_gradients[name] = pruned_flat.reshape(grad.shape)
                else:
                    pruned_gradients[name] = grad
            else:
                pruned_gradients[name] = grad

        return pruned_gradients

    def _random_pruning(self, gradients: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Randomly prune gradients"""
        pruned_gradients = {}

        for name, grad in gradients.items():
            if isinstance(grad, torch.Tensor):
                # Create random mask
                mask = torch.bernoulli(torch.full_like(grad, 1 - self.pruning_ratio))
                pruned_gradients[name] = grad * mask
            else:
                pruned_gradients[name] = grad

        return pruned_gradients

    def _structured_pruning(self, gradients: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply structured pruning (prune entire channels/neurons)"""
        pruned_gradients = {}

        for name, grad in gradients.items():
            if isinstance(grad, torch.Tensor) and grad.dim() >= 2:
                # For 2D gradients (like linear layers), prune entire rows/columns
                if 'weight' in name:
                    pruned_gradients[name] = self._prune_structured_weight(grad)
                elif 'bias' in name:
                    pruned_gradients[name] = self._prune_structured_bias(grad)
                else:
                    # Fallback to magnitude pruning
                    pruned_gradients[name] = self._magnitude_based_pruning({name: grad})[name]
            else:
                # For 1D or scalar gradients, use magnitude pruning
                pruned_gradients[name] = self._magnitude_based_pruning({name: grad})[name]

        return pruned_gradients

    def _prune_structured_weight(self, weight_grad: torch.Tensor) -> torch.Tensor:
        """Prune entire rows/columns from weight gradients"""
        if weight_grad.dim() == 2:
            # For 2D weight matrices, prune entire rows (output neurons)
            out_features, in_features = weight_grad.shape

            # Calculate L2 norm of each row
            row_norms = torch.norm(weight_grad, dim=1)

            # Determine number of rows to keep
            num_keep = int(out_features * (1 - self.pruning_ratio))

            if num_keep > 0:
                # Keep top-k rows by norm
                _, top_indices = torch.topk(row_norms, num_keep)

                # Create mask
                mask = torch.zeros_like(weight_grad)
                mask[top_indices, :] = 1.0

                return weight_grad * mask

        return weight_grad

    def _prune_structured_bias(self, bias_grad: torch.Tensor) -> torch.Tensor:
        """Prune bias gradients corresponding to pruned neurons"""
        if bias_grad.dim() == 1:
            # Calculate absolute values
            bias_abs = torch.abs(bias_grad)

            # Determine number of elements to keep
            num_keep = int(len(bias_grad) * (1 - self.pruning_ratio))

            if num_keep > 0:
                # Keep top-k elements
                _, top_indices = torch.topk(bias_abs, num_keep)

                # Create mask
                mask = torch.zeros_like(bias_grad)
                mask[top_indices] = 1.0

                return bias_grad * mask

        return bias_grad

    def _topk_pruning(self, gradients: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Keep only top-k gradients globally across all parameters"""
        # Collect all gradient values
        all_gradients = []
        gradient_info = []  # Store (name, original_shape, flat_indices)

        for name, grad in gradients.items():
            if isinstance(grad, torch.Tensor):
                grad_flat = grad.flatten()
                all_gradients.append(grad_flat)
                gradient_info.append((name, grad.shape, len(grad_flat)))

        if not all_gradients:
            return gradients

        # Concatenate all gradients
        all_grads_tensor = torch.cat(all_gradients)

        # Calculate global top-k
        total_elements = len(all_grads_tensor)
        k = int(total_elements * (1 - self.pruning_ratio))

        if k > 0:
            # Get top-k indices globally
            _, top_indices = torch.topk(torch.abs(all_grads_tensor), k)

            # Create global mask
            global_mask = torch.zeros_like(all_grads_tensor)
            global_mask[top_indices] = 1.0

            # Apply mask and reconstruct gradients
            masked_grads = all_grads_tensor * global_mask

            # Reconstruct individual gradients
            pruned_gradients = {}
            start_idx = 0

            for name, original_shape, length in gradient_info:
                end_idx = start_idx + length
                grad_masked = masked_grads[start_idx:end_idx].reshape(original_shape)
                pruned_gradients[name] = grad_masked
                start_idx = end_idx

            # Add non-tensor gradients
            for name, grad in gradients.items():
                if not isinstance(grad, torch.Tensor):
                    pruned_gradients[name] = grad

            return pruned_gradients

        return gradients

    def _adapt_pruning_ratio(self, gradients: Dict[str, torch.Tensor]):
        """Adapt pruning ratio based on gradient patterns"""
        # Calculate current gradient sparsity
        total_elements = 0
        zero_elements = 0

        for grad in gradients.values():
            if isinstance(grad, torch.Tensor):
                total_elements += grad.numel()
                zero_elements += (torch.abs(grad) < 1e-8).sum().item()

        current_sparsity = zero_elements / total_elements if total_elements > 0 else 0
        self.sparsity_history.append(current_sparsity)

        # Calculate gradient norm
        current_norm = self.compute_gradient_norm(gradients)
        self.gradient_history.append(current_norm)

        # Keep only recent history
        if len(self.sparsity_history) > 10:
            self.sparsity_history.pop(0)
            self.gradient_history.pop(0)

        if len(self.sparsity_history) > 2:
            # Increase pruning if gradients are naturally sparse
            if current_sparsity > 0.5:
                self.pruning_ratio = min(0.9, self.pruning_ratio + self.adaptation_rate)
            else:
                self.pruning_ratio = max(0.1, self.pruning_ratio - self.adaptation_rate)

        # Update config
        self.defense_config['pruning_ratio'] = self.pruning_ratio

    def get_privacy_cost(self) -> float:
        """Estimate privacy cost of gradient pruning"""
        # Privacy increases with more pruning
        return self.pruning_ratio * 1.5  # Scaling factor

    def analyze_pruning_effect(self, original_gradients: Dict[str, torch.Tensor],
                               pruned_gradients: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Analyze the effect of pruning on gradients"""
        analysis = {}

        for name in original_gradients:
            if name in pruned_gradients and isinstance(original_gradients[name], torch.Tensor):
                orig = original_gradients[name]
                pruned = pruned_gradients[name]

                # Calculate sparsity
                total_elements = orig.numel()
                zero_elements = (torch.abs(pruned) < 1e-8).sum().item()
                sparsity = zero_elements / total_elements

                # Calculate information retention
                orig_norm = torch.norm(orig).item()
                pruned_norm = torch.norm(pruned).item()
                retention_ratio = pruned_norm / (orig_norm + 1e-8)

                # Calculate correlation
                if orig_norm > 0 and pruned_norm > 0:
                    correlation = torch.sum(orig * pruned).item() / (orig_norm * pruned_norm)
                else:
                    correlation = 0.0

                analysis[name] = {
                    'sparsity': sparsity,
                    'retention_ratio': retention_ratio,
                    'correlation': correlation,
                    'compression_ratio': 1 - sparsity
                }

        return analysis


class AdaptiveGradientPruning(GradientPruningDefense):
    """Adaptive gradient pruning with dynamic strategy selection"""

    def __init__(self, device: str = 'cuda', **kwargs):
        super().__init__(device, adaptive=True, **kwargs)

        self.strategy_performance = {
            'magnitude': [],
            'random': [],
            'structured': [],
            'topk': []
        }
        self.current_strategy_idx = 0
        self.strategies = ['magnitude', 'random', 'structured', 'topk']
        self.evaluation_interval = 5
        self.applications_count = 0

    def apply_defense(self, gradients: Dict[str, torch.Tensor],
                      utility_feedback: Optional[float] = None, **kwargs) -> Dict[str, torch.Tensor]:
        """Apply adaptive pruning with strategy selection"""

        self.applications_count += 1

        # Evaluate and switch strategies periodically
        if self.applications_count % self.evaluation_interval == 0:
            if utility_feedback is not None:
                self._update_strategy_performance(utility_feedback)
            self._select_best_strategy()

        return super().apply_defense(gradients, **kwargs)

    def _update_strategy_performance(self, utility_feedback: float):
        """Update performance metrics for current strategy"""
        current_strategy = self.strategies[self.current_strategy_idx]
        self.strategy_performance[current_strategy].append(utility_feedback)

        # Keep only recent performance
        if len(self.strategy_performance[current_strategy]) > 10:
            self.strategy_performance[current_strategy].pop(0)

    def _select_best_strategy(self):
        """Select best performing strategy"""
        best_strategy = None
        best_performance = float('-inf')

        for strategy, performance_list in self.strategy_performance.items():
            if performance_list:
                avg_performance = np.mean(performance_list)
                if avg_performance > best_performance:
                    best_performance = avg_performance
                    best_strategy = strategy

        if best_strategy and best_strategy != self.pruning_strategy:
            self.pruning_strategy = best_strategy
            self.current_strategy_idx = self.strategies.index(best_strategy)
            self.logger.info(f"Switched to {best_strategy} pruning strategy")


class LayerWiseGradientPruning(GradientPruningDefense):
    """Layer-wise gradient pruning with different ratios per layer"""

    def __init__(self, device: str = 'cuda', layer_pruning_ratios: Optional[Dict[str, float]] = None, **kwargs):
        super().__init__(device, **kwargs)

        self.layer_pruning_ratios = layer_pruning_ratios or {}
        self.default_ratio = self.pruning_ratio

    def apply_defense(self, gradients: Dict[str, torch.Tensor], **kwargs) -> Dict[str, torch.Tensor]:
        """Apply layer-wise pruning with different ratios"""
        pruned_gradients = {}

        for name, grad in gradients.items():
            if isinstance(grad, torch.Tensor):
                # Get layer-specific pruning ratio
                layer_ratio = self._get_layer_pruning_ratio(name)

                # Temporarily set pruning ratio for this layer
                original_ratio = self.pruning_ratio
                self.pruning_ratio = layer_ratio

                # Apply pruning to single gradient
                layer_result = self._magnitude_based_pruning({name: grad})
                pruned_gradients[name] = layer_result[name]

                # Restore original ratio
                self.pruning_ratio = original_ratio
            else:
                pruned_gradients[name] = grad

        return pruned_gradients

    def _get_layer_pruning_ratio(self, layer_name: str) -> float:
        """Get pruning ratio for specific layer"""
        # Check for exact match
        if layer_name in self.layer_pruning_ratios:
            return self.layer_pruning_ratios[layer_name]

        # Check for partial matches
        for pattern, ratio in self.layer_pruning_ratios.items():
            if pattern in layer_name:
                return ratio

        # Use default ratio
        return self.default_ratio

    def set_layer_pruning_ratio(self, layer_pattern: str, ratio: float):
        """Set pruning ratio for layers matching pattern"""
        self.layer_pruning_ratios[layer_pattern] = ratio
