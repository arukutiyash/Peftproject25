import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any, Union
from defenses.defense_base import BaseDefense
from defenses.mixup_defense import MixUpDefense
from defenses.instahide_defense import InstaHideDefense
from defenses.differential_privacy import DifferentialPrivacyDefense #combined_defenses.py
from defenses.grad_prune_defense import GradientPruningDefense


class CombinedDefense(BaseDefense):
    """Combine multiple defense mechanisms"""

    def __init__(self, defenses: List[Tuple[BaseDefense, float]],
                 combination_strategy: str = 'sequential', device: str = 'cuda', **kwargs):
        """
        Initialize combined defense

        Args:
            defenses: List of (defense_instance, weight) tuples
            combination_strategy: How to combine defenses ('sequential', 'parallel', 'adaptive')
            device: Computing device
            **kwargs: Additional parameters
        """
        super().__init__(device, **kwargs)

        self.defenses = defenses
        self.combination_strategy = combination_strategy

        # Validate weights sum to 1 for parallel combination
        if combination_strategy == 'parallel':
            total_weight = sum(weight for _, weight in defenses)
            if abs(total_weight - 1.0) > 1e-6:
                # Normalize weights
                self.defenses = [(defense, weight / total_weight) for defense, weight in defenses]

        self.defense_config.update({
            'combination_strategy': combination_strategy,
            'num_defenses': len(defenses),
            'defense_types': [type(d).__name__ for d, _ in defenses]
        })

    def apply_defense(self, gradients: Dict[str, torch.Tensor], **kwargs) -> Dict[str, torch.Tensor]:
        """Apply combined defense"""

        if self.combination_strategy == 'sequential':
            return self._apply_sequential(gradients, **kwargs)
        elif self.combination_strategy == 'parallel':
            return self._apply_parallel(gradients, **kwargs)
        elif self.combination_strategy == 'adaptive':
            return self._apply_adaptive(gradients, **kwargs)
        else:
            self.logger.error(f"Unknown combination strategy: {self.combination_strategy}")
            return gradients

    def _apply_sequential(self, gradients: Dict[str, torch.Tensor], **kwargs) -> Dict[str, torch.Tensor]:
        """Apply defenses sequentially"""
        current_gradients = gradients

        for defense, weight in self.defenses:
            if weight > 0:  # Only apply if weight is positive
                defended_gradients = defense.apply_defense_with_validation(current_gradients, **kwargs)

                # Weighted combination with original
                if weight < 1.0:
                    current_gradients = self._weighted_combine(
                        current_gradients, defended_gradients, weight
                    )
                else:
                    current_gradients = defended_gradients

        return current_gradients

    def _apply_parallel(self, gradients: Dict[str, torch.Tensor], **kwargs) -> Dict[str, torch.Tensor]:
        """Apply defenses in parallel and combine results"""
        defended_results = []
        weights = []

        for defense, weight in self.defenses:
            if weight > 0:
                defended_gradients = defense.apply_defense_with_validation(gradients, **kwargs)
                defended_results.append(defended_gradients)
                weights.append(weight)

        if not defended_results:
            return gradients

        # Weighted combination of all defended gradients
        combined_gradients = {}

        for name in gradients:
            if isinstance(gradients[name], torch.Tensor):
                # Weighted sum of defended gradients
                combined_grad = torch.zeros_like(gradients[name])

                for defended_grads, weight in zip(defended_results, weights):
                    if name in defended_grads:
                        combined_grad += weight * defended_grads[name]

                combined_gradients[name] = combined_grad
            else:
                combined_gradients[name] = gradients[name]

        return combined_gradients

    def _apply_adaptive(self, gradients: Dict[str, torch.Tensor], **kwargs) -> Dict[str, torch.Tensor]:
        """Adaptively select and apply defenses based on gradient characteristics"""

        # Analyze gradient characteristics
        grad_analysis = self._analyze_gradients(gradients)

        # Select best defense based on analysis
        selected_defense, selected_weight = self._select_defense(grad_analysis)

        if selected_defense:
            return selected_defense.apply_defense_with_validation(gradients, **kwargs)

        return gradients

    def _weighted_combine(self, gradients1: Dict[str, torch.Tensor],
                          gradients2: Dict[str, torch.Tensor],
                          weight: float) -> Dict[str, torch.Tensor]:
        """Combine two gradient dictionaries with weighting"""
        combined = {}

        for name in gradients1:
            if isinstance(gradients1[name], torch.Tensor) and name in gradients2:
                combined[name] = (1 - weight) * gradients1[name] + weight * gradients2[name]
            else:
                combined[name] = gradients1[name]

        return combined

    def _analyze_gradients(self, gradients: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Analyze gradient characteristics for adaptive selection"""
        analysis = {}

        # Calculate overall gradient norm
        total_norm = self.compute_gradient_norm(gradients)
        analysis['gradient_norm'] = total_norm

        # Calculate sparsity
        total_elements = 0
        zero_elements = 0

        for grad in gradients.values():
            if isinstance(grad, torch.Tensor):
                total_elements += grad.numel()
                zero_elements += (torch.abs(grad) < 1e-8).sum().item()

        analysis['sparsity'] = zero_elements / total_elements if total_elements > 0 else 0

        # Calculate variance across gradients
        all_values = []
        for grad in gradients.values():
            if isinstance(grad, torch.Tensor):
                all_values.extend(grad.flatten().tolist())

        if all_values:
            analysis['variance'] = float(torch.var(torch.tensor(all_values)))
        else:
            analysis['variance'] = 0.0

        return analysis

    def _select_defense(self, analysis: Dict[str, float]) -> Tuple[Optional[BaseDefense], float]:
        """Select best defense based on gradient analysis"""

        # Simple heuristic-based selection
        gradient_norm = analysis.get('gradient_norm', 0)
        sparsity = analysis.get('sparsity', 0)
        variance = analysis.get('variance', 0)

        # High norm gradients -> use DP or pruning
        if gradient_norm > 10.0:
            for defense, weight in self.defenses:
                if isinstance(defense, (DifferentialPrivacyDefense, GradientPruningDefense)):
                    return defense, weight

        # High sparsity -> use pruning
        if sparsity > 0.5:
            for defense, weight in self.defenses:
                if isinstance(defense, GradientPruningDefense):
                    return defense, weight

        # High variance -> use mixup or instahide
        if variance > 1.0:
            for defense, weight in self.defenses:
                if isinstance(defense, (MixUpDefense, InstaHideDefense)):
                    return defense, weight

        # Default: use first defense
        if self.defenses:
            return self.defenses[0]

        return None, 0.0

    def get_privacy_cost(self) -> float:
        """Calculate combined privacy cost"""
        if self.combination_strategy == 'sequential':
            # Sequential: sum of individual costs
            return sum(defense.get_privacy_cost() * weight for defense, weight in self.defenses)
        elif self.combination_strategy == 'parallel':
            # Parallel: weighted average
            return sum(defense.get_privacy_cost() * weight for defense, weight in self.defenses)
        else:
            # Adaptive: maximum cost (worst case)
            return max(defense.get_privacy_cost() for defense, _ in self.defenses) if self.defenses else 0.0


class MixUpDPCombination(CombinedDefense):
    """Specific combination of MixUp and Differential Privacy"""

    def __init__(self, device: str = 'cuda', mixup_alpha: float = 1.0,
                 dp_epsilon: float = 1.0, combination_weight: float = 0.5, **kwargs):
        # Initialize individual defenses
        mixup_defense = MixUpDefense(device=device, alpha=mixup_alpha)
        dp_defense = DifferentialPrivacyDefense(device=device, epsilon=dp_epsilon)

        # Combine with specified weight
        defenses = [
            (mixup_defense, combination_weight),
            (dp_defense, 1 - combination_weight)
        ]

        super().__init__(defenses, combination_strategy='sequential', device=device, **kwargs)


class InstaHidePruningCombination(CombinedDefense):
    """Specific combination of InstaHide and Gradient Pruning"""

    def __init__(self, device: str = 'cuda', k_mix: int = 4,
                 pruning_ratio: float = 0.5, **kwargs):
        # Initialize individual defenses
        instahide_defense = InstaHideDefense(device=device, k_mix=k_mix)
        pruning_defense = GradientPruningDefense(device=device, pruning_ratio=pruning_ratio)

        # Apply sequentially: InstaHide first, then pruning
        defenses = [
            (instahide_defense, 1.0),
            (pruning_defense, 1.0)
        ]

        super().__init__(defenses, combination_strategy='sequential', device=device, **kwargs)


class ComprehensiveDefense(CombinedDefense):
    """Comprehensive defense using all available mechanisms"""

    def __init__(self, device: str = 'cuda', defense_weights: Optional[Dict[str, float]] = None, **kwargs):
        # Default weights for all defenses
        default_weights = {
            'mixup': 0.25,
            'instahide': 0.25,
            'differential_privacy': 0.25,
            'gradient_pruning': 0.25
        }

        if defense_weights:
            default_weights.update(defense_weights)

        # Initialize all defenses
        mixup_defense = MixUpDefense(device=device)
        instahide_defense = InstaHideDefense(device=device)
        dp_defense = DifferentialPrivacyDefense(device=device)
        pruning_defense = GradientPruningDefense(device=device)

        defenses = [
            (mixup_defense, default_weights['mixup']),
            (instahide_defense, default_weights['instahide']),
            (dp_defense, default_weights['differential_privacy']),
            (pruning_defense, default_weights['gradient_pruning'])
        ]

        super().__init__(defenses, combination_strategy='parallel', device=device, **kwargs)


class AdaptiveComprehensiveDefense(ComprehensiveDefense):
    """Adaptive comprehensive defense that adjusts based on attack detection"""

    def __init__(self, device: str = 'cuda', **kwargs):
        super().__init__(device=device, **kwargs)

        self.attack_detection_threshold = 2.0
        self.recent_gradient_patterns = []
        self.base_weights = {defense: weight for defense, weight in self.defenses}

    def detect_potential_attack(self, gradients: Dict[str, torch.Tensor]) -> bool:
        """Detect potential gradient inversion attack"""

        # Analyze gradient patterns
        current_norm = self.compute_gradient_norm(gradients)
        self.recent_gradient_patterns.append(current_norm)

        # Keep only recent history
        if len(self.recent_gradient_patterns) > 20:
            self.recent_gradient_patterns.pop(0)

        if len(self.recent_gradient_patterns) > 5:
            # Check for anomalous patterns
            mean_norm = np.mean(self.recent_gradient_patterns[:-1])
            std_norm = np.std(self.recent_gradient_patterns[:-1])

            if std_norm > 0:
                z_score = abs(current_norm - mean_norm) / std_norm
                return z_score > self.attack_detection_threshold

        return False

    def adapt_defense_weights(self, attack_detected: bool):
        """Adapt defense weights based on attack detection"""

        if attack_detected:
            # Increase privacy-preserving defenses
            self.defenses = [
                (defense, min(1.0, weight * 1.5)) if isinstance(defense, (DifferentialPrivacyDefense, InstaHideDefense))
                else (defense, max(0.1, weight * 0.8))
                for defense, weight in self.defenses
            ]
        else:
            # Gradually return to base weights
            self.defenses = [
                (defense, 0.9 * weight + 0.1 * base_weight)
                for (defense, weight), base_weight in zip(self.defenses, self.base_weights.values())
            ]

        # Renormalize weights
        total_weight = sum(weight for _, weight in self.defenses)
        if total_weight > 0:
            self.defenses = [(defense, weight / total_weight) for defense, weight in self.defenses]

    def apply_defense(self, gradients: Dict[str, torch.Tensor], **kwargs) -> Dict[str, torch.Tensor]:
        """Apply adaptive comprehensive defense"""

        # Detect potential attacks
        attack_detected = self.detect_potential_attack(gradients)

        # Adapt defense weights
        self.adapt_defense_weights(attack_detected)

        # Apply defenses with adapted weights
        return super().apply_defense(gradients, **kwargs)


# Utility functions for creating common defense combinations
def create_lightweight_defense(device: str = 'cuda') -> CombinedDefense:
    """Create lightweight defense combination for minimal overhead"""
    pruning_defense = GradientPruningDefense(device=device, pruning_ratio=0.3)

    return CombinedDefense(
        defenses=[(pruning_defense, 1.0)],
        combination_strategy='sequential',
        device=device
    )


def create_strong_privacy_defense(device: str = 'cuda') -> CombinedDefense:
    """Create strong privacy defense for maximum protection"""
    dp_defense = DifferentialPrivacyDefense(device=device, epsilon=0.5)
    instahide_defense = InstaHideDefense(device=device, k_mix=6, mask_ratio=0.7)

    return CombinedDefense(
        defenses=[
            (instahide_defense, 0.6),
            (dp_defense, 0.4)
        ],
        combination_strategy='sequential',
        device=device
    )


def create_balanced_defense(device: str = 'cuda') -> CombinedDefense:
    """Create balanced defense for good privacy-utility tradeoff"""
    mixup_defense = MixUpDefense(device=device, alpha=1.0)
    dp_defense = DifferentialPrivacyDefense(device=device, epsilon=1.0)
    pruning_defense = GradientPruningDefense(device=device, pruning_ratio=0.4)

    return CombinedDefense(
        defenses=[
            (mixup_defense, 0.4),
            (dp_defense, 0.3),
            (pruning_defense, 0.3)
        ],
        combination_strategy='parallel',
        device=device
    )
