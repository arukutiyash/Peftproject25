import torch
import torch.nn as nn
import numpy as np
import math
from typing import Dict, List, Tuple, Optional, Any
from defenses.defense_base import BaseDefense #differential_privacy.py


class DifferentialPrivacyDefense(BaseDefense):
    """Differential Privacy defense using Gaussian mechanism"""

    def __init__(self, device: str = 'cuda', epsilon: float = 1.0, delta: float = 1e-5,
                 sensitivity: float = 1.0, clipping_threshold: float = 1.0, **kwargs):
        """
        Initialize DP defense

        Args:
            device: Computing device
            epsilon: Privacy parameter (smaller = more private)
            delta: Privacy parameter (should be small, e.g., 1/n^2)
            sensitivity: L2 sensitivity of the mechanism
            clipping_threshold: Gradient clipping threshold
            **kwargs: Additional defense parameters
        """
        super().__init__(device, **kwargs)

        self.epsilon = epsilon
        self.delta = delta
        self.sensitivity = sensitivity
        self.clipping_threshold = clipping_threshold

        # Calculate noise scale using Gaussian mechanism
        self.noise_scale = self._calculate_noise_scale()

        # Update defense config
        self.defense_config.update({
            'epsilon': epsilon,
            'delta': delta,
            'sensitivity': sensitivity,
            'clipping_threshold': clipping_threshold,
            'noise_scale': self.noise_scale,
            'defense_type': 'differential_privacy'
        })

        # Privacy accounting
        self.privacy_spent = 0.0
        self.query_count = 0

    def _calculate_noise_scale(self) -> float:
        """Calculate noise scale for Gaussian mechanism"""
        if self.epsilon == 0:
            return float('inf')

        # Gaussian mechanism: σ = √(2 ln(1.25/δ)) * Δ / ε
        # Where Δ is the L2 sensitivity
        c = math.sqrt(2 * math.log(1.25 / self.delta))
        sigma = c * self.sensitivity / self.epsilon

        return sigma

    def apply_defense(self, gradients: Dict[str, torch.Tensor],
                      **kwargs) -> Dict[str, torch.Tensor]:
        """Apply differential privacy defense to gradients"""

        # Step 1: Clip gradients to bound sensitivity
        clipped_gradients = self.clip_gradients(gradients, self.clipping_threshold)

        # Step 2: Add calibrated Gaussian noise
        noisy_gradients = self.add_gaussian_noise(clipped_gradients)

        # Step 3: Update privacy accounting
        self._update_privacy_accounting()

        return noisy_gradients

    def add_gaussian_noise(self, gradients: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Add calibrated Gaussian noise to gradients"""
        noisy_gradients = {}

        for name, grad in gradients.items():
            if isinstance(grad, torch.Tensor):
                # Generate Gaussian noise with calibrated scale
                noise = torch.randn_like(grad) * self.noise_scale
                noisy_gradients[name] = grad + noise.to(grad.device)
            else:
                noisy_gradients[name] = grad

        return noisy_gradients

    def _update_privacy_accounting(self):
        """Update privacy accounting using composition theorems"""
        self.query_count += 1

        # Simple composition (conservative)
        self.privacy_spent += self.epsilon

        # Advanced composition could be implemented here
        # For now, we use basic composition

    def get_privacy_cost(self) -> float:
        """Get current privacy cost"""
        return self.privacy_spent

    def get_remaining_budget(self, total_budget: float = 10.0) -> float:
        """Get remaining privacy budget"""
        return max(0.0, total_budget - self.privacy_spent)

    def is_privacy_exhausted(self, total_budget: float = 10.0) -> bool:
        """Check if privacy budget is exhausted"""
        return self.privacy_spent >= total_budget

    def reset_privacy_accounting(self):
        """Reset privacy accounting"""
        self.privacy_spent = 0.0
        self.query_count = 0


class AdaptiveDPDefense(DifferentialPrivacyDefense):
    """Adaptive DP that adjusts noise based on privacy budget and utility"""

    def __init__(self, device: str = 'cuda', initial_epsilon: float = 1.0,
                 total_budget: float = 10.0, adaptation_strategy: str = 'linear', **kwargs):
        """
        Initialize adaptive DP defense

        Args:
            device: Computing device
            initial_epsilon: Initial epsilon value
            total_budget: Total privacy budget
            adaptation_strategy: How to adapt epsilon ('linear', 'exponential', 'utility_based')
            **kwargs: Additional parameters
        """
        super().__init__(device, epsilon=initial_epsilon, **kwargs)

        self.initial_epsilon = initial_epsilon
        self.total_budget = total_budget
        self.adaptation_strategy = adaptation_strategy
        self.utility_history = []

    def adapt_epsilon(self, utility_metric: Optional[float] = None):
        """Adapt epsilon based on remaining budget and utility"""
        remaining_budget = self.get_remaining_budget(self.total_budget)

        if self.adaptation_strategy == 'linear':
            # Linearly decrease epsilon as budget is consumed
            budget_ratio = remaining_budget / self.total_budget
            self.epsilon = self.initial_epsilon * budget_ratio

        elif self.adaptation_strategy == 'exponential':
            # Exponentially decrease epsilon
            budget_ratio = remaining_budget / self.total_budget
            self.epsilon = self.initial_epsilon * (budget_ratio ** 2)

        elif self.adaptation_strategy == 'utility_based' and utility_metric is not None:
            # Adapt based on utility feedback
            self.utility_history.append(utility_metric)

            if len(self.utility_history) > 1:
                utility_trend = self.utility_history[-1] - self.utility_history[-2]

                if utility_trend < -0.1:  # Utility dropping significantly
                    self.epsilon = min(self.initial_epsilon, self.epsilon * 1.1)  # Reduce noise
                elif utility_trend > 0.05:  # Utility stable/improving
                    self.epsilon = max(0.1, self.epsilon * 0.9)  # Increase noise

        # Recalculate noise scale
        self.noise_scale = self._calculate_noise_scale()

        # Update config
        self.defense_config['epsilon'] = self.epsilon
        self.defense_config['noise_scale'] = self.noise_scale

    def apply_defense(self, gradients: Dict[str, torch.Tensor],
                      utility_metric: Optional[float] = None, **kwargs) -> Dict[str, torch.Tensor]:
        """Apply adaptive DP defense"""
        # Adapt epsilon before applying defense
        self.adapt_epsilon(utility_metric)

        # Check if privacy budget is exhausted
        if self.is_privacy_exhausted(self.total_budget):
            self.logger.warning("Privacy budget exhausted! Applying maximum noise.")
            self.epsilon = 0.01  # Very small epsilon = very high noise
            self.noise_scale = self._calculate_noise_scale()

        return super().apply_defense(gradients, **kwargs)


class RenyiDPDefense(DifferentialPrivacyDefense):
    """Rényi Differential Privacy defense with tighter composition"""

    def __init__(self, device: str = 'cuda', alpha: float = 2.0, **kwargs):
        """
        Initialize RDP defense

        Args:
            device: Computing device
            alpha: Rényi parameter (alpha > 1)
            **kwargs: Additional parameters
        """
        super().__init__(device, **kwargs)

        self.alpha = alpha
        self.rdp_spent = 0.0

        # Update config
        self.defense_config.update({
            'alpha': alpha,
            'privacy_mechanism': 'renyi_dp'
        })

    def _calculate_rdp_noise_scale(self) -> float:
        """Calculate noise scale for RDP mechanism"""
        if self.epsilon == 0:
            return float('inf')

        # RDP noise scale calculation
        sigma = math.sqrt(self.alpha) * self.sensitivity / self.epsilon
        return sigma

    def apply_defense(self, gradients: Dict[str, torch.Tensor], **kwargs) -> Dict[str, torch.Tensor]:
        """Apply RDP defense"""
        # Use RDP-specific noise scale
        original_noise_scale = self.noise_scale
        self.noise_scale = self._calculate_rdp_noise_scale()

        # Apply standard DP mechanism with RDP noise
        defended_gradients = super().apply_defense(gradients, **kwargs)

        # Update RDP accounting
        self._update_rdp_accounting()

        # Restore original noise scale
        self.noise_scale = original_noise_scale

        return defended_gradients

    def _update_rdp_accounting(self):
        """Update RDP privacy accounting"""
        # RDP composition: ε_α(M^k) = k * ε_α(M)
        rdp_cost = self.alpha * (self.sensitivity ** 2) / (2 * (self.noise_scale ** 2))
        self.rdp_spent += rdp_cost

        # Convert to (ε, δ)-DP using RDP conversion
        self._convert_rdp_to_dp()

    def _convert_rdp_to_dp(self):
        """Convert RDP to (ε, δ)-DP"""
        if self.rdp_spent == 0:
            return

        # RDP to DP conversion: ε = ε_α + ln(1/δ)/(α-1)
        converted_epsilon = self.rdp_spent + math.log(1 / self.delta) / (self.alpha - 1)

        # Update privacy spent with converted value
        self.privacy_spent = converted_epsilon


class LocalDPDefense(BaseDefense):
    """Local Differential Privacy defense (client-side noise addition)"""

    def __init__(self, device: str = 'cuda', epsilon_local: float = 1.0,
                 mechanism: str = 'laplace', **kwargs):
        """
        Initialize local DP defense

        Args:
            device: Computing device
            epsilon_local: Local privacy parameter
            mechanism: Noise mechanism ('laplace', 'gaussian', 'randomized_response')
            **kwargs: Additional parameters
        """
        super().__init__(device, **kwargs)

        self.epsilon_local = epsilon_local
        self.mechanism = mechanism

        self.defense_config.update({
            'epsilon_local': epsilon_local,
            'mechanism': mechanism,
            'defense_type': 'local_differential_privacy'
        })

    def apply_defense(self, gradients: Dict[str, torch.Tensor], **kwargs) -> Dict[str, torch.Tensor]:
        """Apply local DP defense"""
        if self.mechanism == 'laplace':
            return self._apply_laplace_mechanism(gradients)
        elif self.mechanism == 'gaussian':
            return self._apply_gaussian_mechanism(gradients)
        elif self.mechanism == 'randomized_response':
            return self._apply_randomized_response(gradients)
        else:
            self.logger.error(f"Unknown mechanism: {self.mechanism}")
            return gradients

    def _apply_laplace_mechanism(self, gradients: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply Laplace mechanism for local DP"""
        scale = 1.0 / self.epsilon_local  # Laplace scale parameter

        noisy_gradients = {}
        for name, grad in gradients.items():
            if isinstance(grad, torch.Tensor):
                # Sample from Laplace distribution
                laplace_noise = torch.distributions.Laplace(0, scale).sample(grad.shape).to(grad.device)
                noisy_gradients[name] = grad + laplace_noise
            else:
                noisy_gradients[name] = grad

        return noisy_gradients

    def _apply_gaussian_mechanism(self, gradients: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply Gaussian mechanism for local DP"""
        sigma = math.sqrt(2 * math.log(1.25)) / self.epsilon_local

        noisy_gradients = {}
        for name, grad in gradients.items():
            if isinstance(grad, torch.Tensor):
                gaussian_noise = torch.randn_like(grad) * sigma
                noisy_gradients[name] = grad + gaussian_noise
            else:
                noisy_gradients[name] = grad

        return noisy_gradients

    def _apply_randomized_response(self, gradients: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply randomized response mechanism"""
        p = math.exp(self.epsilon_local) / (math.exp(self.epsilon_local) + 1)

        noisy_gradients = {}
        for name, grad in gradients.items():
            if isinstance(grad, torch.Tensor):
                # Binary randomized response on sign of gradients
                signs = torch.sign(grad)
                random_flips = torch.bernoulli(torch.full_like(grad, 1 - p))
                flipped_signs = signs * (1 - 2 * random_flips)  # Flip signs with probability 1-p

                noisy_gradients[name] = torch.abs(grad) * flipped_signs
            else:
                noisy_gradients[name] = grad

        return noisy_gradients

    def get_privacy_cost(self) -> float:
        """Local DP cost is per client"""
        return self.epsilon_local
