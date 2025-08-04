import torch
import torch.nn as nn
import numpy as np
from abc import ABC, abstractmethod #defense_base.py
from typing import Dict, List, Tuple, Optional, Any, Union
import logging


class BaseDefense(ABC):
    """Base class for privacy-preserving defense mechanisms"""

    def __init__(self, device: str = 'cuda', **kwargs):
        """
        Initialize base defense

        Args:
            device: Computing device
            **kwargs: Defense-specific parameters
        """
        self.device = device
        self.defense_config = {
            'enabled': True,
            'noise_scale': 0.1,
            'privacy_budget': 1.0,
            'clipping_threshold': 1.0
        }
        self.defense_config.update(kwargs)

        # Defense statistics
        self.stats = {
            'total_applications': 0,
            'gradient_norm_before': [],
            'gradient_norm_after': [],
            'privacy_cost': 0.0,
            'utility_loss': 0.0
        }

        # Setup logging
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def apply_defense(self, gradients: Dict[str, torch.Tensor],
                      batch_data: Optional[torch.Tensor] = None,
                      **kwargs) -> Dict[str, torch.Tensor]:
        """
        Apply defense mechanism to gradients

        Args:
            gradients: Dictionary of parameter gradients
            batch_data: Original batch data (if needed for defense)
            **kwargs: Additional defense parameters

        Returns:
            Defended gradients
        """
        pass

    @abstractmethod
    def get_privacy_cost(self) -> float:
        """Calculate privacy cost of the defense mechanism"""
        pass

    def compute_gradient_norm(self, gradients: Dict[str, torch.Tensor]) -> float:
        """Compute L2 norm of gradients"""
        total_norm = 0.0
        for grad in gradients.values():
            if isinstance(grad, torch.Tensor):
                total_norm += torch.norm(grad).item() ** 2
        return np.sqrt(total_norm)

    def clip_gradients(self, gradients: Dict[str, torch.Tensor],
                       max_norm: float) -> Dict[str, torch.Tensor]:
        """Clip gradients to maximum norm"""
        current_norm = self.compute_gradient_norm(gradients)

        if current_norm > max_norm:
            scale_factor = max_norm / current_norm
            clipped_gradients = {}

            for name, grad in gradients.items():
                if isinstance(grad, torch.Tensor):
                    clipped_gradients[name] = grad * scale_factor
                else:
                    clipped_gradients[name] = grad

            return clipped_gradients

        return gradients

    def add_noise(self, gradients: Dict[str, torch.Tensor],
                  noise_scale: float) -> Dict[str, torch.Tensor]:
        """Add Gaussian noise to gradients"""
        noisy_gradients = {}

        for name, grad in gradients.items():
            if isinstance(grad, torch.Tensor):
                noise = torch.randn_like(grad) * noise_scale
                noisy_gradients[name] = grad + noise.to(grad.device)
            else:
                noisy_gradients[name] = grad

        return noisy_gradients

    def update_stats(self, gradients_before: Dict[str, torch.Tensor],
                     gradients_after: Dict[str, torch.Tensor]):
        """Update defense statistics"""
        self.stats['total_applications'] += 1

        norm_before = self.compute_gradient_norm(gradients_before)
        norm_after = self.compute_gradient_norm(gradients_after)

        self.stats['gradient_norm_before'].append(norm_before)
        self.stats['gradient_norm_after'].append(norm_after)

        # Calculate utility loss as relative change in gradient norm
        if norm_before > 0:
            utility_loss = abs(norm_after - norm_before) / norm_before
            self.stats['utility_loss'] = utility_loss

    def get_defense_summary(self) -> Dict[str, Any]:
        """Get summary of defense performance"""
        if not self.stats['gradient_norm_before']:
            return {'error': 'No defense applications recorded'}

        return {
            'defense_type': self.__class__.__name__,
            'total_applications': self.stats['total_applications'],
            'config': self.defense_config,
            'avg_gradient_norm_before': np.mean(self.stats['gradient_norm_before']),
            'avg_gradient_norm_after': np.mean(self.stats['gradient_norm_after']),
            'avg_utility_loss': np.mean([self.stats['utility_loss']]) if self.stats['utility_loss'] else 0.0,
            'privacy_cost': self.get_privacy_cost(),
            'gradient_reduction_ratio': (
                np.mean(self.stats['gradient_norm_after']) /
                np.mean(self.stats['gradient_norm_before'])
                if self.stats['gradient_norm_before'] else 1.0
            )
        }

    def reset_stats(self):
        """Reset defense statistics"""
        self.stats = {
            'total_applications': 0,
            'gradient_norm_before': [],
            'gradient_norm_after': [],
            'privacy_cost': 0.0,
            'utility_loss': 0.0
        }

    def validate_gradients(self, gradients: Dict[str, torch.Tensor]) -> bool:
        """Validate gradient dictionary format"""
        if not isinstance(gradients, dict):
            return False

        for name, grad in gradients.items():
            if not isinstance(grad, torch.Tensor):
                self.logger.warning(f"Non-tensor gradient found: {name}")
                continue

            if torch.isnan(grad).any() or torch.isinf(grad).any():
                self.logger.warning(f"Invalid values in gradient: {name}")
                return False

        return True

    def apply_defense_with_validation(self, gradients: Dict[str, torch.Tensor],
                                      **kwargs) -> Dict[str, torch.Tensor]:
        """Apply defense with validation and statistics tracking"""
        if not self.defense_config['enabled']:
            return gradients

        # Validate input
        if not self.validate_gradients(gradients):
            self.logger.error("Invalid gradients provided to defense")
            return gradients

        # Store original gradients for statistics
        gradients_before = {k: v.clone() for k, v in gradients.items() if isinstance(v, torch.Tensor)}

        # Apply defense
        defended_gradients = self.apply_defense(gradients, **kwargs)

        # Validate output
        if not self.validate_gradients(defended_gradients):
            self.logger.error("Defense produced invalid gradients")
            return gradients

        # Update statistics
        self.update_stats(gradients_before, defended_gradients)

        return defended_gradients
