import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np #mixup_defense.py
from typing import Dict, List, Tuple, Optional, Any
from defenses.defense_base import BaseDefense


class MixUpDefense(BaseDefense):
    """MixUp defense for gradient inversion attacks"""

    def __init__(self, device: str = 'cuda', alpha: float = 1.0,
                 mixup_prob: float = 0.5, **kwargs):
        """
        Initialize MixUp defense

        Args:
            device: Computing device
            alpha: Beta distribution parameter for mixup coefficient
            mixup_prob: Probability of applying mixup to a batch
            **kwargs: Additional defense parameters
        """
        super().__init__(device, **kwargs)

        self.alpha = alpha
        self.mixup_prob = mixup_prob

        # Update defense config
        self.defense_config.update({
            'alpha': alpha,
            'mixup_prob': mixup_prob,
            'defense_type': 'data_augmentation'
        })

    def generate_mixup_parameters(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate mixup parameters for a batch"""
        if np.random.random() > self.mixup_prob:
            # No mixup applied
            return torch.ones(batch_size, device=self.device), torch.arange(batch_size, device=self.device)

        # Sample mixing coefficient from Beta distribution
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1.0

        # Generate random permutation for mixing
        batch_indices = torch.randperm(batch_size, device=self.device)

        # Broadcast lambda to batch size
        lam_tensor = torch.full((batch_size,), lam, device=self.device)

        return lam_tensor, batch_indices

    def mixup_data(self, x: torch.Tensor, y: torch.Tensor,
                   lam: torch.Tensor, indices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply mixup to input data and labels"""
        # Mix inputs: x_mixed = lam * x + (1 - lam) * x[indices]
        lam_expanded = lam.view(-1, 1, 1, 1)  # Expand for broadcasting
        mixed_x = lam_expanded * x + (1 - lam_expanded) * x[indices]

        # For labels, we'll return both original and mixed labels with lambda
        y_mixed = y[indices]

        return mixed_x, y, y_mixed

    def mixup_loss(self, pred: torch.Tensor, y_a: torch.Tensor, y_b: torch.Tensor,
                   lam: torch.Tensor, criterion=nn.CrossEntropyLoss()) -> torch.Tensor:
        """Compute mixup loss"""
        lam_mean = lam.mean()  # Average lambda across batch
        return lam_mean * criterion(pred, y_a) + (1 - lam_mean) * criterion(pred, y_b)

    def apply_defense(self, gradients: Dict[str, torch.Tensor],
                      batch_data: Optional[torch.Tensor] = None,
                      batch_labels: Optional[torch.Tensor] = None,
                      model: Optional[nn.Module] = None,
                      **kwargs) -> Dict[str, torch.Tensor]:
        """
        Apply MixUp defense by modifying gradients based on mixed data

        Note: MixUp is typically applied during training, but here we simulate
        the effect on gradients for defense purposes
        """
        if batch_data is None or model is None:
            self.logger.warning("MixUp requires batch_data and model - returning original gradients")
            return gradients

        batch_size = batch_data.shape[0]

        # Generate mixup parameters
        lam, indices = self.generate_mixup_parameters(batch_size)

        # Apply mixup to data
        if batch_labels is not None:
            mixed_data, labels_a, labels_b = self.mixup_data(batch_data, batch_labels, lam, indices)
        else:
            # Create dummy labels if not provided
            dummy_labels = torch.randint(0, 100, (batch_size,), device=self.device)
            mixed_data, labels_a, labels_b = self.mixup_data(batch_data, dummy_labels, lam, indices)

        # Compute gradients on mixed data
        mixed_gradients = self._compute_mixed_gradients(model, mixed_data, labels_a, labels_b, lam)

        # Combine with original gradients
        defended_gradients = self._combine_gradients(gradients, mixed_gradients, lam.mean().item())

        return defended_gradients

    def _compute_mixed_gradients(self, model: nn.Module, mixed_data: torch.Tensor,
                                 labels_a: torch.Tensor, labels_b: torch.Tensor,
                                 lam: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute gradients on mixed data"""
        model.train()
        model.zero_grad()

        # Forward pass on mixed data
        outputs, _ = model(mixed_data)

        # Compute mixup loss
        loss = self.mixup_loss(outputs, labels_a, labels_b, lam)

        # Backward pass
        loss.backward()

        # Extract gradients
        mixed_gradients = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                mixed_gradients[name] = param.grad.clone().detach()

        return mixed_gradients

    def _combine_gradients(self, original_gradients: Dict[str, torch.Tensor],
                           mixed_gradients: Dict[str, torch.Tensor],
                           mixing_weight: float) -> Dict[str, torch.Tensor]:
        """Combine original and mixed gradients"""
        combined_gradients = {}

        for name in original_gradients:
            if name in mixed_gradients:
                # Weighted combination of gradients
                combined_gradients[name] = (
                        mixing_weight * original_gradients[name] +
                        (1 - mixing_weight) * mixed_gradients[name]
                )
            else:
                combined_gradients[name] = original_gradients[name]

        return combined_gradients

    def get_privacy_cost(self) -> float:
        """
        Estimate privacy cost of MixUp
        MixUp provides implicit privacy but no formal guarantees
        """
        # Heuristic: privacy increases with more mixing (lower alpha)
        if self.alpha > 0:
            mixing_entropy = -self.mixup_prob * np.log(self.mixup_prob) if self.mixup_prob > 0 else 0
            return max(0.1, 1.0 / self.alpha) * mixing_entropy
        return 0.5

    def analyze_mixing_effect(self, gradients_before: Dict[str, torch.Tensor],
                              gradients_after: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Analyze the effect of mixing on gradients"""
        analysis = {}

        for name in gradients_before:
            if name in gradients_after:
                before = gradients_before[name]
                after = gradients_after[name]

                # Compute similarity metrics
                cosine_sim = F.cosine_similarity(before.flatten().unsqueeze(0),
                                                 after.flatten().unsqueeze(0)).item()

                norm_ratio = torch.norm(after).item() / (torch.norm(before).item() + 1e-8)

                mse = F.mse_loss(before, after).item()

                analysis[name] = {
                    'cosine_similarity': cosine_sim,
                    'norm_ratio': norm_ratio,
                    'mse': mse
                }

        return analysis


class AdaptiveMixUpDefense(MixUpDefense):
    """Adaptive MixUp that adjusts mixing based on gradient patterns"""

    def __init__(self, device: str = 'cuda', initial_alpha: float = 1.0,
                 adaptation_rate: float = 0.1, **kwargs):
        super().__init__(device, alpha=initial_alpha, **kwargs)

        self.initial_alpha = initial_alpha
        self.adaptation_rate = adaptation_rate
        self.gradient_history = []

    def adapt_alpha(self, gradients: Dict[str, torch.Tensor]):
        """Adapt alpha based on gradient patterns"""
        current_norm = self.compute_gradient_norm(gradients)
        self.gradient_history.append(current_norm)

        # Keep only recent history
        if len(self.gradient_history) > 10:
            self.gradient_history.pop(0)

        if len(self.gradient_history) > 1:
            # Increase mixing (decrease alpha) if gradients are stable
            gradient_variance = np.var(self.gradient_history)

            if gradient_variance < 0.1:  # Stable gradients
                self.alpha = max(0.1, self.alpha - self.adaptation_rate)
            else:  # Unstable gradients
                self.alpha = min(2.0, self.alpha + self.adaptation_rate)

    def apply_defense(self, gradients: Dict[str, torch.Tensor], **kwargs) -> Dict[str, torch.Tensor]:
        """Apply adaptive MixUp defense"""
        self.adapt_alpha(gradients)
        return super().apply_defense(gradients, **kwargs)
