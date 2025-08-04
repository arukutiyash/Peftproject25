import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from defenses.defense_base import BaseDefense #instahide_defense.py


class InstaHideDefense(BaseDefense):
    """InstaHide defense mechanism"""

    def __init__(self, device: str = 'cuda', k_mix: int = 4,
                 mask_ratio: float = 0.5, sign_flip_prob: float = 0.5, **kwargs):
        """
        Initialize InstaHide defense

        Args:
            device: Computing device
            k_mix: Number of images to mix (k parameter in InstaHide)
            mask_ratio: Ratio of pixels to mask
            sign_flip_prob: Probability of flipping signs in mask
            **kwargs: Additional defense parameters
        """
        super().__init__(device, **kwargs)

        self.k_mix = k_mix
        self.mask_ratio = mask_ratio
        self.sign_flip_prob = sign_flip_prob

        # Update defense config
        self.defense_config.update({
            'k_mix': k_mix,
            'mask_ratio': mask_ratio,
            'sign_flip_prob': sign_flip_prob,
            'defense_type': 'data_transformation'
        })

        # InstaHide uses a fixed set of public images for mixing
        self.public_images = None
        self._initialize_public_dataset()

    def _initialize_public_dataset(self):
        """Initialize public dataset for InstaHide mixing"""
        # Create synthetic public images (in practice, use real public dataset)
        self.public_images = torch.randn(100, 3, 32, 32, device=self.device)
        # Normalize to [0, 1] range
        self.public_images = torch.sigmoid(self.public_images)

    def generate_mixing_coefficients(self, batch_size: int) -> torch.Tensor:
        """Generate mixing coefficients for InstaHide"""
        # Sample k mixing coefficients that sum to 1
        coefficients = torch.zeros(batch_size, self.k_mix, device=self.device)

        for i in range(batch_size):
            # Generate random coefficients
            raw_coeffs = torch.rand(self.k_mix, device=self.device)
            # Normalize to sum to 1
            coefficients[i] = raw_coeffs / raw_coeffs.sum()

        return coefficients

    def generate_pixel_mask(self, image_shape: Tuple[int, ...]) -> torch.Tensor:
        """Generate binary mask for pixel-wise encryption"""
        mask = torch.bernoulli(torch.full(image_shape, self.mask_ratio, device=self.device))

        # Apply sign flips
        sign_flips = torch.bernoulli(torch.full(image_shape, self.sign_flip_prob, device=self.device))
        signs = 2 * sign_flips - 1  # Convert to {-1, 1}

        return mask * signs

    def select_mixing_images(self, batch_size: int) -> torch.Tensor:
        """Select images for mixing from public dataset"""
        mixing_images = torch.zeros(batch_size, self.k_mix - 1, 3, 32, 32, device=self.device)

        for i in range(batch_size):
            # Randomly select k-1 public images
            indices = torch.randperm(self.public_images.shape[0])[:self.k_mix - 1]
            mixing_images[i] = self.public_images[indices]

        return mixing_images

    def apply_instahide_encoding(self, private_images: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Apply InstaHide encoding to private images"""
        batch_size, channels, height, width = private_images.shape

        # Generate mixing coefficients
        mixing_coeffs = self.generate_mixing_coefficients(batch_size)

        # Select public images for mixing
        mixing_images = self.select_mixing_images(batch_size)

        # Generate pixel masks
        pixel_masks = torch.stack([
            self.generate_pixel_mask((channels, height, width))
            for _ in range(batch_size)
        ])

        # Apply InstaHide transformation
        encoded_images = torch.zeros_like(private_images)

        for i in range(batch_size):
            # Mix private image with public images
            mixed_image = mixing_coeffs[i, 0] * private_images[i]

            for k in range(1, self.k_mix):
                mixed_image += mixing_coeffs[i, k] * mixing_images[i, k - 1]

            # Apply pixel-wise mask
            encoded_images[i] = mixed_image * pixel_masks[i]

        # Store encoding parameters for potential decoding
        encoding_params = {
            'mixing_coefficients': mixing_coeffs,
            'mixing_images': mixing_images,
            'pixel_masks': pixel_masks
        }

        return encoded_images, encoding_params

    def apply_defense(self, gradients: Dict[str, torch.Tensor],
                      batch_data: Optional[torch.Tensor] = None,
                      model: Optional[nn.Module] = None,
                      **kwargs) -> Dict[str, torch.Tensor]:
        """
        Apply InstaHide defense by computing gradients on encoded data
        """
        if batch_data is None or model is None:
            self.logger.warning("InstaHide requires batch_data and model")
            return self._apply_gradient_noise(gradients)

        # Apply InstaHide encoding to input data
        encoded_data, encoding_params = self.apply_instahide_encoding(batch_data)

        # Compute gradients on encoded data
        encoded_gradients = self._compute_encoded_gradients(model, encoded_data, **kwargs)

        # Mix original and encoded gradients
        defended_gradients = self._combine_gradients(gradients, encoded_gradients)

        return defended_gradients

    def _compute_encoded_gradients(self, model: nn.Module, encoded_data: torch.Tensor,
                                   batch_labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Compute gradients on InstaHide encoded data"""
        model.train()
        model.zero_grad()

        # Forward pass on encoded data
        outputs, _ = model(encoded_data)

        # Compute loss (use dummy labels if not provided)
        if batch_labels is None:
            batch_labels = torch.randint(0, 100, (encoded_data.shape[0],), device=self.device)

        loss = nn.CrossEntropyLoss()(outputs, batch_labels)

        # Backward pass
        loss.backward()

        # Extract gradients
        encoded_gradients = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                encoded_gradients[name] = param.grad.clone().detach()

        return encoded_gradients

    def _combine_gradients(self, original_gradients: Dict[str, torch.Tensor],
                           encoded_gradients: Dict[str, torch.Tensor],
                           mixing_weight: float = 0.5) -> Dict[str, torch.Tensor]:
        """Combine original and encoded gradients"""
        combined_gradients = {}

        for name in original_gradients:
            if name in encoded_gradients:
                combined_gradients[name] = (
                        mixing_weight * original_gradients[name] +
                        (1 - mixing_weight) * encoded_gradients[name]
                )
            else:
                combined_gradients[name] = original_gradients[name]

        return combined_gradients

    def _apply_gradient_noise(self, gradients: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply noise-based defense as fallback"""
        noise_scale = 0.1 * self.mask_ratio  # Scale noise with mask ratio
        return self.add_noise(gradients, noise_scale)

    def get_privacy_cost(self) -> float:
        """Estimate privacy cost of InstaHide"""
        # Privacy increases with more mixing and higher mask ratio
        mixing_factor = 1.0 / self.k_mix  # More mixing = lower cost per image
        mask_factor = self.mask_ratio  # More masking = higher privacy

        return mixing_factor * mask_factor * 2.0  # Scaling factor

    def analyze_encoding_strength(self, original_data: torch.Tensor,
                                  encoded_data: torch.Tensor) -> Dict[str, float]:
        """Analyze the strength of InstaHide encoding"""
        # Compute correlation between original and encoded data
        original_flat = original_data.flatten()
        encoded_flat = encoded_data.flatten()

        correlation = torch.corrcoef(torch.stack([original_flat, encoded_flat]))[0, 1].item()

        # Compute mutual information proxy
        mse = nn.MSELoss()(original_data, encoded_data).item()

        # Compute entropy of encoded data
        encoded_normalized = (encoded_data - encoded_data.mean()) / encoded_data.std()
        entropy_proxy = -torch.mean(encoded_normalized * torch.log(torch.abs(encoded_normalized) + 1e-8)).item()

        return {
            'correlation': abs(correlation),
            'mse': mse,
            'entropy_proxy': entropy_proxy,
            'privacy_score': 1.0 - abs(correlation)  # Higher is better
        }


class AdaptiveInstaHideDefense(InstaHideDefense):
    """Adaptive InstaHide that adjusts parameters based on attack detection"""

    def __init__(self, device: str = 'cuda', **kwargs):
        super().__init__(device, **kwargs)

        self.attack_detected = False
        self.gradient_anomaly_threshold = 2.0
        self.recent_gradient_norms = []

    def detect_attack(self, gradients: Dict[str, torch.Tensor]) -> bool:
        """Simple attack detection based on gradient anomalies"""
        current_norm = self.compute_gradient_norm(gradients)
        self.recent_gradient_norms.append(current_norm)

        # Keep only recent history
        if len(self.recent_gradient_norms) > 20:
            self.recent_gradient_norms.pop(0)

        if len(self.recent_gradient_norms) > 5:
            mean_norm = np.mean(self.recent_gradient_norms[:-1])
            std_norm = np.std(self.recent_gradient_norms[:-1])

            # Check if current norm is anomalous
            if std_norm > 0:
                z_score = abs(current_norm - mean_norm) / std_norm
                return z_score > self.gradient_anomaly_threshold

        return False

    def adapt_parameters(self):
        """Adapt InstaHide parameters based on attack detection"""
        if self.attack_detected:
            # Increase privacy when attack detected
            self.k_mix = min(8, self.k_mix + 1)
            self.mask_ratio = min(0.8, self.mask_ratio + 0.1)
            self.sign_flip_prob = min(0.8, self.sign_flip_prob + 0.1)
        else:
            # Gradually reduce privacy overhead
            self.k_mix = max(2, self.k_mix - 0.1)
            self.mask_ratio = max(0.3, self.mask_ratio - 0.05)
            self.sign_flip_prob = max(0.3, self.sign_flip_prob - 0.05)

    def apply_defense(self, gradients: Dict[str, torch.Tensor], **kwargs) -> Dict[str, torch.Tensor]:
        """Apply adaptive InstaHide defense"""
        # Detect potential attacks
        self.attack_detected = self.detect_attack(gradients)

        # Adapt parameters
        self.adapt_parameters()

        # Apply defense with adapted parameters
        return super().apply_defense(gradients, **kwargs)
