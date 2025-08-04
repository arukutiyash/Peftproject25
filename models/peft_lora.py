import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List
import math


class LoRALinear(nn.Module):
    """Linear layer with Low-Rank Adaptation"""

    def __init__(self, original_layer: nn.Linear, rank: int = 16, alpha: float = 16,
                 dropout: float = 0.0, malicious: bool = False):
        super().__init__()

        self.original_layer = original_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # LoRA matrices
        self.lora_A = nn.Parameter(torch.randn(rank, original_layer.in_features))
        self.lora_B = nn.Parameter(torch.zeros(original_layer.out_features, rank))

        self.dropout = nn.Dropout(dropout)

        # Initialize weights
        if malicious:
            self.init_malicious_weights()
        else:
            self.init_weights()

        # Freeze original layer
        for param in self.original_layer.parameters():
            param.requires_grad = False

    def init_weights(self):
        """Initialize LoRA weights"""
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def init_malicious_weights(self):
        """Initialize weights for gradient inversion attack"""
        with torch.no_grad():
            # Initialize A to create specific patterns
            nn.init.normal_(self.lora_A, mean=0, std=0.02)

            # Initialize B to create identity-like mapping when combined with A
            nn.init.zeros_(self.lora_B)
            min_dim = min(self.original_layer.out_features, self.rank)
            self.lora_B[:min_dim, :min_dim] = torch.eye(min_dim) * 0.1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with LoRA adaptation"""
        # Original transformation
        result = self.original_layer(x)

        # LoRA adaptation: x @ A^T @ B^T = x @ (BA)^T
        lora_out = self.dropout(x) @ self.lora_A.T @ self.lora_B.T

        # Apply scaling and add to original output
        result = result + lora_out * self.scaling

        return result

    def get_delta_weight(self):
        """Get the delta weight matrix for analysis"""
        return (self.lora_B @ self.lora_A) * self.scaling


class MaliciousLoRALinear(LoRALinear):
    """Malicious LoRA layer designed for gradient inversion attacks"""

    def __init__(self, original_layer: nn.Linear, rank: int = 16, alpha: float = 16,
                 position_encoding: Optional[torch.Tensor] = None):
        super().__init__(original_layer, rank, alpha, dropout=0.0, malicious=True)

        self.position_encoding = position_encoding
        if position_encoding is not None:
            self.incorporate_position_encoding()

    def incorporate_position_encoding(self):
        """Incorporate position encoding into LoRA matrices"""
        with torch.no_grad():
            if self.position_encoding is not None:
                # Use position encoding to bias the LoRA transformation
                min_rank = min(self.rank, len(self.position_encoding))
                self.lora_A[:min_rank, 0] = self.position_encoding[:min_rank]


class LoRAConfig:
    """Configuration class for LoRA"""

    def __init__(self,
                 rank: int = 16,
                 alpha: float = 16,
                 dropout: float = 0.0,
                 target_modules: List[str] = None,
                 malicious: bool = False):
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        self.target_modules = target_modules or ["qkv", "proj", "fc1", "fc2"]
        self.malicious = malicious


def add_lora_to_vit(model, config: LoRAConfig):
    """Add LoRA to Vision Transformer"""

    for i, block in enumerate(model.blocks):

        # Apply LoRA to attention layers
        if "qkv" in config.target_modules:
            if config.malicious:
                pos_encoding = model.pos_embed[0, i] if hasattr(model, 'pos_embed') else None
                block.attn.qkv = MaliciousLoRALinear(
                    original_layer=block.attn.qkv,
                    rank=config.rank,
                    alpha=config.alpha,
                    position_encoding=pos_encoding
                )
            else:
                block.attn.qkv = LoRALinear(
                    original_layer=block.attn.qkv,
                    rank=config.rank,
                    alpha=config.alpha,
                    dropout=config.dropout
                )

        if "proj" in config.target_modules:
            if config.malicious:
                pos_encoding = model.pos_embed[0, (i + model.embed_dim // 4) % model.pos_embed.size(1)] if hasattr(
                    model, 'pos_embed') else None
                block.attn.proj = MaliciousLoRALinear(
                    original_layer=block.attn.proj,
                    rank=config.rank,
                    alpha=config.alpha,
                    position_encoding=pos_encoding
                )
            else:
                block.attn.proj = LoRALinear(
                    original_layer=block.attn.proj,
                    rank=config.rank,
                    alpha=config.alpha,
                    dropout=config.dropout
                )

        # Apply LoRA to MLP layers
        if "fc1" in config.target_modules:
            if config.malicious:
                pos_encoding = model.pos_embed[0, (i + model.embed_dim // 2) % model.pos_embed.size(1)] if hasattr(
                    model, 'pos_embed') else None
                block.mlp.fc1 = MaliciousLoRALinear(
                    original_layer=block.mlp.fc1,
                    rank=config.rank,
                    alpha=config.alpha,
                    position_encoding=pos_encoding
                )
            else:
                block.mlp.fc1 = LoRALinear(
                    original_layer=block.mlp.fc1,
                    rank=config.rank,
                    alpha=config.alpha,
                    dropout=config.dropout
                )

        if "fc2" in config.target_modules:
            if config.malicious:
                pos_encoding = model.pos_embed[0, (i + 3 * model.embed_dim // 4) % model.pos_embed.size(1)] if hasattr(
                    model, 'pos_embed') else None
                block.mlp.fc2 = MaliciousLoRALinear(
                    original_layer=block.mlp.fc2,
                    rank=config.rank,
                    alpha=config.alpha,
                    position_encoding=pos_encoding
                )
            else:
                block.mlp.fc2 = LoRALinear(
                    original_layer=block.mlp.fc2,
                    rank=config.rank,
                    alpha=config.alpha,
                    dropout=config.dropout
                )

    # Ensure only LoRA parameters are trainable
    for name, param in model.named_parameters():
        if 'lora_' not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True

    return model


def get_lora_parameters(model):
    """Get all LoRA parameters for gradient computation"""
    lora_params = []
    for name, param in model.named_parameters():
        if 'lora_' in name and param.requires_grad:
            lora_params.append(param)
    return lora_params


def get_lora_gradients(model):
    """Extract LoRA gradients for attack analysis"""
    gradients = {}
    for name, param in model.named_parameters():
        if 'lora_' in name and param.grad is not None:
            gradients[name] = param.grad.clone()
    return gradients


def merge_lora_weights(model):
    """Merge LoRA weights into the original model (for inference)"""
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, LoRALinear):
                # Merge LoRA adaptation into original weights
                delta_w = module.get_delta_weight()
                if delta_w.shape == module.original_layer.weight.shape:
                    module.original_layer.weight.data += delta_w


def unmerge_lora_weights(model):
    """Unmerge LoRA weights from the original model"""
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, LoRALinear):
                # Remove LoRA adaptation from original weights
                delta_w = module.get_delta_weight()
                if delta_w.shape == module.original_layer.weight.shape:
                    module.original_layer.weight.data -= delta_w
