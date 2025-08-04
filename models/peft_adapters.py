import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any


class AdapterModule(nn.Module):
    """Bottleneck adapter module"""

    def __init__(self, input_dim: int, bottleneck_dim: int = 64, dropout: float = 0.0,
                 init_option: str = "bert", adapter_scalar: float = 1.0):
        super().__init__()

        self.input_dim = input_dim
        self.bottleneck_dim = bottleneck_dim
        self.adapter_scalar = adapter_scalar

        # Down projection
        self.down_proj = nn.Linear(input_dim, bottleneck_dim)

        # Activation function
        self.activation = nn.ReLU()

        # Up projection
        self.up_proj = nn.Linear(bottleneck_dim, input_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Initialize weights
        self.init_weights(init_option)

    def init_weights(self, option: str = "bert"):
        """Initialize adapter weights"""
        if option == "bert":
            # BERT-style initialization
            nn.init.normal_(self.down_proj.weight, std=0.02)
            nn.init.normal_(self.up_proj.weight, std=0.02)
            nn.init.zeros_(self.down_proj.bias)
            nn.init.zeros_(self.up_proj.bias)
        elif option == "xavier":
            nn.init.xavier_uniform_(self.down_proj.weight)
            nn.init.xavier_uniform_(self.up_proj.weight)
            nn.init.zeros_(self.down_proj.bias)
            nn.init.zeros_(self.up_proj.bias)
        elif option == "zero":
            # Initialize to zero for near-identity at start
            nn.init.zeros_(self.down_proj.weight)
            nn.init.zeros_(self.up_proj.weight)
            nn.init.zeros_(self.down_proj.bias)
            nn.init.zeros_(self.up_proj.bias)

    def forward(self, x: torch.Tensor, residual_input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with residual connection
        Args:
            x: Input tensor from the main model
            residual_input: Input for residual connection (typically same as x)
        """
        # Bottleneck transformation
        h = self.down_proj(x)
        h = self.activation(h)
        h = self.dropout(h)
        h = self.up_proj(h)

        # Apply adapter scaling and residual connection
        output = residual_input + self.adapter_scalar * h
        return output


class MaliciousAdapterModule(AdapterModule):
    """Malicious adapter designed for gradient inversion attacks"""

    def __init__(self, input_dim: int, bottleneck_dim: int = 64,
                 position_encoding: Optional[torch.Tensor] = None):
        super().__init__(input_dim, bottleneck_dim, dropout=0.0, init_option="zero")

        self.position_encoding = position_encoding

        # Override initialization for attack
        self.init_malicious_weights()

    def init_malicious_weights(self):
        """Initialize weights for gradient inversion attack"""
        # Set up projection as identity-like transformation
        with torch.no_grad():
            # Down projection: create specific patterns for patch recovery
            self.down_proj.weight.fill_(0.0)
            self.down_proj.bias.fill_(0.0)

            # Up projection: identity mapping scaled
            self.up_proj.weight.fill_(0.0)
            torch.nn.init.eye_(self.up_proj.weight[:self.input_dim, :min(self.bottleneck_dim, self.input_dim)])
            self.up_proj.bias.fill_(0.0)

            # If position encoding provided, use it for targeted recovery
            if self.position_encoding is not None:
                # Set biases to position-specific values
                if self.bottleneck_dim <= len(self.position_encoding):
                    self.down_proj.bias[:self.bottleneck_dim] = self.position_encoding[:self.bottleneck_dim]


class AdapterConfig:
    """Configuration class for adapters"""

    def __init__(self,
                 bottleneck_dim: int = 64,
                 dropout: float = 0.0,
                 init_option: str = "bert",
                 adapter_scalar: float = 1.0,
                 target_modules: list = None,
                 malicious: bool = False):
        self.bottleneck_dim = bottleneck_dim
        self.dropout = dropout
        self.init_option = init_option
        self.adapter_scalar = adapter_scalar
        self.target_modules = target_modules or ["attn", "mlp"]
        self.malicious = malicious


def add_adapters_to_vit(model, config: AdapterConfig):
    """Add adapter modules to Vision Transformer"""

    for i, block in enumerate(model.blocks):
        # Add adapter after attention
        if "attn" in config.target_modules:
            if config.malicious:
                # For attack scenarios
                pos_encoding = model.pos_embed[0, i % model.pos_embed.size(1)] if hasattr(model, 'pos_embed') else None
                block.attn_peft = MaliciousAdapterModule(
                    input_dim=model.embed_dim,
                    bottleneck_dim=config.bottleneck_dim,
                    position_encoding=pos_encoding
                )
            else:
                block.attn_peft = AdapterModule(
                    input_dim=model.embed_dim,
                    bottleneck_dim=config.bottleneck_dim,
                    dropout=config.dropout,
                    init_option=config.init_option,
                    adapter_scalar=config.adapter_scalar
                )

        # Add adapter after MLP
        if "mlp" in config.target_modules:
            if config.malicious:
                pos_encoding = model.pos_embed[0, (i + model.embed_dim // 2) % model.pos_embed.size(1)] if hasattr(
                    model, 'pos_embed') else None
                block.mlp_peft = MaliciousAdapterModule(
                    input_dim=model.embed_dim,
                    bottleneck_dim=config.bottleneck_dim,
                    position_encoding=pos_encoding
                )
            else:
                block.mlp_peft = AdapterModule(
                    input_dim=model.embed_dim,
                    bottleneck_dim=config.bottleneck_dim,
                    dropout=config.dropout,
                    init_option=config.init_option,
                    adapter_scalar=config.adapter_scalar
                )

    # Freeze backbone parameters, keep adapter parameters trainable
    for name, param in model.named_parameters():
        if 'peft' not in name.lower():
            param.requires_grad = False
        else:
            param.requires_grad = True

    return model


def get_adapter_parameters(model):
    """Get all adapter parameters for gradient computation"""
    adapter_params = []
    for name, param in model.named_parameters():
        if 'peft' in name.lower() and param.requires_grad:
            adapter_params.append(param)
    return adapter_params


def get_adapter_gradients(model):
    """Extract adapter gradients for attack analysis"""
    gradients = {}
    for name, param in model.named_parameters():
        if 'peft' in name.lower() and param.grad is not None:
            gradients[name] = param.grad.clone()
    return gradients
