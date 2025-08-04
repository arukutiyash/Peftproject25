import torch
import torch.nn as nn
from typing import Optional, Dict


class BiasModule(nn.Module):
    """Learnable bias module for bias-tuning"""

    def __init__(self, dim: int, init_value: float = 0.0):
        super().__init__()
        self.bias = nn.Parameter(torch.full((dim,), init_value))

    def forward(self, x):
        return x + self.bias


class MaliciousBiasModule(BiasModule):
    """Malicious bias module for gradient inversion attacks"""

    def __init__(self, dim: int, position_encoding: Optional[torch.Tensor] = None):
        super().__init__(dim, init_value=0.0)

        if position_encoding is not None:
            with torch.no_grad():
                min_dim = min(dim, len(position_encoding))
                self.bias[:min_dim] = position_encoding[:min_dim]


class BiasConfig:
    """Configuration class for bias-tuning"""

    def __init__(self,
                 target_modules: list = None,
                 init_value: float = 0.0,
                 malicious: bool = False):
        self.target_modules = target_modules or ["linear", "layernorm"]
        self.init_value = init_value
        self.malicious = malicious


def add_bias_tuning_to_vit(model, config: BiasConfig):
    """Add bias-tuning to Vision Transformer"""

    # Add bias modules to different components
    for i, block in enumerate(model.blocks):

        # Add bias to attention components
        if "linear" in config.target_modules:
            # QKV projection bias
            if config.malicious:
                pos_encoding = model.pos_embed[0, i] if hasattr(model, 'pos_embed') else None
                block.attn_qkv_bias = MaliciousBiasModule(
                    dim=block.attn.qkv.out_features,
                    position_encoding=pos_encoding
                )
            else:
                block.attn_qkv_bias = BiasModule(
                    dim=block.attn.qkv.out_features,
                    init_value=config.init_value
                )

            # Attention projection bias
            if config.malicious:
                pos_encoding = model.pos_embed[0, (i + model.embed_dim // 4) % model.pos_embed.size(1)] if hasattr(
                    model, 'pos_embed') else None
                block.attn_proj_bias = MaliciousBiasModule(
                    dim=block.attn.proj.out_features,
                    position_encoding=pos_encoding
                )
            else:
                block.attn_proj_bias = BiasModule(
                    dim=block.attn.proj.out_features,
                    init_value=config.init_value
                )

            # MLP biases
            if config.malicious:
                pos_encoding = model.pos_embed[0, (i + model.embed_dim // 2) % model.pos_embed.size(1)] if hasattr(
                    model, 'pos_embed') else None
                block.mlp_fc1_bias = MaliciousBiasModule(
                    dim=block.mlp.fc1.out_features,
                    position_encoding=pos_encoding
                )
                block.mlp_fc2_bias = MaliciousBiasModule(
                    dim=block.mlp.fc2.out_features,
                    position_encoding=pos_encoding
                )
            else:
                block.mlp_fc1_bias = BiasModule(
                    dim=block.mlp.fc1.out_features,
                    init_value=config.init_value
                )
                block.mlp_fc2_bias = BiasModule(
                    dim=block.mlp.fc2.out_features,
                    init_value=config.init_value
                )

        # Add bias to layer norms
        if "layernorm" in config.target_modules:
            if config.malicious:
                pos_encoding = model.pos_embed[0, (i + model.embed_dim // 3) % model.pos_embed.size(1)] if hasattr(
                    model, 'pos_embed') else None
                block.norm1_bias = MaliciousBiasModule(
                    dim=model.embed_dim,
                    position_encoding=pos_encoding
                )
                block.norm2_bias = MaliciousBiasModule(
                    dim=model.embed_dim,
                    position_encoding=pos_encoding
                )
            else:
                block.norm1_bias = BiasModule(
                    dim=model.embed_dim,
                    init_value=config.init_value
                )
                block.norm2_bias = BiasModule(
                    dim=model.embed_dim,
                    init_value=config.init_value
                )

        # Modify forward pass to include bias modules
        _add_bias_forward_hooks(block)

    # Freeze all parameters except bias modules
    for name, param in model.named_parameters():
        if 'bias' not in name.lower() or name.endswith('.bias'):  # Skip original biases
            param.requires_grad = False
        else:
            param.requires_grad = True

    # Ensure bias module parameters are trainable
    for name, param in model.named_parameters():
        if any(bias_name in name for bias_name in ['attn_qkv_bias', 'attn_proj_bias',
                                                   'mlp_fc1_bias', 'mlp_fc2_bias',
                                                   'norm1_bias', 'norm2_bias']):
            param.requires_grad = True

    return model


def _add_bias_forward_hooks(block):
    """Add forward hooks to apply bias modules"""

    def attn_forward_hook(module, input, output):
        """Hook for attention module"""
        attn_out, attn_weights = output

        # Apply bias if available
        if hasattr(block, 'attn_proj_bias'):
            attn_out = block.attn_proj_bias(attn_out)

        return attn_out, attn_weights

    def mlp_forward_hook(module, input, output):
        """Hook for MLP module"""
        mlp_out = output

        # Apply bias if available
        if hasattr(block, 'mlp_fc2_bias'):
            mlp_out = block.mlp_fc2_bias(mlp_out)

        return mlp_out

    def norm_forward_hook(norm_module, bias_attr):
        """Create hook for layer norm with bias"""

        def hook(module, input, output):
            if hasattr(block, bias_attr):
                bias_module = getattr(block, bias_attr)
                return bias_module(output)
            return output

        return hook

    # Register hooks
    block.attn.register_forward_hook(attn_forward_hook)
    block.mlp.register_forward_hook(mlp_forward_hook)

    if hasattr(block, 'norm1_bias'):
        block.norm1.register_forward_hook(norm_forward_hook(block.norm1, 'norm1_bias'))
    if hasattr(block, 'norm2_bias'):
        block.norm2.register_forward_hook(norm_forward_hook(block.norm2, 'norm2_bias'))


def get_bias_parameters(model):
    """Get all bias tuning parameters for gradient computation"""
    bias_params = []
    for name, param in model.named_parameters():
        if any(bias_name in name for bias_name in ['attn_qkv_bias', 'attn_proj_bias',
                                                   'mlp_fc1_bias', 'mlp_fc2_bias',
                                                   'norm1_bias', 'norm2_bias']) and param.requires_grad:
            bias_params.append(param)
    return bias_params


def get_bias_gradients(model):
    """Extract bias gradients for attack analysis"""
    gradients = {}
    for name, param in model.named_parameters():
        if any(bias_name in name for bias_name in ['attn_qkv_bias', 'attn_proj_bias',
                                                   'mlp_fc1_bias', 'mlp_fc2_bias',
                                                   'norm1_bias', 'norm2_bias']) and param.grad is not None:
            gradients[name] = param.grad.clone()
    return gradients
