import torch
import torch.nn as nn
import torch.nn.functional as F  # <- ADD THIS LINE
from typing import Optional, Dict, Tuple


class PrefixEncoder(nn.Module):
    """Prefix encoder for generating prefix embeddings"""

    def __init__(self, prefix_length: int, embed_dim: int, hidden_dim: Optional[int] = None):
        super().__init__()
        self.prefix_length = prefix_length
        self.embed_dim = embed_dim

        if hidden_dim is None:
            hidden_dim = embed_dim

        # Two-layer MLP for prefix generation
        self.prefix_mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, embed_dim * 2)  # For key and value
        )

        # Learnable prefix embeddings
        self.prefix_embeddings = nn.Parameter(torch.randn(prefix_length, embed_dim))

    def forward(self):
        """Generate prefix key-value pairs"""
        # Get prefix embeddings
        prefix_embeds = self.prefix_embeddings  # (prefix_length, embed_dim)

        # Generate key-value pairs
        kv = self.prefix_mlp(prefix_embeds)  # (prefix_length, embed_dim * 2)
        key, value = kv.chunk(2, dim=-1)  # Each: (prefix_length, embed_dim)

        return key, value


class PrefixAttention(nn.Module):
    """Multi-head attention with prefix-tuning"""

    def __init__(self, embed_dim: int = 768, n_heads: int = 12, prefix_length: int = 10,
                 qkv_bias: bool = True, attn_drop: float = 0., proj_drop: float = 0.):
        super().__init__()
        assert embed_dim % n_heads == 0

        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.scale = self.head_dim ** -0.5
        self.prefix_length = prefix_length

        # Standard attention components
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # Prefix encoder
        self.prefix_encoder = PrefixEncoder(prefix_length, embed_dim)

    def forward(self, x):
        B, N, C = x.shape

        # Generate Q, K, V from input
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each: (B, n_heads, N, head_dim)

        # Get prefix key-value pairs
        prefix_k, prefix_v = self.prefix_encoder()  # Each: (prefix_length, embed_dim)
        prefix_k = prefix_k.unsqueeze(0).repeat(B, 1, 1)  # (B, prefix_length, embed_dim)
        prefix_v = prefix_v.unsqueeze(0).repeat(B, 1, 1)  # (B, prefix_length, embed_dim)

        # Reshape prefix for multi-head attention
        prefix_k = prefix_k.reshape(B, self.prefix_length, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        prefix_v = prefix_v.reshape(B, self.prefix_length, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        # Concatenate prefix with regular key-value pairs
        k = torch.cat([prefix_k, k], dim=2)  # (B, n_heads, prefix_length + N, head_dim)
        v = torch.cat([prefix_v, v], dim=2)  # (B, n_heads, prefix_length + N, head_dim)

        # Attention computation
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x, attn


class MaliciousPrefixEncoder(PrefixEncoder):
    """Malicious prefix encoder for gradient inversion attacks"""

    def __init__(self, prefix_length: int, embed_dim: int,
                 position_encodings: Optional[torch.Tensor] = None):
        super().__init__(prefix_length, embed_dim)

        self.position_encodings = position_encodings
        self.init_malicious_weights()

    def init_malicious_weights(self):
        """Initialize weights for gradient inversion attack"""
        with torch.no_grad():
            # Set prefix embeddings to specific patterns for patch recovery
            if self.position_encodings is not None:
                min_len = min(self.prefix_length, len(self.position_encodings))
                self.prefix_embeddings[:min_len] = self.position_encodings[:min_len]

            # Set MLP to near-identity transformation
            for layer in self.prefix_mlp:
                if isinstance(layer, nn.Linear):
                    nn.init.eye_(layer.weight[:min(layer.in_features, layer.out_features),
                                 :min(layer.in_features, layer.out_features)])
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)


class PrefixConfig:
    """Configuration class for prefix-tuning"""

    def __init__(self,
                 prefix_length: int = 10,
                 hidden_dim: Optional[int] = None,
                 target_modules: list = None,
                 malicious: bool = False):
        self.prefix_length = prefix_length
        self.hidden_dim = hidden_dim
        self.target_modules = target_modules or ["attn"]
        self.malicious = malicious


def add_prefix_to_vit(model, config: PrefixConfig):
    """Add prefix-tuning to Vision Transformer"""

    for i, block in enumerate(model.blocks):
        if "attn" in config.target_modules:
            # Replace attention with prefix attention
            original_attn = block.attn

            if config.malicious:
                # Create malicious prefix attention for attacks
                pos_encodings = model.pos_embed[0, :config.prefix_length] if hasattr(model, 'pos_embed') else None
                prefix_encoder = MaliciousPrefixEncoder(
                    prefix_length=config.prefix_length,
                    embed_dim=model.embed_dim,
                    position_encodings=pos_encodings
                )
            else:
                prefix_encoder = PrefixEncoder(
                    prefix_length=config.prefix_length,
                    embed_dim=model.embed_dim,
                    hidden_dim=config.hidden_dim
                )

            # Create new prefix attention
            prefix_attn = PrefixAttention(
                embed_dim=model.embed_dim,
                n_heads=original_attn.n_heads,
                prefix_length=config.prefix_length,
                qkv_bias=True,
                attn_drop=0.0,
                proj_drop=0.0
            )

            # Replace encoder in prefix attention
            prefix_attn.prefix_encoder = prefix_encoder

            # Copy original attention weights
            prefix_attn.qkv.weight.data = original_attn.qkv.weight.data.clone()
            prefix_attn.qkv.bias.data = original_attn.qkv.bias.data.clone()
            prefix_attn.proj.weight.data = original_attn.proj.weight.data.clone()
            prefix_attn.proj.bias.data = original_attn.proj.bias.data.clone()

            block.attn = prefix_attn

    # Freeze backbone parameters, keep prefix parameters trainable
    for name, param in model.named_parameters():
        if 'prefix' not in name.lower():
            param.requires_grad = False
        else:
            param.requires_grad = True

    return model


def get_prefix_parameters(model):
    """Get all prefix parameters for gradient computation"""
    prefix_params = []
    for name, param in model.named_parameters():
        if 'prefix' in name.lower() and param.requires_grad:
            prefix_params.append(param)
    return prefix_params


def get_prefix_gradients(model):
    """Extract prefix gradients for attack analysis"""
    gradients = {}
    for name, param in model.named_parameters():
        if 'prefix' in name.lower() and param.grad is not None:
            gradients[name] = param.grad.clone()
    return gradients
