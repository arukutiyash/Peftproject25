import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class PatchEmbedding(nn.Module):
    """Convert image to patch embeddings"""

    def __init__(self, img_size: int = 32, patch_size: int = 4, in_channels: int = 3, embed_dim: int = 768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim

        # Linear projection layer
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (batch_size, channels, height, width)
        B, C, H, W = x.shape
        assert H == self.img_size and W == self.img_size, f"Input image size ({H}*{W}) doesn't match model ({self.img_size}*{self.img_size})"

        x = self.proj(x)  # (B, embed_dim, H//patch_size, W//patch_size)
        x = x.flatten(2)  # (B, embed_dim, n_patches)
        x = x.transpose(1, 2)  # (B, n_patches, embed_dim)
        return x


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism"""

    def __init__(self, embed_dim: int = 768, n_heads: int = 12, qkv_bias: bool = True, attn_drop: float = 0.,
                 proj_drop: float = 0.):
        super().__init__()
        assert embed_dim % n_heads == 0

        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape

        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each: (B, n_heads, N, head_dim)

        # Attention computation
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x, attn


class MLP(nn.Module):
    """Multi-layer perceptron"""

    def __init__(self, in_features: int, hidden_features: Optional[int] = None, out_features: Optional[int] = None,
                 act_layer=nn.GELU, drop: float = 0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer encoder block with PEFT support"""

    def __init__(self, embed_dim: int = 768, n_heads: int = 12, mlp_ratio: float = 4.0,
                 qkv_bias: bool = True, drop: float = 0., attn_drop: float = 0.,
                 peft_config: Optional[dict] = None):
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, n_heads, qkv_bias, attn_drop, drop)

        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = MLP(embed_dim, mlp_hidden_dim, act_layer=nn.GELU, drop=drop)

        # PEFT modules will be added by specific PEFT classes
        self.peft_config = peft_config
        self.attn_peft = None
        self.mlp_peft = None

    def forward(self, x):
        # Pre-norm architecture
        attn_out, attn_weights = self.attn(self.norm1(x))

        # Add PEFT adaptation after attention if available
        if self.attn_peft is not None:
            attn_out = self.attn_peft(attn_out, self.norm1(x))

        x = x + attn_out

        mlp_out = self.mlp(self.norm2(x))

        # Add PEFT adaptation after MLP if available
        if self.mlp_peft is not None:
            mlp_out = self.mlp_peft(mlp_out, self.norm2(x))

        x = x + mlp_out

        return x, attn_weights


class VisionTransformer(nn.Module):
    """Vision Transformer with PEFT support"""

    def __init__(self, img_size: int = 32, patch_size: int = 4, in_channels: int = 3,
                 num_classes: int = 100, embed_dim: int = 768, depth: int = 12,
                 n_heads: int = 12, mlp_ratio: float = 4.0, qkv_bias: bool = True,
                 drop_rate: float = 0., attn_drop_rate: float = 0.,
                 peft_config: Optional[dict] = None):
        super().__init__()

        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.n_patches

        # Positional embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim,
                n_heads=n_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                peft_config=peft_config
            )
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        # Classification head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights"""
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        """Forward pass through feature extraction"""
        B = x.shape[0]

        # Patch embedding
        x = self.patch_embed(x)  # (B, n_patches, embed_dim)

        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, embed_dim)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, n_patches + 1, embed_dim)

        # Add positional embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Through transformer blocks
        attention_weights = []
        for block in self.blocks:
            x, attn_weights = block(x)
            attention_weights.append(attn_weights)

        x = self.norm(x)
        return x, attention_weights

    def forward(self, x):
        """Complete forward pass"""
        x, attention_weights = self.forward_features(x)
        x = self.head(x[:, 0])  # Use class token for classification
        return x, attention_weights

    def get_patch_embeddings(self, x):
        """Get patch embeddings for attack analysis"""
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        return x

    def freeze_backbone(self):
        """Freeze all parameters except PEFT modules"""
        for name, param in self.named_parameters():
            if 'peft' not in name.lower():
                param.requires_grad = False

    def unfreeze_all(self):
        """Unfreeze all parameters"""
        for param in self.parameters():
            param.requires_grad = True


def create_vit_base_patch4_32(num_classes: int = 100, peft_config: Optional[dict] = None):
    """Create ViT-Base model optimized for CIFAR-100 (32x32 images)"""
    model = VisionTransformer(
        img_size=32,
        patch_size=4,
        in_channels=3,
        num_classes=num_classes,
        embed_dim=768,
        depth=12,
        n_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        peft_config=peft_config
    )
    return model


def create_vit_small_patch4_32(num_classes: int = 100, peft_config: Optional[dict] = None):
    """Create ViT-Small model for faster experimentation"""
    model = VisionTransformer(
        img_size=32,
        patch_size=4,
        in_channels=3,
        num_classes=num_classes,
        embed_dim=384,
        depth=12,
        n_heads=6,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        peft_config=peft_config
    )
    return model
