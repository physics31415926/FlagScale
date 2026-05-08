# Copyright (c) 2025, BAAI. All rights reserved.
# SigLIP Vision Transformer for BAGEL, with flash attention and 2D RoPE.
# Adapted from original BAGEL siglip_navit.py - kept as non-TP module (replicated).

import torch
from torch import nn
from transformers.activations import ACT2FN
from flash_attn import flash_attn_varlen_func


class SiglipVisionConfig:
    """Simple config holder for SigLIP ViT."""

    def __init__(
        self,
        hidden_size=1152,
        intermediate_size=4304,
        num_hidden_layers=27,
        num_attention_heads=16,
        num_channels=3,
        image_size=980,
        patch_size=14,
        hidden_act="gelu_pytorch_tanh",
        layer_norm_eps=1e-6,
        attention_dropout=0.0,
        rope=True,
    ):
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_act = hidden_act
        self.layer_norm_eps = layer_norm_eps
        self.attention_dropout = attention_dropout
        self.rope = rope


class RotaryEmbedding2D(nn.Module):
    def __init__(self, dim, max_h, max_w, base=10000):
        super().__init__()
        freq = torch.arange(0, dim, 2, dtype=torch.int64).float() / dim
        inv_freq = 1.0 / (base**freq)
        grid_h = torch.arange(0, max_h).to(inv_freq.dtype)[:, None].repeat(1, max_w)
        grid_w = torch.arange(0, max_w).to(inv_freq.dtype)[None, :].repeat(max_h, 1)
        cos_h, sin_h = self._forward_one_side(grid_h, inv_freq)
        cos_w, sin_w = self._forward_one_side(grid_w, inv_freq)
        self.register_buffer("cos_h", cos_h)
        self.register_buffer("sin_h", sin_h)
        self.register_buffer("cos_w", cos_w)
        self.register_buffer("sin_w", sin_w)

    def _forward_one_side(self, grid, inv_freq):
        freqs = grid[..., None] * inv_freq[None, None, :]
        emb = torch.cat((freqs, freqs), dim=-1).flatten(0, 1)
        return emb.cos(), emb.sin()


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_vit(q, k, cos, sin):
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class SiglipVisionEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.patch_size = config.patch_size
        # Use Linear instead of Conv2d for patchified input
        self.patch_embedding = nn.Linear(
            config.num_channels * config.patch_size**2, self.embed_dim, bias=True
        )
        # Legacy position embedding (exists in checkpoint, not used with RoPE)
        num_patches = (config.image_size // config.patch_size) ** 2
        self.position_embedding = nn.Embedding(num_patches, self.embed_dim)

    def forward(self, packed_pixel_values, packed_flattened_position_ids):
        # Cast input to model dtype (bf16 if --bf16 is set)
        packed_pixel_values = packed_pixel_values.to(self.patch_embedding.weight.dtype)
        return self.patch_embedding(packed_pixel_values)


class SiglipFlashAttention2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.config = config
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, hidden_states, cu_seqlens, max_seqlen, cos_h=None, sin_h=None, cos_w=None, sin_w=None):
        total_q_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states).view(total_q_len, self.num_heads, self.head_dim)
        key_states = self.k_proj(hidden_states).view(total_q_len, self.num_heads, self.head_dim)
        value_states = self.v_proj(hidden_states).view(total_q_len, self.num_heads, self.head_dim)

        if self.config.rope and cos_h is not None:
            qh, qw = query_states[:, :, : self.head_dim // 2], query_states[:, :, self.head_dim // 2 :]
            kh, kw = key_states[:, :, : self.head_dim // 2], key_states[:, :, self.head_dim // 2 :]
            qh, kh = apply_rotary_pos_emb_vit(qh, kh, cos_h, sin_h)
            qw, kw = apply_rotary_pos_emb_vit(qw, kw, cos_w, sin_w)
            query_states = torch.cat([qh, qw], dim=-1)
            key_states = torch.cat([kh, kw], dim=-1)

        attn_output = flash_attn_varlen_func(
            query_states.to(torch.bfloat16),
            key_states.to(torch.bfloat16),
            value_states.to(torch.bfloat16),
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            causal=False,
        )
        return self.out_proj(attn_output.reshape(total_q_len, -1))


class SiglipMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states):
        return self.fc2(self.activation_fn(self.fc1(hidden_states)))


class SiglipEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = SiglipFlashAttention2(config)
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = SiglipMLP(config)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states, cu_seqlens, max_seqlen, cos_h=None, sin_h=None, cos_w=None, sin_w=None):
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn(hidden_states, cu_seqlens, max_seqlen, cos_h, sin_h, cos_w, sin_w)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class SiglipVisionTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embeddings = SiglipVisionEmbeddings(config)
        if config.rope:
            max_size = config.image_size // config.patch_size
            dim_head = config.hidden_size // config.num_attention_heads
            self.rope = RotaryEmbedding2D(dim_head // 2, max_size, max_size)
        self.encoder = nn.Module()
        self.encoder.layers = nn.ModuleList([SiglipEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.post_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, packed_pixel_values, packed_flattened_position_ids, cu_seqlens, max_seqlen):
        hidden_states = self.embeddings(packed_pixel_values, packed_flattened_position_ids)
        extra = {}
        if self.config.rope:
            extra = dict(
                cos_h=self.rope.cos_h[packed_flattened_position_ids],
                sin_h=self.rope.sin_h[packed_flattened_position_ids],
                cos_w=self.rope.cos_w[packed_flattened_position_ids],
                sin_w=self.rope.sin_w[packed_flattened_position_ids],
            )
        for layer in self.encoder.layers:
            hidden_states = layer(hidden_states, cu_seqlens, max_seqlen, **extra)
        return self.post_layernorm(hidden_states)


class SiglipVisionModel(nn.Module):
    """Top-level SigLIP vision model wrapper."""

    def __init__(self, config):
        super().__init__()
        self.vision_model = SiglipVisionTransformer(config)
        self.config = config

    def forward(self, packed_pixel_values, packed_flattened_position_ids, cu_seqlens, max_seqlen):
        return self.vision_model(packed_pixel_values, packed_flattened_position_ids, cu_seqlens, max_seqlen)
