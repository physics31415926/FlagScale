# Copyright (c) 2025, BAAI. All rights reserved.
# Mixture-of-Transformers (MoT) layer for BAGEL.
# Key design: Q/K/V/O projections are SPLIT by token type (und/gen),
# but attention is computed on the FULL sequence (all tokens attend together).

import torch
from torch import nn
import torch.nn.functional as F
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer.transformer_config import TransformerConfig
from flash_attn import flash_attn_varlen_func


class Qwen2RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class Qwen2RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=32768, base=1000000.0):
        super().__init__()
        self.dim = dim
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, position_ids):
        inv_freq_expanded = self.inv_freq[None, :].expand(position_ids.shape[0], -1)
        position_ids_expanded = position_ids[:, None].float()
        freqs = (inv_freq_expanded * position_ids_expanded)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    """Apply RoPE. q/k: [seq_len, num_heads, head_dim], cos/sin: [seq_len, head_dim]."""
    cos = cos.unsqueeze(1)  # [seq_len, 1, head_dim]
    sin = sin.unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class PackedAttentionMoT(nn.Module):
    """MoT Attention: split Q/K/V/O projections, shared attention computation.
    
    Uses standard Megatron TP pattern: Q/K/V split across TP ranks (local heads),
    attention computed on local heads, O_proj with input_is_parallel=True.
    
    In "und" mode: all tokens use und projections.
    In "mixed" mode: und tokens use und projections, gen tokens use gen projections.
    Attention is ALWAYS computed on the full sequence (all tokens attend together).
    """
    def __init__(self, config: TransformerConfig, has_gen_path=True):
        super().__init__()
        h = config.hidden_size
        nh = config.num_attention_heads
        nkv = config.num_query_groups
        self.head_dim = h // nh
        self.num_heads = nh
        self.num_kv_heads = nkv
        
        # Get TP world size for local head calculation
        from megatron.core import parallel_state
        tp_size = parallel_state.get_tensor_model_parallel_world_size()
        self.num_local_heads = nh // tp_size
        self.num_local_kv_heads = nkv // tp_size
        
        # Understanding path projections (TP-parallel, gather_output=False for local heads)
        self.q_proj = ColumnParallelLinear(h, nh * self.head_dim, config=config,
                                           init_method=config.init_method, bias=True, gather_output=False)
        self.k_proj = ColumnParallelLinear(h, nkv * self.head_dim, config=config,
                                           init_method=config.init_method, bias=True, gather_output=False)
        self.v_proj = ColumnParallelLinear(h, nkv * self.head_dim, config=config,
                                           init_method=config.init_method, bias=True, gather_output=False)
        self.o_proj = RowParallelLinear(nh * self.head_dim, h, config=config,
                                        init_method=config.output_layer_init_method, bias=False, input_is_parallel=True, skip_bias_add=False)
        
        # QK norms for understanding path
        self.q_norm = Qwen2RMSNorm(self.head_dim)
        self.k_norm = Qwen2RMSNorm(self.head_dim)
        
        # Generation path projections (only if has_gen_path)
        self.has_gen_path = has_gen_path
        if has_gen_path:
            self.q_proj_moe_gen = ColumnParallelLinear(h, nh * self.head_dim, config=config,
                                                       init_method=config.init_method, bias=True, gather_output=False)
            self.k_proj_moe_gen = ColumnParallelLinear(h, nkv * self.head_dim, config=config,
                                                       init_method=config.init_method, bias=True, gather_output=False)
            self.v_proj_moe_gen = ColumnParallelLinear(h, nkv * self.head_dim, config=config,
                                                       init_method=config.init_method, bias=True, gather_output=False)
            self.o_proj_moe_gen = RowParallelLinear(nh * self.head_dim, h, config=config,
                                                    init_method=config.output_layer_init_method, bias=False, input_is_parallel=True, skip_bias_add=False)
            self.q_norm_moe_gen = Qwen2RMSNorm(self.head_dim)
            self.k_norm_moe_gen = Qwen2RMSNorm(self.head_dim)

    def _compute_qkv(self, hidden_states, und_idx, gen_idx, mode):
        """Compute Q/K/V with split projections. Output has LOCAL heads only (TP-split)."""
        seq_len = hidden_states.shape[0]
        
        # Allocate full-length Q/K/V with LOCAL head counts
        q = hidden_states.new_zeros(seq_len, self.num_local_heads * self.head_dim)
        k = hidden_states.new_zeros(seq_len, self.num_local_kv_heads * self.head_dim)
        v = hidden_states.new_zeros(seq_len, self.num_local_kv_heads * self.head_dim)
        
        if mode == "und":
            # All tokens use und projections
            q_out, _ = self.q_proj(hidden_states)
            k_out, _ = self.k_proj(hidden_states)
            v_out, _ = self.v_proj(hidden_states)
            q = q_out
            k = k_out
            v = v_out
        else:
            # Mixed mode: split by token type
            if und_idx is not None and len(und_idx) > 0:
                h_und = hidden_states[und_idx]
                q_und, _ = self.q_proj(h_und)
                k_und, _ = self.k_proj(h_und)
                v_und, _ = self.v_proj(h_und)
                q[und_idx] = q_und
                k[und_idx] = k_und
                v[und_idx] = v_und
            if gen_idx is not None and len(gen_idx) > 0 and self.has_gen_path:
                h_gen = hidden_states[gen_idx]
                q_gen, _ = self.q_proj_moe_gen(h_gen)
                k_gen, _ = self.k_proj_moe_gen(h_gen)
                v_gen, _ = self.v_proj_moe_gen(h_gen)
                q[gen_idx] = q_gen
                k[gen_idx] = k_gen
                v[gen_idx] = v_gen
        
        # Reshape to [seq_len, local_num_heads, head_dim]
        q = q.view(seq_len, self.num_local_heads, self.head_dim)
        k = k.view(seq_len, self.num_local_kv_heads, self.head_dim)
        v = v.view(seq_len, self.num_local_kv_heads, self.head_dim)
        
        # Apply QK norms (split by token type)
        if mode == "und":
            q = self.q_norm(q)
            k = self.k_norm(k)
        else:
            q_normed = torch.zeros_like(q)
            k_normed = torch.zeros_like(k)
            if und_idx is not None and len(und_idx) > 0:
                q_normed[und_idx] = self.q_norm(q[und_idx])
                k_normed[und_idx] = self.k_norm(k[und_idx])
            if gen_idx is not None and len(gen_idx) > 0 and self.has_gen_path:
                q_normed[gen_idx] = self.q_norm_moe_gen(q[gen_idx])
                k_normed[gen_idx] = self.k_norm_moe_gen(k[gen_idx])
            q = q_normed
            k = k_normed
        
        return q, k, v

    def forward(self, packed_sequence, cu_seqlens, max_seqlen, cos, sin, und_idx, gen_idx, mode="und"):
        """
        Args:
            packed_sequence: [total_seq_len, hidden_size]
            cu_seqlens: [num_samples + 1] cumulative sequence lengths
            max_seqlen: max sequence length in batch
            cos, sin: [total_seq_len, head_dim] rotary embeddings
            und_idx: indices of understanding tokens in full sequence
            gen_idx: indices of generation tokens in full sequence
            mode: "und" (all tokens use und path) or "mixed" (split by token type)
        """
        # Step 1: Compute Q/K/V with split projections
        q, k, v = self._compute_qkv(packed_sequence, und_idx, gen_idx, mode)
        
        # Step 2: Apply RoPE to full sequence
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        # Step 3: Flash attention on FULL sequence (all tokens attend together)
        # Validate cu_seqlens
        assert cu_seqlens.dtype == torch.int32, f"cu_seqlens must be int32, got {cu_seqlens.dtype}"
        assert cu_seqlens[-1].item() == q.shape[0], f"cu_seqlens[-1]={cu_seqlens[-1].item()} != q.shape[0]={q.shape[0]}"
        
        attn_output = flash_attn_varlen_func(
            q.bfloat16(), k.bfloat16(), v.bfloat16(),
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            causal=True,
        )
        attn_output = attn_output.to(packed_sequence.dtype)
        
        # Step 4: Reshape and apply split O projections
        attn_output = attn_output.reshape(attn_output.shape[0], -1)  # [seq_len, num_heads * head_dim]
        
        if mode == "und":
            output, _ = self.o_proj(attn_output)
        else:
            output = torch.zeros(attn_output.shape[0], packed_sequence.shape[1],
                                dtype=packed_sequence.dtype, device=packed_sequence.device)
            if und_idx is not None and len(und_idx) > 0:
                o_und, _ = self.o_proj(attn_output[und_idx])
                output[und_idx] = o_und
            if gen_idx is not None and len(gen_idx) > 0 and self.has_gen_path:
                o_gen, _ = self.o_proj_moe_gen(attn_output[gen_idx])
                output[gen_idx] = o_gen
        
        return output


class Qwen2MLP(nn.Module):
    """Single-path MLP (used for both und and gen)."""
    def __init__(self, config):
        super().__init__()
        h = config.hidden_size; f = config.ffn_hidden_size
        self.gate_proj = ColumnParallelLinear(h, f, config=config, init_method=config.init_method, bias=False, gather_output=False)
        self.up_proj = ColumnParallelLinear(h, f, config=config, init_method=config.init_method, bias=False, gather_output=False)
        self.down_proj = RowParallelLinear(f, h, config=config, init_method=config.output_layer_init_method, bias=False, input_is_parallel=True, skip_bias_add=False)

    def forward(self, x):
        gate, _ = self.gate_proj(x)
        up, _ = self.up_proj(x)
        out, _ = self.down_proj(F.silu(gate) * up)
        return out


class MoTTransformerLayer(nn.Module):
    """Full MoT transformer layer: attention + MLP with dual paths."""
    def __init__(self, config: TransformerConfig, has_gen_path=True):
        super().__init__()
        self.has_gen_path = has_gen_path
        
        # Pre-attention layernorms (split for und/gen)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size)
        if has_gen_path:
            self.input_layernorm_moe_gen = Qwen2RMSNorm(config.hidden_size)
        
        # Attention (handles split projections internally)
        self.self_attn = PackedAttentionMoT(config, has_gen_path=has_gen_path)
        
        # Post-attention layernorms (split for und/gen)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size)
        if has_gen_path:
            self.post_attention_layernorm_moe_gen = Qwen2RMSNorm(config.hidden_size)
        
        # MLPs (separate for und/gen)
        self.mlp = Qwen2MLP(config)
        if has_gen_path:
            self.mlp_moe_gen = Qwen2MLP(config)

    def forward(self, packed_sequence, cu_seqlens, max_seqlen, cos, sin, und_idx, gen_idx, mode="und"):
        residual = packed_sequence
        
        # Pre-attention layernorm (split by token type)
        if mode == "und":
            h = self.input_layernorm(packed_sequence)
        else:
            h = torch.zeros_like(packed_sequence)
            if und_idx is not None and len(und_idx) > 0:
                h[und_idx] = self.input_layernorm(packed_sequence[und_idx])
            if gen_idx is not None and len(gen_idx) > 0 and self.has_gen_path:
                h[gen_idx] = self.input_layernorm_moe_gen(packed_sequence[gen_idx])
        
        # Attention (full sequence, split projections)
        attn_out = self.self_attn(h, cu_seqlens, max_seqlen, cos, sin, und_idx, gen_idx, mode)
        hidden_states = residual + attn_out
        
        # Post-attention layernorm + MLP (split by token type)
        residual = hidden_states
        if mode == "und":
            mi = self.post_attention_layernorm(hidden_states)
            mlp_out = self.mlp(mi)
        else:
            mi = torch.zeros_like(hidden_states)
            if und_idx is not None and len(und_idx) > 0:
                mi[und_idx] = self.post_attention_layernorm(hidden_states[und_idx])
            if gen_idx is not None and len(gen_idx) > 0 and self.has_gen_path:
                mi[gen_idx] = self.post_attention_layernorm_moe_gen(hidden_states[gen_idx])
            mlp_out = torch.zeros_like(mi)
            if und_idx is not None and len(und_idx) > 0:
                mlp_out[und_idx] = self.mlp(mi[und_idx])
            if gen_idx is not None and len(gen_idx) > 0 and self.has_gen_path:
                mlp_out[gen_idx] = self.mlp_moe_gen(mi[gen_idx])
        
        return residual + mlp_out
