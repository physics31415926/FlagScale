#!/usr/bin/env python
# Copyright (c) 2025, BAAI. All rights reserved.
# Convert BAGEL HuggingFace checkpoint to Megatron format.
# Simple direct mapping since our model uses same naming as HF (minus prefix).

import argparse
import json
import math
import os
import re
import torch
from safetensors import safe_open


def parse_args():
    parser = argparse.ArgumentParser(description="Convert BAGEL HF checkpoint to Megatron format")
    parser.add_argument("--input-dir", type=str, required=True, help="Path to HF BAGEL model directory")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for Megatron checkpoint")
    parser.add_argument("--tp-size", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--use-ema", action="store_true", help="Load from ema.safetensors instead of model weights")
    return parser.parse_args()


def load_hf_weights(input_dir, use_ema=False):
    """Load all weights from HF safetensors files."""
    state_dict = {}

    if use_ema:
        ema_path = os.path.join(input_dir, "ema.safetensors")
        assert os.path.exists(ema_path), f"EMA file not found: {ema_path}"
        print(f"Loading EMA weights from {ema_path}")
        with safe_open(ema_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)
        return state_dict

    index_file = os.path.join(input_dir, "model.safetensors.index.json")
    state_dict = {}

    if os.path.exists(index_file):
        with open(index_file) as f:
            index = json.load(f)
        weight_map = index["weight_map"]
        loaded_files = set()
        for param_name, filename in weight_map.items():
            if filename not in loaded_files:
                filepath = os.path.join(input_dir, filename)
                with safe_open(filepath, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        state_dict[key] = f.get_tensor(key)
                loaded_files.add(filename)
    else:
        filepath = os.path.join(input_dir, "model.safetensors")
        with safe_open(filepath, framework="pt", device="cpu") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)

    return state_dict


# Keys that need TP column-parallel split (split output dim=0)
COLUMN_PARALLEL_KEYS = [
    "self_attn.q_proj.weight", "self_attn.q_proj.bias",
    "self_attn.k_proj.weight", "self_attn.k_proj.bias",
    "self_attn.v_proj.weight", "self_attn.v_proj.bias",
    "self_attn.q_proj_moe_gen.weight", "self_attn.q_proj_moe_gen.bias",
    "self_attn.k_proj_moe_gen.weight", "self_attn.k_proj_moe_gen.bias",
    "self_attn.v_proj_moe_gen.weight", "self_attn.v_proj_moe_gen.bias",
    "mlp.gate_proj.weight", "mlp.up_proj.weight",
    "mlp_moe_gen.gate_proj.weight", "mlp_moe_gen.up_proj.weight",  # After key renaming
]

# Keys that need TP row-parallel split (split input dim=1 for weight, dim=0 for bias)
ROW_PARALLEL_KEYS = [
    "self_attn.o_proj.weight", "self_attn.o_proj_moe_gen.weight",
    "mlp.down_proj.weight",
    "mlp_moe_gen.down_proj.weight",  # After key renaming
]

# Embedding/output keys split on dim=0
# NOTE: embed_tokens is nn.Embedding (NOT VocabParallelEmbedding), so it's replicated
# lm_head is ColumnParallelLinear, so it IS split
VOCAB_PARALLEL_KEYS = ["lm_head.weight"]


def map_hf_key_to_megatron(hf_key):
    """Map HF key to our model's key.
    
    HF naming:
    - language_model.model.layers.{i}.xxx → layers.{i}.xxx
    - language_model.model.embed_tokens.weight → embed_tokens.weight
    - language_model.model.norm.weight → norm.weight
    - language_model.model.norm_moe_gen.weight → norm_moe_gen.weight
    - language_model.lm_head.weight → lm_head.weight
    - vit_model.xxx → vit_model.xxx
    - connector.xxx → connector.xxx
    - vae2llm.xxx → vae2llm.xxx
    - llm2vae.xxx → llm2vae.xxx
    - time_embedder.xxx → timestep_embedder.xxx (name difference!)
    - latent_pos_embed.xxx → vae_position_embedding.xxx (name difference!)
    - vit_pos_embed.xxx → (handled by vit_model)
    - encoder.xxx / decoder.xxx → skip (VAE, frozen)
    """
    # Skip VAE encoder/decoder weights
    if hf_key.startswith("encoder.") or hf_key.startswith("decoder."):
        return None

    # LLM backbone
    if hf_key.startswith("language_model.model."):
        key = hf_key[len("language_model.model."):]
    elif hf_key.startswith("language_model."):
        key = hf_key[len("language_model."):]
    else:
        key = None

    if key is not None:
        # Rename MoT MLP keys: mlp.gate_proj_moe_gen → mlp_moe_gen.gate_proj
        key = re.sub(r'mlp\.(gate_proj|up_proj|down_proj)_moe_gen\.', r'mlp_moe_gen.\1.', key)
        # Rename MoT attention keys: self_attn.q_proj_moe_gen → self_attn_moe_gen.q_proj (if applicable)
        # Actually check if model uses self_attn_moe_gen or keeps them in self_attn
        # For now, keep attention moe_gen keys as-is since model may handle them differently
        return key

    # Connectors with name mapping
    if hf_key.startswith("time_embedder."):
        return "timestep_embedder." + hf_key[len("time_embedder."):]
    if hf_key.startswith("latent_pos_embed."):
        return "vae_position_embedding." + hf_key[len("latent_pos_embed."):]

    # Direct mapping (vit_model, connector, vae2llm, llm2vae, vit_pos_embed)
    return hf_key


def get_tp_split_info(megatron_key, tp_size):
    """Determine how to split a key across TP ranks."""
    if tp_size == 1:
        return None, None

    # ViT model weights are NOT split across TP ranks (ViT is replicated)
    if megatron_key.startswith("vit_model."):
        return None, None

    for pattern in COLUMN_PARALLEL_KEYS:
        if megatron_key.endswith(pattern):
            return "column", 0

    for pattern in ROW_PARALLEL_KEYS:
        if megatron_key.endswith(pattern):
            return "row", 1  # split along input dim

    for pattern in VOCAB_PARALLEL_KEYS:
        if megatron_key == pattern:
            return "vocab", 0

    return None, None


def convert(input_dir, output_dir, tp_size, use_ema=False):
    print(f"Loading HF weights from {input_dir}...")
    hf_state_dict = load_hf_weights(input_dir, use_ema=use_ema)
    print(f"Loaded {len(hf_state_dict)} tensors")

    # Compute padded vocab size (same logic as Megatron's _vocab_size_with_padding)
    # NullTokenizer adds 1 EOD token, then pads to multiple of (make_vocab_size_divisible_by * tp_size)
    orig_vocab_size = 152064  # from config
    tokenizer_vocab_size = orig_vocab_size + 1  # NullTokenizer adds EOD
    multiple = 1 * tp_size  # make_vocab_size_divisible_by=1
    padded_vocab_size = int(math.ceil(tokenizer_vocab_size / multiple) * multiple)
    print(f"Vocab: orig={orig_vocab_size}, tokenizer={tokenizer_vocab_size}, padded={padded_vocab_size}")

    # Initialize per-rank state dicts
    megatron_state_dicts = [{"model": {}} for _ in range(tp_size)]

    mapped = 0
    skipped = 0
    for hf_key, tensor in hf_state_dict.items():
        megatron_key = map_hf_key_to_megatron(hf_key)
        if megatron_key is None:
            skipped += 1
            continue

        # Pad vocab-related weights to padded_vocab_size
        if megatron_key in ("embed_tokens.weight", "lm_head.weight"):
            if tensor.shape[0] < padded_vocab_size:
                pad_rows = padded_vocab_size - tensor.shape[0]
                tensor = torch.cat([tensor, torch.zeros(pad_rows, tensor.shape[1], dtype=tensor.dtype)], dim=0)
                print(f"  Padded {megatron_key}: {tensor.shape[0] - pad_rows} -> {tensor.shape[0]}")

        split_type, split_dim = get_tp_split_info(megatron_key, tp_size)

        if split_type is None:
            # Replicate across all TP ranks
            for i in range(tp_size):
                megatron_state_dicts[i]["model"][megatron_key] = tensor.clone()
        else:
            # Split across TP ranks
            chunks = list(tensor.chunk(tp_size, dim=split_dim))
            for i in range(tp_size):
                megatron_state_dicts[i]["model"][megatron_key] = chunks[i].clone()

        mapped += 1

    print(f"Mapped: {mapped}, Skipped (VAE): {skipped}")

    # Save
    os.makedirs(output_dir, exist_ok=True)
    iter_dir = os.path.join(output_dir, "iter_0000001")
    os.makedirs(iter_dir, exist_ok=True)

    for i in range(tp_size):
        rank_dir = os.path.join(iter_dir, f"mp_rank_{i:02d}")
        os.makedirs(rank_dir, exist_ok=True)
        save_path = os.path.join(rank_dir, "model_optim_rng.pt")
        torch.save(megatron_state_dicts[i], save_path)
        print(f"Saved TP rank {i}: {save_path} ({len(megatron_state_dicts[i]['model'])} keys)")

    # Save latest iteration marker
    with open(os.path.join(output_dir, "latest_checkpointed_iteration.txt"), "w") as f:
        f.write("1\n")

    print("Conversion complete!")


if __name__ == "__main__":
    args = parse_args()
    convert(args.input_dir, args.output_dir, args.tp_size, use_ema=args.use_ema)
