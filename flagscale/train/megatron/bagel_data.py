# Copyright (c) 2025, BAAI. All rights reserved.
# BAGEL data pipeline adapted for Megatron DP/TP parallelism.
#
# Key design:
# - DP: Each DP rank loads different data (sharded by DP rank/world_size)
# - TP: Only TP rank 0 reads from data_iterator, broadcasts to other TP ranks
# - Uses broadcast_data pattern from qwen2_5_vl for TP synchronization

import os
import sys
import json
import math
import random
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset

from megatron.training import get_args, print_rank_0
from megatron.core import parallel_state
from megatron.core.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_src_rank,
)


# ============================================================
# broadcast_data: TP rank 0 → all TP ranks
# Adapted from qwen2_5_vl/tensor_parallel.py to handle BAGEL's
# variable-length packed tensors
# ============================================================

_MAX_DATA_DIM = 5


def _build_key_size_numel_dictionaries(keys, data):
    """Build the size on rank 0 and broadcast."""
    max_dim = _MAX_DATA_DIM
    sizes = [-1 for _ in range(max_dim) for _ in keys]

    if get_tensor_model_parallel_rank() == 0:
        offset = 0
        for key in keys:
            assert data[key].dim() < max_dim, f'tensor {key} has too many dims'
            size = data[key].size()
            for i, s in enumerate(size):
                sizes[i + offset] = s
            offset += max_dim

    sizes_cuda = torch.tensor(sizes, dtype=torch.long, device='cuda')
    torch.distributed.broadcast(
        sizes_cuda, get_tensor_model_parallel_src_rank(),
        group=get_tensor_model_parallel_group()
    )

    sizes_cpu = sizes_cuda.cpu()
    key_size = {}
    key_numel = {}
    total_numel = 0
    offset = 0
    for key in keys:
        i = 0
        size = []
        numel = 1
        while sizes_cpu[offset + i] >= 0:
            this_size = sizes_cpu[offset + i]
            size.append(this_size)
            numel *= this_size
            i += 1
        key_size[key] = size
        key_numel[key] = numel
        total_numel += numel
        offset += max_dim

    return key_size, key_numel, total_numel


def broadcast_data(keys, data, datatype):
    """Broadcast data from TP rank 0 to all TP ranks.
    
    Args:
        keys: list of keys in the data dictionary to broadcast
        data: data dictionary (only valid on TP rank 0, None on others)
        datatype: torch dtype for all tensors
    
    Returns:
        dict mapping keys to tensors on all TP ranks
    """
    key_size, key_numel, total_numel = _build_key_size_numel_dictionaries(keys, data)

    if get_tensor_model_parallel_rank() == 0:
        flatten_data = torch.cat(
            [data[key].contiguous().view(-1) for key in keys], dim=0
        ).cuda()
    else:
        flatten_data = torch.empty(
            total_numel, device=torch.cuda.current_device(), dtype=datatype
        )

    torch.distributed.broadcast(
        flatten_data, get_tensor_model_parallel_src_rank(),
        group=get_tensor_model_parallel_group()
    )

    output = {}
    offset = 0
    for key in keys:
        size = key_size[key]
        numel = key_numel[key]
        output[key] = flatten_data.narrow(0, offset, numel).view(size)
        offset += numel

    return output


def broadcast_scalar(name, data, dtype=torch.long):
    """Broadcast a single scalar from TP rank 0."""
    if get_tensor_model_parallel_rank() == 0:
        tensor = torch.tensor([data[name]], dtype=dtype, device='cuda')
    else:
        tensor = torch.empty(1, dtype=dtype, device='cuda')
    torch.distributed.broadcast(
        tensor, get_tensor_model_parallel_src_rank(),
        group=get_tensor_model_parallel_group()
    )
    return tensor.item()


# ============================================================
# get_batch: Read data on TP rank 0, broadcast to all TP ranks
# ============================================================

def get_batch(data_iterator):
    """Get a batch with proper DP/TP handling.
    
    - Only TP rank 0 reads from data_iterator (which is already DP-sharded)
    - All tensor fields are broadcast to other TP ranks
    - Returns dict ready for model forward pass
    """
    # Only TP rank 0 reads data
    if data_iterator is not None and get_tensor_model_parallel_rank() == 0:
        data = next(data_iterator)
        # Convert SimpleCustomBatch to dict if needed
        if hasattr(data, 'to_dict'):
            data = data.to_dict()
    else:
        data = None

    # Broadcast number of fields present (to know which fields to expect)
    # Use a flags tensor to indicate which optional fields are present
    # Bit flags: 0=has_vit, 1=has_vae, 2=has_timesteps, 3=has_labels
    if get_tensor_model_parallel_rank() == 0:
        flags = 0
        if 'packed_vit_tokens' in data and data['packed_vit_tokens'] is not None:
            flags |= 1
        if 'packed_vae_token_indexes' in data and data['packed_vae_token_indexes'] is not None:
            flags |= 2
        if 'packed_timesteps' in data and data['packed_timesteps'] is not None:
            flags |= 4
        if 'packed_label_ids' in data and data['packed_label_ids'] is not None:
            flags |= 8
        flags_tensor = torch.tensor([flags], dtype=torch.long, device='cuda')
    else:
        flags_tensor = torch.empty(1, dtype=torch.long, device='cuda')

    torch.distributed.broadcast(
        flags_tensor, get_tensor_model_parallel_src_rank(),
        group=get_tensor_model_parallel_group()
    )
    flags = flags_tensor.item()

    # Broadcast core fields (always present)
    batch = {}

    # Integer tensors
    int_keys = ["packed_text_ids", "packed_text_indexes", "packed_position_ids"]
    int_data = broadcast_data(int_keys, data, torch.long)
    batch["packed_text_ids"] = int_data["packed_text_ids"]
    batch["packed_text_indexes"] = int_data["packed_text_indexes"]
    batch["packed_position_ids"] = int_data["packed_position_ids"]

    # Sequence length (scalar)
    batch["sequence_length"] = broadcast_scalar("sequence_length", data, torch.long)

    # Sample lens (list → tensor for broadcast)
    if get_tensor_model_parallel_rank() == 0:
        data["sample_lens_tensor"] = torch.tensor(data["sample_lens"], dtype=torch.long)
    sample_lens = broadcast_data(["sample_lens_tensor"], data, torch.long)["sample_lens_tensor"]
    batch["sample_lens"] = sample_lens.tolist()

    # ViT fields
    if flags & 1:
        vit_float = broadcast_data(
            ["packed_vit_tokens"], data, torch.float32
        )
        batch["packed_vit_tokens"] = vit_float["packed_vit_tokens"]

        vit_int = broadcast_data(
            ["packed_vit_position_ids", "packed_vit_token_indexes", "vit_token_seqlens"],
            data, torch.long
        )
        batch["packed_vit_position_ids"] = vit_int["packed_vit_position_ids"]
        batch["packed_vit_token_indexes"] = vit_int["packed_vit_token_indexes"]
        batch["vit_token_seqlens"] = vit_int["vit_token_seqlens"]

    # VAE fields
    if flags & 2:
        vae_int = broadcast_data(
            ["packed_vae_token_indexes", "packed_latent_position_ids"],
            data, torch.long
        )
        batch["packed_vae_token_indexes"] = vae_int["packed_vae_token_indexes"]
        batch["packed_latent_position_ids"] = vae_int["packed_latent_position_ids"]

        # padded_images (float) - all ranks must call broadcast_data together
        vae_float = broadcast_data(["padded_images"], data, torch.float32)
        batch["padded_images"] = vae_float["padded_images"]

        # patchified_vae_latent_shapes: list of (h, w) -> encode as Nx2 tensor for broadcast
        if get_tensor_model_parallel_rank() == 0:
            shapes = data.get("patchified_vae_latent_shapes", [])
            if shapes:
                data["vae_latent_shapes_tensor"] = torch.tensor(shapes, dtype=torch.long)
            else:
                data["vae_latent_shapes_tensor"] = torch.zeros(0, 2, dtype=torch.long)
        shapes_int = broadcast_data(["vae_latent_shapes_tensor"], data, torch.long)
        shapes_tensor = shapes_int["vae_latent_shapes_tensor"]
        if shapes_tensor.numel() > 0:
            batch["patchified_vae_latent_shapes"] = [(int(h), int(w)) for h, w in shapes_tensor.reshape(-1, 2)]
        else:
            batch["patchified_vae_latent_shapes"] = []

    # Timestep fields
    if flags & 4:
        ts_float = broadcast_data(["packed_timesteps"], data, torch.float32)
        batch["packed_timesteps"] = ts_float["packed_timesteps"]

        ts_int = broadcast_data(["mse_loss_indexes"], data, torch.long)
        batch["mse_loss_indexes"] = ts_int["mse_loss_indexes"]

    # Label fields
    if flags & 8:
        label_int = broadcast_data(
            ["packed_label_ids", "ce_loss_indexes"],
            data, torch.long
        )
        batch["packed_label_ids"] = label_int["packed_label_ids"]
        batch["ce_loss_indexes"] = label_int["ce_loss_indexes"]

        label_float = broadcast_data(["ce_loss_weights"], data, torch.float32)
        batch["ce_loss_weights"] = label_float["ce_loss_weights"]

    # Attention mask info (for flex attention)
    if get_tensor_model_parallel_rank() == 0 and 'split_lens' in data:
        # Convert lists to tensors for broadcast
        data["split_lens_tensor"] = torch.tensor(data["split_lens"], dtype=torch.long)
        # attn_modes: encode as integers (causal=0, full=1, noise=2)
        mode_map = {"causal": 0, "full": 1, "noise": 2}
        data["attn_modes_tensor"] = torch.tensor(
            [mode_map.get(m, 0) for m in data["attn_modes"]], dtype=torch.long
        )
    elif get_tensor_model_parallel_rank() == 0 and 'split_lens' not in data:
        data["split_lens_tensor"] = torch.tensor([], dtype=torch.long)
        data["attn_modes_tensor"] = torch.tensor([], dtype=torch.long)

    attn_data = broadcast_data(
        ["split_lens_tensor", "attn_modes_tensor"], data, torch.long
    )
    if attn_data["split_lens_tensor"].numel() > 0:
        batch["split_lens"] = attn_data["split_lens_tensor"].tolist()
        inv_mode_map = {0: "causal", 1: "full", 2: "noise"}
        batch["attn_modes"] = [inv_mode_map[m.item()] for m in attn_data["attn_modes_tensor"]]

    return batch


# ============================================================
# Dataloader provider: builds DP-sharded dataloaders
# ============================================================

def build_bagel_dataloaders_provider(train_val_test_num_samples):
    """Build BAGEL dataloaders with proper DP sharding.
    
    - Each DP rank gets a different shard of data
    - TP ranks within same DP group share data (via broadcast in get_batch)
    - Only dataloader ranks (first/last PP stage) create dataloaders
    """
    args = get_args()

    # Only create dataloader on ranks that need it
    # For no PP: all ranks. For PP: first and last stage.
    pp_rank = parallel_state.get_pipeline_model_parallel_rank()
    pp_size = parallel_state.get_pipeline_model_parallel_world_size()
    if pp_size > 1 and pp_rank not in [0, pp_size - 1]:
        return None, None, None

    # Get DP rank/world_size for data sharding
    dp_rank = parallel_state.get_data_parallel_rank()
    dp_world_size = parallel_state.get_data_parallel_world_size()

    # Only TP rank 0 needs actual data loading
    tp_rank = get_tensor_model_parallel_rank()
    if tp_rank != 0:
        # Non-TP-rank-0 processes don't need a dataloader
        # They'll receive data via broadcast_data in get_batch
        return _DummyDataloader(), None, None

    # Setup BAGEL data path
    bagel_src_dir = getattr(args, "bagel_src_dir", None)
    if bagel_src_dir and bagel_src_dir not in sys.path:
        sys.path.insert(0, bagel_src_dir)

    model_path = getattr(args, "bagel_model_path", None)
    dataset_config_file = getattr(args, "dataset_config_file", None)

    import yaml
    from transformers import AutoTokenizer
    from data.data_utils import add_special_tokens
    from data.dataset_base import PackedDataset, collate_wrapper

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer, special_tokens, _ = add_special_tokens(tokenizer)

    # Load dataset config
    with open(dataset_config_file) as f:
        dataset_config = yaml.safe_load(f)

    # Create a simple namespace for data_config
    class DataConfig:
        pass
    data_config = DataConfig()
    for k, v in dataset_config.get("data_config", {}).items():
        setattr(data_config, k, v)

    # Set defaults if not in config
    if not hasattr(data_config, 'max_num_tokens'):
        data_config.max_num_tokens = getattr(args, 'seq_length', 11520)
    if not hasattr(data_config, 'max_num_tokens_per_sample'):
        data_config.max_num_tokens_per_sample = data_config.max_num_tokens
    if not hasattr(data_config, 'expected_num_tokens'):
        data_config.expected_num_tokens = int(data_config.max_num_tokens * 0.95)
    if not hasattr(data_config, 'vit_patch_size'):
        data_config.vit_patch_size = 14
    if not hasattr(data_config, 'max_num_patch_per_side'):
        data_config.max_num_patch_per_side = 70
    if not hasattr(data_config, 'vae_image_downsample'):
        data_config.vae_image_downsample = 16  # vae_downsample * latent_patch_size = 8*2
    if not hasattr(data_config, 'max_latent_size'):
        data_config.max_latent_size = 64
    if not hasattr(data_config, 'text_cond_dropout_prob'):
        data_config.text_cond_dropout_prob = 0.0
    if not hasattr(data_config, 'vit_cond_dropout_prob'):
        data_config.vit_cond_dropout_prob = 0.0
    if not hasattr(data_config, 'vae_cond_dropout_prob'):
        data_config.vae_cond_dropout_prob = 0.0

    # Build PackedDataset with DP rank/world_size for sharding
    packed_dataset = PackedDataset(
        data_config=data_config,
        tokenizer=tokenizer,
        special_tokens=special_tokens,
        local_rank=dp_rank,          # DP rank for data sharding
        world_size=dp_world_size,    # DP world size
        num_workers=getattr(args, 'num_workers', 4),
        use_flex=True,               # Use flex attention (split_lens + attn_modes)
    )
    packed_dataset.set_epoch(seed=42)

    # Create DataLoader
    train_dataloader = DataLoader(
        packed_dataset,
        batch_size=1,  # PackedDataset already packs multiple samples
        num_workers=getattr(args, 'num_workers', 4),
        collate_fn=collate_wrapper(),
        pin_memory=True,
        prefetch_factor=2,
    )

    print_rank_0(
        f"BAGEL dataloader created: dp_rank={dp_rank}/{dp_world_size}, "
        f"max_tokens={data_config.max_num_tokens}"
    )

    return _CyclicDataloader(train_dataloader), None, None


class _CyclicDataloader:
    """Wraps a dataloader to cycle infinitely."""
    def __init__(self, dataloader):
        self._dataloader = dataloader
        self._iter = iter(self._cycle())

    def _cycle(self):
        while True:
            for batch in self._dataloader:
                yield batch

    def __next__(self):
        return next(self._iter)

    def __iter__(self):
        return self

    def __len__(self):
        # Return a large number for Megatron's sampler
        # Actual iteration count is controlled by train_iters
        return 10**9


class _DummyDataloader:
    """Dummy dataloader for non-TP-rank-0 processes."""
    def __next__(self):
        return None

    def __iter__(self):
        return self


def build_bagel_dataloaders(bagel_src_dir, model_path, dataset_config_file, 
                             dp_rank, dp_world_size, num_workers, seed):
    """Wrapper to build BAGEL dataloader with explicit DP parameters.
    
    This is called only on TP rank 0 of each DP group.
    """
    import sys
    if bagel_src_dir and bagel_src_dir not in sys.path:
        sys.path.insert(0, bagel_src_dir)

    import yaml
    from transformers import AutoTokenizer
    from data.data_utils import add_special_tokens
    from data.dataset_base import PackedDataset, collate_wrapper
    from torch.utils.data import DataLoader

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer, special_tokens, _ = add_special_tokens(tokenizer)

    # Load dataset config - the YAML file IS the grouped_datasets dict
    from data.dataset_base import DataConfig as BagelDataConfig
    with open(dataset_config_file) as f:
        grouped_datasets = yaml.safe_load(f)

    # Use BAGEL's native DataConfig which expects grouped_datasets as first arg
    data_config = BagelDataConfig(
        grouped_datasets=grouped_datasets,
        text_cond_dropout_prob=0.0,
        vit_cond_dropout_prob=0.0,
        vae_cond_dropout_prob=0.0,
        vae_image_downsample=16,  # vae_downsample * latent_patch_size = 8*2
        max_latent_size=64,
        vit_patch_size=14,
        max_num_patch_per_side=70,
    )

    # Build PackedDataset with DP sharding
    from megatron.training import get_args
    args = get_args()
    seq_length = getattr(args, 'seq_length', 11520)
    packed_dataset = PackedDataset(
        data_config=data_config,
        tokenizer=tokenizer,
        special_tokens=special_tokens,
        local_rank=dp_rank,
        world_size=dp_world_size,
        num_workers=num_workers,
        expected_num_tokens=int(seq_length * 0.95),
        max_num_tokens_per_sample=seq_length,
        max_num_tokens=seq_length,
        use_flex=True,
    )
    packed_dataset.set_epoch(seed=seed)

    # Create DataLoader
    dataloader = DataLoader(
        packed_dataset,
        batch_size=1,
        num_workers=num_workers,
        collate_fn=collate_wrapper(),
        pin_memory=True,
        prefetch_factor=2,
    )

    return _CyclicDataloader(dataloader)
