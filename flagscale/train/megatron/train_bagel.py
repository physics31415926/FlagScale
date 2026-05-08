# Copyright (c) 2025, BAAI. All rights reserved.
# BAGEL-7B-MoT training entrypoint for FlagScale Megatron Native.
# Proper DP/TP data pipeline integration.

import os
import sys
import functools
from typing import Union

import torch
import torch.nn.functional as F

from megatron.training import get_args, print_rank_0
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training.training import pretrain
from megatron.core.enums import ModelType
from megatron.core import parallel_state

from flagscale.train.megatron.bagel_data import (
    get_batch,
    build_bagel_dataloaders,
)

# Global VAE model (loaded once, used in forward_step)
_VAE_MODEL = None
_VAE_PARAMS = None


def _get_vae_model():
    """Lazy load VAE model on all ranks."""
    global _VAE_MODEL, _VAE_PARAMS
    if _VAE_MODEL is None:
        args = get_args()
        vae_path = os.path.join(getattr(args, "bagel_model_path", ""), "ae.safetensors")
        if os.path.exists(vae_path):
            from flagscale.models.megatron.bagel.autoencoder import load_vae
            _VAE_MODEL, _VAE_PARAMS = load_vae(vae_path)
            _VAE_MODEL = _VAE_MODEL.to(device=torch.cuda.current_device(), dtype=torch.float32)
            _VAE_MODEL.eval()
            print_rank_0(f"[VAE] Loaded from {vae_path}")
    return _VAE_MODEL


def model_provider(pre_process=True, post_process=True):
    """Build the BAGEL model."""
    from flagscale.models.megatron.bagel.bagel_model import BagelModel
    args = get_args()
    config = core_transformer_config_from_args(args)

    bagel_config = {
        "vocab_size": args.padded_vocab_size,
        "rope_theta": getattr(args, "rotary_base", 1000000.0),
        "max_latent_size": getattr(args, "max_latent_size", 64),
        "latent_patch_size": getattr(args, "latent_patch_size", 2),
        "vae_z_channels": getattr(args, "vae_z_channels", 16),
        "vae_downsample": getattr(args, "vae_downsample", 8),
        "connector_act": "gelu_pytorch_tanh",
        "vit_config": {
            "hidden_size": getattr(args, "vit_hidden_size", 1152),
            "intermediate_size": getattr(args, "vit_intermediate_size", 4304),
            "num_hidden_layers": getattr(args, "vit_num_layers", 26),
            "num_attention_heads": getattr(args, "vit_num_attention_heads", 16),
            "num_channels": 3,
            "image_size": getattr(args, "vit_image_size", 980),
            "patch_size": getattr(args, "vit_patch_size", 14),
        },
    }

    model = BagelModel(config=config, bagel_config=bagel_config)
    print_rank_0(f"BAGEL model: {sum(p.numel() for p in model.parameters())/1e9:.2f}B params")
    return model


def loss_func(output_dict):
    """Loss function matching Megatron convention.
    
    Returns:
        (loss, {'lm loss': reporting_loss, 'ce_loss': ..., 'mse_loss': ...})
    """
    ce_loss = output_dict.get("ce_loss", None)
    mse_loss = output_dict.get("mse_loss", None)
    
    # Compute total loss from components
    if ce_loss is not None and mse_loss is not None:
        total_loss = ce_loss + mse_loss
    elif ce_loss is not None:
        total_loss = ce_loss
    elif mse_loss is not None:
        total_loss = mse_loss
    else:
        # No loss computed (e.g., empty batch) - return zero
        total_loss = torch.tensor(0.0, device=torch.cuda.current_device(), requires_grad=True)

    return total_loss, {
        "lm loss": total_loss.clone().detach(),
        "ce_loss": (ce_loss.clone().detach() if ce_loss is not None else torch.tensor(0.0)),
        "mse_loss": (mse_loss.clone().detach() if mse_loss is not None else torch.tensor(0.0)),
    }


def forward_step(data_iterator, model):
    """Forward step with proper DP/TP data handling.
    
    - get_batch handles TP broadcast internally
    - Returns (output_dict, loss_func) per Megatron convention
    """
    batch = get_batch(data_iterator)

    if batch is None:
        dummy = {
            "total_loss": torch.tensor(0.0, device=torch.cuda.current_device()),
            "ce_loss": torch.tensor(0.0, device=torch.cuda.current_device()),
            "mse_loss": torch.tensor(0.0, device=torch.cuda.current_device()),
        }
        return dummy, loss_func

    # Determine mode based on batch content
    has_gen = "packed_vae_token_indexes" in batch and batch["packed_vae_token_indexes"] is not None
    mode = "full" if has_gen else "und"

    # Compute cu_seqlens and max_seqlen from actual index tensors
    # sample_lens from data pipeline may not match actual total_seq_len (padding/truncation issues)
    # Instead, compute total_seq_len from the index tensors directly
    packed_text_indexes = batch["packed_text_indexes"]
    packed_vit_token_indexes = batch.get("packed_vit_token_indexes")
    packed_vae_token_indexes = batch.get("packed_vae_token_indexes")
    
    max_idx = packed_text_indexes.max().item() if len(packed_text_indexes) > 0 else -1
    if packed_vit_token_indexes is not None and len(packed_vit_token_indexes) > 0:
        max_idx = max(max_idx, packed_vit_token_indexes.max().item())
    if packed_vae_token_indexes is not None and len(packed_vae_token_indexes) > 0:
        max_idx = max(max_idx, packed_vae_token_indexes.max().item())
    actual_total_seq_len = max_idx + 1
    
    # For packed data (micro_batch_size=1), treat entire packed sequence as one "sample"
    # All tokens within the packed sequence can attend to each other (causal masking handles the rest)
    cu_seqlens = torch.tensor([0, actual_total_seq_len], dtype=torch.int32, device=torch.cuda.current_device())
    max_seqlen = actual_total_seq_len

    # Compute vit_cu_seqlens and vit_max_seqlen from vit_token_seqlens
    vit_token_seqlens = batch.get("vit_token_seqlens")
    if vit_token_seqlens is not None and len(vit_token_seqlens) > 0:
        if isinstance(vit_token_seqlens, torch.Tensor):
            vit_token_seqlens = vit_token_seqlens.tolist()
        vit_cu_seqlens = torch.tensor([0] + list(torch.cumsum(torch.tensor(vit_token_seqlens), dim=0)), 
                                       dtype=torch.int32, device=torch.cuda.current_device())
        vit_max_seqlen = int(max(vit_token_seqlens))
    else:
        vit_cu_seqlens = None
        vit_max_seqlen = None

    # VAE encode: convert padded_images to padded_latents BEFORE model forward
    padded_images = batch.get("padded_images")
    padded_latents = None
    if padded_images is not None and padded_images.numel() > 0:
        vae_model = _get_vae_model()
        if vae_model is not None:
            with torch.no_grad():
                old_enabled = torch.backends.cudnn.enabled
                torch.backends.cudnn.enabled = False
                padded_latents = vae_model.encode(padded_images.to(
                    device=torch.cuda.current_device(), dtype=torch.float32))
                torch.backends.cudnn.enabled = old_enabled
                print_rank_0(f"[VAE] encoded padded_images {padded_images.shape} -> latents {padded_latents.shape}")

    # Forward through model (map data pipeline keys to model forward params)
    output_dict = model(
        packed_input_ids=batch.get("packed_text_ids"),  # data key: packed_text_ids
        packed_position_ids=batch.get("packed_position_ids"),
        cu_seqlens=cu_seqlens,
        max_seqlen=max_seqlen,
        packed_text_indexes=batch.get("packed_text_indexes"),
        packed_vit_token_indexes=batch.get("packed_vit_token_indexes"),
        packed_vae_token_indexes=batch.get("packed_vae_token_indexes"),
        packed_pixel_values=batch.get("packed_vit_tokens"),  # data key: packed_vit_tokens
        packed_flattened_position_ids=batch.get("packed_vit_position_ids"),  # data key: packed_vit_position_ids
        vit_cu_seqlens=vit_cu_seqlens,
        vit_max_seqlen=vit_max_seqlen,
        padded_latents=padded_latents,  # pre-encoded VAE latents
        patchified_vae_latent_shapes=batch.get("patchified_vae_latent_shapes"),
        packed_latent_position_ids=batch.get("packed_latent_position_ids"),
        packed_timesteps=batch.get("packed_timesteps"),
        mse_loss_indexes=batch.get("mse_loss_indexes"),
        packed_labels=batch.get("packed_label_ids"),  # data key: packed_label_ids
        ce_loss_indexes=batch.get("ce_loss_indexes"),  # integer indices for CE loss positions
        mode=mode,
    )

    return output_dict, loss_func


def train_valid_test_dataloaders_provider(train_val_test_num_samples):
    """Build dataloaders with proper DP sharding.
    
    - Each DP rank gets different data (sharded by file)
    - TP ranks within same DP group share data (via broadcast in get_batch)
    - Only dataloader ranks (first/last PP stage) create dataloaders
    """
    args = get_args()

    # Only create dataloader on ranks that need data
    # For no PP: all ranks need data. For PP: only first stage.
    pp_rank = parallel_state.get_pipeline_model_parallel_rank()
    if pp_rank != 0:
        return None, None, None

    # DP rank/world_size for data sharding
    dp_rank = parallel_state.get_data_parallel_rank()
    dp_world_size = parallel_state.get_data_parallel_world_size()

    # Only TP rank 0 actually loads data; others get via broadcast
    tp_rank = parallel_state.get_tensor_model_parallel_rank()
    if tp_rank != 0:
        return None, None, None

    # Build dataloader using BAGEL's data pipeline
    train_dataloader = build_bagel_dataloaders(
        bagel_src_dir=getattr(args, "bagel_src_dir", None),
        model_path=getattr(args, "bagel_model_path", None),
        dataset_config_file=getattr(args, "dataset_config_file", None),
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    return train_dataloader, None, None


def extra_args_provider(parser):
    """Add BAGEL-specific arguments."""
    group = parser.add_argument_group(title="BAGEL")
    group.add_argument("--bagel-model-path", type=str, default=None,
                       help="Path to HF BAGEL model (for tokenizer/processor)")
    group.add_argument("--bagel-src-dir", type=str, default=None,
                       help="Path to BAGEL source code (for data pipeline)")
    group.add_argument("--dataset-config-file", type=str, default=None,
                       help="Path to BAGEL dataset config YAML")
    group.add_argument("--max-latent-size", type=int, default=64)
    group.add_argument("--latent-patch-size", type=int, default=2)
    group.add_argument("--vae-z-channels", type=int, default=16)
    group.add_argument("--vae-downsample", type=int, default=8)
    group.add_argument("--vit-hidden-size", type=int, default=1152)
    group.add_argument("--vit-intermediate-size", type=int, default=4304)
    group.add_argument("--vit-num-layers", type=int, default=26)
    group.add_argument("--vit-num-attention-heads", type=int, default=16)
    group.add_argument("--vit-image-size", type=int, default=980)
    group.add_argument("--vit-patch-size", type=int, default=14)
    return parser


if __name__ == "__main__":
    train_valid_test_dataloaders_provider.is_distributed = True

    pretrain(
        train_valid_test_dataloaders_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        args_defaults={'tokenizer_type': 'NullTokenizer', 'vocab_size': 152064},
        extra_args_provider=extra_args_provider,
    )
