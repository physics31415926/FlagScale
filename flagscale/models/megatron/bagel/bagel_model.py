# Copyright (c) 2025, BAAI. All rights reserved.
# Top-level BAGEL model for FlagScale Megatron Native training.
# Combines: Qwen2 MoT LLM backbone + SigLIP ViT + VAE connectors.

import torch
from torch import nn
import torch.nn.functional as F

from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig

from .mot_layer import MoTTransformerLayer, Qwen2RMSNorm, Qwen2RotaryEmbedding
from .siglip_vit import SiglipVisionModel, SiglipVisionConfig
from .connectors import MLPconnector, TimestepEmbedder, PositionEmbedding


class BagelModel(MegatronModule):
    """BAGEL-7B-MoT: Multimodal model with visual understanding and generation.
    
    Architecture:
    - LLM: Modified Qwen2 with MoT (dual attention/MLP per layer)
    - ViT: SigLIP vision encoder (replicated, no TP)
    - VAE: Flux autoencoder (frozen, loaded separately in forward_step)
    - Connectors: MLPconnector, vae2llm, llm2vae, TimestepEmbedder, PositionEmbedding
    
    Flow Matching (MSE loss):
    - VAE encodes target images -> clean latents
    - Patchify clean latents -> sequence of patch tokens
    - Sample noise, interpolate: noisy = (1-t)*clean + t*noise
    - Embed noisy latents into LLM hidden space
    - After transformer, predict velocity: llm2vae(hidden) -> predicted
    - Target = noise - clean (velocity v_t = dx_t/dt = x_1 - x_0)
    - MSE loss on velocity prediction
    """

    # Timestep shift factor (from original BAGEL config)
    TIMESTEP_SHIFT = 1.0

    def __init__(self, config: TransformerConfig, bagel_config: dict):
        super().__init__(config=config)
        self.config = config
        self.bagel_config = bagel_config
        self.hidden_size = config.hidden_size
        self.vocab_size = bagel_config.get("vocab_size", 152064)
        self.num_layers = config.num_layers
        self.max_latent_size = bagel_config.get("max_latent_size", 64)
        self.latent_patch_size = bagel_config.get("latent_patch_size", 2)
        self.vae_z_channels = bagel_config.get("vae_z_channels", 16)
        self.vae_downsample = bagel_config.get("vae_downsample", 8)

        # Compute patch_latent_dim for VAE latent tokens
        self.patch_latent_dim = self.vae_z_channels * self.latent_patch_size * self.latent_patch_size

        # Gradient checkpointing (saves activation memory at cost of recomputation)
        self.gradient_checkpointing = True

        # === LLM Backbone (Qwen2 MoT) ===
        self.embed_tokens = nn.Embedding(self.vocab_size, self.hidden_size)
        self.layers = nn.ModuleList([
            MoTTransformerLayer(config, has_gen_path=True) for i in range(self.num_layers)
        ])
        self.norm = Qwen2RMSNorm(self.hidden_size, eps=1e-6)
        self.norm_moe_gen = Qwen2RMSNorm(self.hidden_size, eps=1e-6)
        self.lm_head = ColumnParallelLinear(
            self.hidden_size, self.vocab_size,
            config=config, init_method=config.init_method,
            bias=False, gather_output=True,
        )

        # Rotary embedding
        head_dim = self.hidden_size // config.num_attention_heads
        self.rotary_emb = Qwen2RotaryEmbedding(
            dim=head_dim,
            max_position_embeddings=32768,
            base=bagel_config.get("rope_theta", 1000000.0),
        )

        # === Vision Encoder (SigLIP) ===
        vit_cfg = bagel_config.get("vit_config", {})
        self.vit_config = SiglipVisionConfig(
            hidden_size=vit_cfg.get("hidden_size", 1152),
            intermediate_size=vit_cfg.get("intermediate_size", 4304),
            num_hidden_layers=vit_cfg.get("num_hidden_layers", 26),
            num_attention_heads=vit_cfg.get("num_attention_heads", 16),
            num_channels=vit_cfg.get("num_channels", 3),
            image_size=vit_cfg.get("image_size", 980),
            patch_size=vit_cfg.get("patch_size", 14),
        )
        self.vit_model = SiglipVisionModel(self.vit_config)

        # === Connectors ===
        connector_act = bagel_config.get("connector_act", "gelu_pytorch_tanh")
        self.connector = MLPconnector(
            in_dim=self.vit_config.hidden_size,
            out_dim=self.hidden_size,
            hidden_act=connector_act,
        )
        # VAE <-> LLM connectors
        self.vae2llm = nn.Linear(self.patch_latent_dim, self.hidden_size)
        self.llm2vae = nn.Linear(self.hidden_size, self.patch_latent_dim)

        # Timestep and position embeddings for generation
        self.timestep_embedder = TimestepEmbedder(self.hidden_size)
        self.vae_position_embedding = PositionEmbedding(
            max_num_patch_per_side=self.max_latent_size,
            hidden_size=self.hidden_size,
        )
        # ViT position embedding (in LLM hidden space, added AFTER connector projection)
        vit_max_patches_per_side = vit_cfg.get('image_size', 980) // vit_cfg.get('patch_size', 14)
        self.vit_pos_embed = PositionEmbedding(
            max_num_patch_per_side=vit_max_patches_per_side,
            hidden_size=self.hidden_size,  # LLM hidden size (3584)
        )

    def encode_images(self, packed_pixel_values, packed_flattened_position_ids, cu_seqlens, max_seqlen):
        """Encode images through ViT + connector."""
        vit_output = self.vit_model(packed_pixel_values, packed_flattened_position_ids, cu_seqlens, max_seqlen)
        return self.connector(vit_output)

    def patchify_and_flow_match(self, padded_latents, patchified_vae_latent_shapes,
                                packed_latent_position_ids, packed_timesteps):
        """Patchify VAE latents and apply flow matching (noise interpolation).
        
        This implements the core flow matching logic from original BAGEL:
        1. Patchify each latent using its actual (h, w) shape
        2. Sample noise
        3. Apply timestep shift: t = shift*sigmoid(t) / (1 + (shift-1)*sigmoid(t))
        4. Interpolate: noisy = (1-t)*clean + t*noise
        5. Embed: vae2llm(noisy) + time_embed(t) + pos_embed(pos_ids)
        
        Args:
            padded_latents: [N, z_channels, H_max, W_max] VAE-encoded latents (padded)
            patchified_vae_latent_shapes: list of (h, w) tuples per image (patchified spatial dims)
            packed_latent_position_ids: [num_latent_tokens] position IDs
            packed_timesteps: [num_latent_tokens] raw timesteps (from randn, per-token)
            
        Returns:
            latent_embeds: [num_latent_tokens, hidden_size] embedded noisy latents
            velocity_target: [num_latent_tokens, patch_latent_dim] target for MSE loss
        """
        p = self.latent_patch_size
        device = padded_latents.device
        dtype = padded_latents.dtype

        # Step 1: Patchify each image's latent using its actual shape
        packed_latent_list = []
        for i, (h, w) in enumerate(patchified_vae_latent_shapes):
            # latent[i] shape: [C, H_padded, W_padded], crop to actual [C, h*p, w*p]
            latent = padded_latents[i, :, :h * p, :w * p]  # [C, h*p, w*p]
            # Reshape to patches: [C, h, p, w, p] -> [h, w, p, p, C] -> [h*w, p*p*C]
            latent = latent.reshape(self.vae_z_channels, h, p, w, p)
            latent = torch.einsum("chpwq->hwpqc", latent).reshape(-1, self.patch_latent_dim)
            packed_latent_list.append(latent)
        
        packed_latent_clean = torch.cat(packed_latent_list, dim=0)  # [num_latent_tokens, patch_latent_dim]

        # Step 2: Sample noise
        noise = torch.randn_like(packed_latent_clean)

        # Step 3: Apply timestep sigmoid + shift
        # packed_timesteps is per-token (same value for all tokens of one image)
        t = torch.sigmoid(packed_timesteps.float())  # [num_latent_tokens]
        t = self.TIMESTEP_SHIFT * t / (1 + (self.TIMESTEP_SHIFT - 1) * t)

        # Step 4: Flow interpolation: noisy = (1-t)*clean + t*noise
        packed_latent_noisy = (1 - t[:, None]) * packed_latent_clean + t[:, None] * noise

        # Step 5: Embed noisy latents
        packed_latent_noisy = packed_latent_noisy.to(self.vae2llm.weight.dtype)
        latent_embeds = self.vae2llm(packed_latent_noisy)
        
        # Add position embedding
        pos_embeds = self.vae_position_embedding(packed_latent_position_ids)
        latent_embeds = latent_embeds + pos_embeds
        
        # Add timestep embedding
        t_for_embed = t.to(self.timestep_embedder.mlp[0].weight.dtype)
        t_embeds = self.timestep_embedder(t_for_embed)  # [num_latent_tokens, hidden_size]
        latent_embeds = latent_embeds + t_embeds

        # Velocity target: v = x_1 - x_0 = noise - clean (pointing from data to noise)
        velocity_target = noise - packed_latent_clean

        return latent_embeds, velocity_target, t

    def forward(
        self,
        # Token inputs
        packed_input_ids=None,
        packed_position_ids=None,
        # Sequence packing info
        cu_seqlens=None,
        max_seqlen=None,
        # Token type indexes
        packed_text_indexes=None,
        packed_vit_token_indexes=None,
        packed_vae_token_indexes=None,
        # ViT inputs
        packed_pixel_values=None,
        packed_flattened_position_ids=None,
        vit_cu_seqlens=None,
        vit_max_seqlen=None,
        # VAE / Flow matching inputs
        padded_latents=None,
        patchified_vae_latent_shapes=None,
        packed_latent_position_ids=None,
        packed_timesteps=None,
        mse_loss_indexes=None,
        # Labels for CE loss
        packed_labels=None,
        ce_loss_indexes=None,
        # Mode
        mode="und",
    ):
        """
        Forward pass for BAGEL model.
        
        Returns dict with 'ce_loss' and/or 'mse_loss'.
        """
        # === Build hidden states ===
        assert packed_input_ids.max().item() < self.vocab_size, \
            f"Token ID {packed_input_ids.max().item()} >= vocab_size {self.vocab_size}"
        text_embeds = self.embed_tokens(packed_input_ids)

        # Compute total sequence length from all indexes
        max_idx = packed_text_indexes.max().item() if packed_text_indexes is not None and len(packed_text_indexes) > 0 else 0
        if packed_vit_token_indexes is not None and len(packed_vit_token_indexes) > 0:
            max_idx = max(max_idx, packed_vit_token_indexes.max().item())
        if packed_vae_token_indexes is not None and len(packed_vae_token_indexes) > 0:
            max_idx = max(max_idx, packed_vae_token_indexes.max().item())
        total_seq_len = max_idx + 1

        hidden_states = torch.zeros(
            total_seq_len, self.config.hidden_size,
            dtype=text_embeds.dtype, device=text_embeds.device
        )

        # Scatter text embeddings
        hidden_states[packed_text_indexes] = text_embeds

        # Inject ViT features for understanding
        if packed_pixel_values is not None and packed_vit_token_indexes is not None and len(packed_vit_token_indexes) > 0:
            vit_features = self.encode_images(
                packed_pixel_values, packed_flattened_position_ids, vit_cu_seqlens, vit_max_seqlen
            )
            # Add ViT position embedding (in LLM space, after connector)
            vit_pos_emb = self.vit_pos_embed(packed_flattened_position_ids)
            vit_features = vit_features + vit_pos_emb
            hidden_states[packed_vit_token_indexes] = vit_features.to(hidden_states.dtype)

        # Inject VAE latent features for generation (with flow matching)
        velocity_target = None
        flow_timesteps = None
        if (padded_latents is not None and packed_vae_token_indexes is not None 
                and len(packed_vae_token_indexes) > 0 and patchified_vae_latent_shapes is not None):
            latent_embeds, velocity_target, flow_timesteps = self.patchify_and_flow_match(
                padded_latents, patchified_vae_latent_shapes,
                packed_latent_position_ids, packed_timesteps
            )
            hidden_states[packed_vae_token_indexes] = latent_embeds.to(hidden_states.dtype)

        # === Compute RoPE ===
        cos, sin = self.rotary_emb(packed_position_ids)

        # === Determine und/gen token indexes for MoT routing ===
        und_idx = packed_text_indexes
        if packed_vit_token_indexes is not None and len(packed_vit_token_indexes) > 0:
            und_idx = torch.cat([packed_text_indexes, packed_vit_token_indexes], dim=0)
        gen_idx = packed_vae_token_indexes if mode == "full" else None

        # === Run through transformer layers ===
        for i, layer in enumerate(self.layers):
            if self.training and self.gradient_checkpointing:
                def create_custom_forward(module, _cu, _ms, _cos, _sin, _ui, _gi, _mode):
                    def custom_forward(hs):
                        return module(hs, _cu, _ms, _cos, _sin, _ui, _gi, _mode)
                    return custom_forward
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer, cu_seqlens, max_seqlen, cos, sin, und_idx, gen_idx, mode),
                    hidden_states,
                    use_reentrant=True,
                )
            else:
                hidden_states = layer(
                    hidden_states, cu_seqlens, max_seqlen, cos, sin,
                    und_idx, gen_idx, mode=mode,
                )

        # === Final norm ===
        hidden_states = self.norm(hidden_states)

        # === Compute losses ===
        losses = {}

        # CE loss (text prediction)
        if packed_labels is not None and ce_loss_indexes is not None and len(ce_loss_indexes) > 0:
            ce_hidden = hidden_states[ce_loss_indexes]
            logits, _ = self.lm_head(ce_hidden)
            # Truncate logits to actual vocab_size (lm_head may pad for TP alignment)
            if logits.shape[-1] > self.vocab_size:
                logits = logits[:, :self.vocab_size]
            ce_loss = F.cross_entropy(logits, packed_labels, reduction="mean")
            losses["ce_loss"] = ce_loss

        # MSE loss (flow matching velocity prediction)
        if velocity_target is not None and mse_loss_indexes is not None and len(mse_loss_indexes) > 0:
            # Get hidden states at MSE loss positions
            mse_hidden = hidden_states[mse_loss_indexes]
            # Predict velocity in latent space
            predicted_velocity = self.llm2vae(mse_hidden)
            # Only compute loss where timestep > 0 (after sigmoid, this means original t != -inf)
            # flow_timesteps corresponds to packed_vae_token_indexes positions
            # mse_loss_indexes is a subset of packed_vae_token_indexes
            # We need to find which flow_timesteps correspond to mse_loss_indexes
            # Since mse_loss_indexes are positions in the full sequence, and packed_vae_token_indexes
            # maps VAE token positions, we need the mapping.
            # In original BAGEL: mse_loss_indexes selects from total_seq_len, and velocity_target
            # is aligned with packed_vae_token_indexes. We need to select the subset.
            
            # Build mapping: for each position in mse_loss_indexes, find its index in packed_vae_token_indexes
            # This gives us the corresponding velocity_target row
            # Since mse_loss_indexes ⊆ packed_vae_token_indexes (both are positions in full sequence),
            # we can use a set lookup
            vae_pos_to_idx = {}
            for idx, pos in enumerate(packed_vae_token_indexes.tolist()):
                vae_pos_to_idx[pos] = idx
            
            target_indices = []
            for pos in mse_loss_indexes.tolist():
                if pos in vae_pos_to_idx:
                    target_indices.append(vae_pos_to_idx[pos])
            target_indices = torch.tensor(target_indices, device=velocity_target.device, dtype=torch.long)
            
            # Select corresponding velocity targets and timesteps
            selected_target = velocity_target[target_indices]
            selected_timesteps = flow_timesteps[target_indices]
            
            # Only compute MSE where timestep > 0 (t=0 means use clean image, no denoising needed)
            has_mse = selected_timesteps > 0
            if has_mse.any():
                mse_loss = F.mse_loss(
                    predicted_velocity[has_mse],
                    selected_target[has_mse].to(predicted_velocity.dtype),
                    reduction="mean",
                )
                losses["mse_loss"] = mse_loss

        return losses

    def set_input_tensor(self, input_tensor):
        """Required by Megatron pipeline parallelism (unused for PP=1)."""
        pass
