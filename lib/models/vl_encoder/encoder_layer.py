from typing import Optional, Tuple
from dataclasses import dataclass

import torch
from torch import nn, Tensor

from transformers.utils import logging, ModelOutput
from transformers.activations import ACT2FN
from transformers.pytorch_utils import meshgrid

from ..multi_scale_deformable_attention import MSDeformAttn
from ..configuration_mono3dvg_v2 import Mono3DVGv2Config

logger = logging.get_logger(__name__)


@dataclass
class VisionLanguageModelOutput(ModelOutput):
    """
    Vision Language Model Outputs that contains vision and language outputs.
    
    Args:
        last_vision_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Sequence of hidden-states at the vision output of the last layer of the model.
        last_text_hidden_state (`torch.FloatTensor` of shape `(batch_size, text_length, hidden_size)`):
            Sequence of hidden-states at the text output of the last layer of the model.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """
    last_vision_hidden_state: torch.FloatTensor = None
    last_text_hidden_state: torch.FloatTensor = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


def get_reference_points(spatial_shapes, valid_ratios, device):
    """
    Get reference points for each feature map. Used in decoder.

    Args:
        spatial_shapes (`torch.LongTensor` of shape `(num_feature_levels, 2)`):
            Spatial shapes of each feature map.
        valid_ratios (`torch.FloatTensor` of shape `(batch_size, num_feature_levels, 2)`):
            Valid ratios of each feature map.
        device (`torch.device`):
            Device on which to create the tensors.
    Returns:
        `torch.FloatTensor` of shape `(batch_size, num_queries, num_feature_levels, 2)`
    """
    reference_points_list = []
    for level, (height, width) in enumerate(spatial_shapes):
        ref_y, ref_x = meshgrid(
            torch.linspace(0.5, height - 0.5, height, dtype=torch.float32, device=device),
            torch.linspace(0.5, width - 0.5, width, dtype=torch.float32, device=device),
            indexing="ij",
        )
        # TODO: valid_ratios could be useless here. check https://github.com/fundamentalvision/Deformable-DETR/issues/36
        ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, level, 1] * height)
        ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, level, 0] * width)
        ref = torch.stack((ref_x, ref_y), -1)
        reference_points_list.append(ref)
    reference_points = torch.cat(reference_points_list, 1)
    reference_points = reference_points[:, :, None] * valid_ratios[:, None]
    return reference_points


class DeformableDetrEncoderLayer(nn.Module):
    def __init__(self, config: Mono3DVGv2Config):
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = MSDeformAttn(
            config, num_heads=config.encoder_attention_heads, n_points=config.encoder_n_points
        )
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_embeddings: torch.Tensor = None,
        reference_points=None,
        spatial_shapes=None,
        level_start_index=None,
        output_attentions: bool = False,
    ):
        """
        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Input to the layer.
            attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
                Attention mask.
            position_embeddings (`torch.FloatTensor`, *optional*):
                Position embeddings, to be added to `hidden_states`.
            reference_points (`torch.FloatTensor`, *optional*):
                Reference points.
            spatial_shapes (`torch.LongTensor`, *optional*):
                Spatial shapes of the backbone feature maps.
            level_start_index (`torch.LongTensor`, *optional*):
                Level start index.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states

        # Apply Multi-scale Deformable Attention Module on the multi-scale feature maps.
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            output_attentions=output_attentions,
        )

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)

        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        if self.training:
            if torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any():
                clamp_value = torch.finfo(hidden_states.dtype).max - 1000
                hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class VisionCrossEncoderLayer(nn.Module):
    """Cross-modal encoder layer including image self-attention and text-to-image cross attention."""
    def __init__(self, config: Mono3DVGv2Config):
        super().__init__()
        self.embed_dim = config.d_model
        
        # image self attention
        self.self_attn = MSDeformAttn(
            config, num_heads=config.encoder_attention_heads, n_points=config.encoder_n_points
        )
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropout = config.dropout
        # text-to-image cross attention
        self.cross_attn_text = nn.MultiheadAttention(self.embed_dim, config.encoder_attention_heads, dropout=config.dropout, batch_first=True)
        self.cross_attn_text_layer_norm = nn.LayerNorm(self.embed_dim)
        
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def with_pos_embed(self, tensor: torch.Tensor, position_embeddings: Optional[Tensor]):
        return tensor if position_embeddings is None else tensor + position_embeddings

    def forward(
        self,
        vision_embeds: torch.Tensor,
        vision_attention_mask: torch.Tensor,
        text_embeds: torch.Tensor,
        text_attention_mask: torch.Tensor,
        position_embeddings: torch.Tensor = None,
        reference_points=None,
        spatial_shapes=None,
        level_start_index=None,
        output_attentions: bool = False,
    ):
        """
        Args:
            vision_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Input to the layer.
            vision_attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
                Attention mask.
            text_embeds (`torch.FloatTensor` of shape `(batch_size, text_length, hidden_size)`):
                Input to the layer.
            text_attention_mask (`torch.FloatTensor` of shape `(batch_size, text_length)`):
                Text attention mask.
            position_embeddings (`torch.FloatTensor`, *optional*):
                Position embeddings, to be added to `vision_embeds`.
            reference_points (`torch.FloatTensor`, *optional*):
                Reference points.
            spatial_shapes (`torch.LongTensor`, *optional*):
                Spatial shapes of the backbone feature maps.
            level_start_index (`torch.LongTensor`, *optional*):
                Level start index.
            output_attentions (`bool`, *optional*):
                Whether or not to return the cross attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = vision_embeds

        # Apply Multi-scale Deformable Attention Module on the multi-scale feature maps.
        vision_embeds, self_attn_weights = self.self_attn(
            hidden_states=vision_embeds,
            attention_mask=vision_attention_mask,
            encoder_hidden_states=vision_embeds,
            encoder_attention_mask=vision_attention_mask,
            position_embeddings=position_embeddings,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            output_attentions=output_attentions,
        )

        vision_embeds = nn.functional.dropout(vision_embeds, p=self.dropout, training=self.training)
        vision_embeds = residual + vision_embeds
        vision_embeds = self.self_attn_layer_norm(vision_embeds)

        # text-to-image cross attention
        residual = vision_embeds
        vision_embeds, cross_attn_weights = self.cross_attn_text(
            query=self.with_pos_embed(vision_embeds, position_embeddings),
            key=text_embeds,
            value=text_embeds,
            key_padding_mask=text_attention_mask,
        )
        vision_embeds = nn.functional.dropout(vision_embeds, p=self.dropout, training=self.training)
        vision_embeds = residual + vision_embeds
        vision_embeds = self.cross_attn_text_layer_norm(vision_embeds)

        residual = vision_embeds
        vision_embeds = self.activation_fn(self.fc1(vision_embeds))
        vision_embeds = nn.functional.dropout(vision_embeds, p=self.activation_dropout, training=self.training)

        vision_embeds = self.fc2(vision_embeds)
        vision_embeds = nn.functional.dropout(vision_embeds, p=self.dropout, training=self.training)

        vision_embeds = residual + vision_embeds
        vision_embeds = self.final_layer_norm(vision_embeds)

        if self.training:
            if torch.isinf(vision_embeds).any() or torch.isnan(vision_embeds).any():
                clamp_value = torch.finfo(vision_embeds.dtype).max - 1000
                vision_embeds = torch.clamp(vision_embeds, min=-clamp_value, max=clamp_value)

        outputs = (vision_embeds,)

        if output_attentions:
            outputs += (cross_attn_weights,)

        return outputs


class LanguageCrossEncoderLayer(nn.Module):
    """Cross-modal encoder layer including text self-attention and image-to-text cross attention."""
    def __init__(self, config: Mono3DVGv2Config):
        super().__init__()
        self.embed_dim = config.d_model
        
        # text self attention
        self.self_attn = nn.MultiheadAttention(self.embed_dim, config.encoder_attention_heads, dropout=config.dropout, batch_first=True)
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropout = config.dropout
        # image-to-text cross attention
        self.cross_attn_img = nn.MultiheadAttention(self.embed_dim, config.encoder_attention_heads, dropout=config.dropout, batch_first=True)
        self.cross_attn_img_layer_norm = nn.LayerNorm(self.embed_dim)
        
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)
    
    def with_pos_embed(self, tensor: torch.Tensor, position_embeddings: Optional[Tensor]):
        return tensor if position_embeddings is None else tensor + position_embeddings
    
    def forward(
        self,
        vision_embeds: torch.Tensor,
        vision_attention_mask: torch.Tensor,
        text_embeds: torch.Tensor,
        text_attention_mask: torch.Tensor,
        position_embeddings: torch.Tensor = None,
        output_attentions: bool = False,
    ):
        """
        Args:
            vision_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Input to the layer.
            vision_attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
                Attention mask.
            text_embeds (`torch.FloatTensor` of shape `(batch_size, text_length, hidden_size)`):
                Input to the layer.
            text_attention_mask (`torch.FloatTensor` of shape `(batch_size, text_length)`):
                Text attention mask.
            position_embeddings (`torch.FloatTensor`, *optional*):
                Position embeddings, to be added to `vision_embeds`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the cross attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = text_embeds

        # text self-attention
        text_embeds, self_attn_weights = self.self_attn(
            query=text_embeds,
            key=text_embeds,
            value=text_embeds,
            key_padding_mask=text_attention_mask,
        )

        text_embeds = nn.functional.dropout(text_embeds, p=self.dropout, training=self.training)
        text_embeds = residual + text_embeds
        text_embeds = self.self_attn_layer_norm(text_embeds)

        # image-to-text cross attention
        residual = text_embeds
        text_embeds, cross_attn_weights = self.cross_attn_img(
            query=text_embeds,
            key=self.with_pos_embed(vision_embeds, position_embeddings),
            value=self.with_pos_embed(vision_embeds, position_embeddings),
            key_padding_mask=vision_attention_mask,
        )
        text_embeds = nn.functional.dropout(text_embeds, p=self.dropout, training=self.training)
        text_embeds = residual + text_embeds
        text_embeds = self.cross_attn_img_layer_norm(text_embeds)

        residual = text_embeds
        text_embeds = self.activation_fn(self.fc1(text_embeds))
        text_embeds = nn.functional.dropout(text_embeds, p=self.activation_dropout, training=self.training)

        text_embeds = self.fc2(text_embeds)
        text_embeds = nn.functional.dropout(text_embeds, p=self.dropout, training=self.training)

        text_embeds = residual + text_embeds
        text_embeds = self.final_layer_norm(text_embeds)

        if self.training:
            if torch.isinf(text_embeds).any() or torch.isnan(text_embeds).any():
                clamp_value = torch.finfo(text_embeds.dtype).max - 1000
                text_embeds = torch.clamp(text_embeds, min=-clamp_value, max=clamp_value)

        outputs = (text_embeds,)

        if output_attentions:
            outputs += (cross_attn_weights,)

        return outputs

