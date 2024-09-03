import os
import copy
import math
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from transformers import PreTrainedModel, RobertaTokenizerFast
from transformers.utils import (
    ModelOutput, 
    is_scipy_available,
    requires_backends, 
)
from transformers.activations import ACT2FN

from .configuration_mono3dvg_v2 import Mono3DVGv2Config
from .depth_predictor import DDNLoss, DepthPredictor
from .multi_scale_deformable_attention import MSDeformAttn
from .vl_encoder import build_vision_language_encoder
from .backbone import (
    build_position_encoding, 
    build_vision_backbone, 
    VisionBackboneModel,
    build_language_backbone,
    LanguageBackboneModel,
    LearnedPositionEmbedding
)
from utils.box_ops import generalized_box_iou, box_cxcylrtb_to_xyxy, box_cxcywh_to_xyxy

if is_scipy_available():
    from scipy.optimize import linear_sum_assignment


@dataclass
class Mono3DVGv2DecoderOutput(ModelOutput):
    """
    Base class for outputs of the Mono3DVGv2Decoder. This class adds three attributes to
    BaseModelOutputWithCrossAttentions, namely:
    - a stacked tensor of intermediate decoder hidden states (i.e. the output of each decoder layer)
    - a stacked tensor of intermediate reference points.
    - a stacked tensor of intermediate object 3dimensions.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        intermediate_hidden_states (`torch.FloatTensor` of shape `(batch_size, config.decoder_layers, num_queries, hidden_size)`):
            Stacked intermediate hidden states (output of each layer of the decoder).
        intermediate_reference_points (`torch.FloatTensor` of shape `(batch_size, config.decoder_layers, sequence_length, hidden_size)`):
            Stacked intermediate reference points (reference points of each layer of the decoder).
        intermediate_reference_dims (`torch.FloatTensor` of shape `(batch_size, config.decoder_layers, sequence_length, hidden_size)`):
            Stacked intermediate object 3dimensions (reference 3dimensions of each layer of the decoder).
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the output of each layer
            plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
            the self-attention heads.
        cross_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` and `config.add_cross_attention=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights of the decoder's cross-attention layer, after the attention softmax,
            used to compute the weighted average in the cross-attention heads.
    """

    last_hidden_state: torch.FloatTensor = None
    intermediate_hidden_states: torch.FloatTensor = None
    intermediate_reference_points: torch.FloatTensor = None
    intermediate_reference_dims: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class Mono3DVGv2ModelOutput(ModelOutput):
    """
    Base class for outputs of the Mono 3DVGv2 encoder-decoder model.

    Args:
        init_reference_points (`torch.FloatTensor` of shape  `(batch_size, num_queries, 4)`):
            Initial reference points sent through the Transformer decoder.
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_queries, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the decoder of the model.
        intermediate_hidden_states (`torch.FloatTensor` of shape `(batch_size, config.decoder_layers, num_queries, hidden_size)`):
            Stacked intermediate hidden states (output of each layer of the decoder).
        intermediate_reference_points (`torch.FloatTensor` of shape `(batch_size, config.decoder_layers, num_queries, 4)`):
            Stacked intermediate reference points (reference points of each layer of the decoder).
        intermediate_reference_dims (`torch.FloatTensor` of shape `(batch_size, config.decoder_layers, sequence_length, hidden_size)`):
            Stacked intermediate object 3dimensions (reference 3dimensions of each layer of the decoder).
        decoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, num_queries, hidden_size)`. Hidden-states of the decoder at the output of each layer
            plus the initial embedding outputs.
        decoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, num_queries,
            num_queries)`. Attentions weights of the decoder, after the attention softmax, used to compute the weighted
            average in the self-attention heads.
        cross_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_queries, num_heads, 4, 4)`.
            Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the
            weighted average in the cross-attention heads.
        encoder_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder of the model.
        encoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_queries, num_heads, 4, 4)`.
            Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
        depth_logits (`torch.FloatTensor` of shape `(batch_size, num_bins + 1, img_height // 16, img_width // 16)`):
            Depth map logits predicted by the encoder_last_hidden_state.
        weighted_depth (`torch.FloatTensor` of shape `(batch_size, img_height // 16, img_width // 16)`):
            Weighted depth map predicted by the encoder_last_hidden_state.
    """
    
    init_reference_points: torch.FloatTensor = None
    last_hidden_state: torch.FloatTensor = None
    intermediate_hidden_states: torch.FloatTensor = None
    intermediate_reference_points: torch.FloatTensor = None
    intermediate_reference_dims: torch.FloatTensor = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    depth_logits: Optional[torch.FloatTensor] = None
    weighted_depth: Optional[torch.FloatTensor] = None


@dataclass
class Mono3DVGv2ForSingleObjectDetectionOutput(ModelOutput):
    """
    Output type of [`Mono3DVGv2ForSingleObjectDetection`].
    
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` are provided)):
            Total loss as a linear combination of a negative log-likehood (cross-entropy) for class prediction and a
            bounding box loss. The latter is defined as a linear combination of the L1 loss and the generalized
            scale-invariant IoU loss.
        loss_dict (`Dict`, *optional*):
            A dictionary containing the individual losses. Useful for logging.
        logits (`torch.FloatTensor` of shape `(batch_size, num_queries, num_classes + 1)`):
            Classification logits (including no-object) for all queries.
        pred_boxes (`torch.FloatTensor` of shape `(batch_size, num_queries, 6)`):
            Normalized boxes3d coordinates for all queries, represented as (center3d_x, center3d_y, l, r, b, t). These
            values are normalized in [0, 1], relative to the size of each individual image in the batch (disregarding
            possible padding). You can use [`~Mono3DVGImageProcessor.post_process_object_detection`] to retrieve the
            unnormalized bounding boxes.
        pred_3d_dim (`torch.FloatTensor` of shape `(batch_size, num_queries, 3)`):
            3D dimensions (h, w, l) for all queries.
        pred_depth (`torch.FloatTensor` of shape `(batch_size, num_queries, 2)`):
            Depth and depth_log_variance for all queries.
        pred_angle (`torch.FloatTensor` of shape `(batch_size, num_queries, 24)`):
            Angle classification first 12, and regression last 12 for all queries.
        pred_depth_map_logits (`torch.FloatTensor` of shape `(batch_size, num_bins + 1, img_height // 16, img_width // 16)`):
            Depth map logits for all queries.
        auxiliary_outputs (`list[Dict]`, *optional*):
            Optional, only returned when auxilary losses are activated (i.e. `config.auxiliary_loss` is set to `True`)
            and labels are provided. It is a list of dictionaries containing the two above keys (`logits` and
            `pred_boxes`) for each decoder layer.
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_queries, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the decoder of the model.
        decoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, num_queries, hidden_size)`. Hidden-states of the decoder at the output of each layer
            plus the initial embedding outputs.
        decoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, num_queries,
            num_queries)`. Attentions weights of the decoder, after the attention softmax, used to compute the weighted
            average in the self-attention heads.
        cross_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_queries, num_heads, 4, 4)`.
            Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the
            weighted average in the cross-attention heads.
        encoder_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder of the model.
        encoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, sequence_length, num_heads, 4,
            4)`. Attentions weights of the encoder, after the attention softmax, used to compute the weighted average
            in the self-attention heads.
        intermediate_hidden_states (`torch.FloatTensor` of shape `(batch_size, config.decoder_layers, num_queries, hidden_size)`):
            Stacked intermediate hidden states (output of each layer of the decoder).
        intermediate_reference_points (`torch.FloatTensor` of shape `(batch_size, config.decoder_layers, num_queries, 4)`):
            Stacked intermediate reference points (reference points of each layer of the decoder).
        init_reference_points (`torch.FloatTensor` of shape  `(batch_size, num_queries, 4)`):
            Initial reference points sent through the Transformer decoder.
    """
    
    loss: Optional[torch.FloatTensor] = None
    loss_dict: Optional[Dict] = None
    logits: torch.FloatTensor = None
    pred_boxes: torch.FloatTensor = None
    pred_3d_dim: torch.FloatTensor = None
    pred_depth: torch.FloatTensor = None
    pred_angle: torch.FloatTensor = None
    pred_depth_map_logits: torch.FloatTensor = None
    auxiliary_outputs: Optional[List[Dict]] = None
    init_reference_points: Optional[torch.FloatTensor] = None
    last_hidden_state: Optional[torch.FloatTensor] = None
    intermediate_hidden_states: Optional[torch.FloatTensor] = None
    intermediate_reference_points: Optional[torch.FloatTensor] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


class TextGuidedAdapter(nn.Module):
    def __init__(
        self, 
        config: Mono3DVGv2Config,
        dropout=0.1,
        n_heads=8,
        n_points=4,
    ):
        super().__init__()
        d_model = config.d_model

        # img2text: Cross attention
        self.img2text_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)

        self.adapt_proj = MLP(input_dim=256, hidden_dim=256, output_dim=256, num_layers=1)
        self.orig_proj = MLP(input_dim=256, hidden_dim=256, output_dim=256, num_layers=1)

        self.tf_pow = 2.0
        self.tf_scale = nn.Parameter(torch.Tensor([1.0]))
        self.tf_sigma = nn.Parameter(torch.Tensor([0.5]))

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.downsample = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, return_indices=False, ceil_mode=False)

        # img2img: Multi-Scale Deformable Attention
        self.img2img_msdeform_attn = MSDeformAttn(config, n_heads, n_points)

        self.norm_text_cond_img = nn.LayerNorm(d_model)
        self.norm_img = nn.LayerNorm(d_model)

        # depth2text: Cross attention
        self.depth2textcross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)

        # depth2depth: Cross attention
        self.depth2depth_attn =  nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)

        self.norm_text_cond_depth = nn.LayerNorm(d_model)
        self.norm_depth = nn.LayerNorm(d_model)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def get_reference_points(self, spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):

            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(
        self, 
        img_feat_src, 
        masks, 
        img_pos_embeds,
        spatial_shapes,
        level_start_index,
        src_valid_ratios,
        word_feat, 
        word_key_padding_mask,
        depth_pos_embed, 
        mask_depth,
        word_pos=None
    ):
        orig_multiscale_img_feat = img_feat_src
        orig_multiscale_masks = masks
        orig_multiscale_img_pos_embeds = img_pos_embeds

        # split four level multi-scale img_feat/masks/img_pos_embeds
        bs, sum, dim = img_feat_src.shape
        img_feat_src_list = img_feat_src.split([H_ * W_ for H_, W_ in spatial_shapes], dim=1)
        masks_list = masks.split([H_ * W_ for H_, W_ in spatial_shapes], dim=1)
        img_pos_embeds_list = img_pos_embeds.split([H_ * W_ for H_, W_ in spatial_shapes], dim=1)

        # For second level img_feat/masks/img_pos_embeds to compute score
        img_feat_src = img_feat_src_list[1]
        masks = masks_list[1]
        img_pos_embeds = img_pos_embeds_list[1]

        imgfeat_adapt, _ = self.img2text_attn(
            query=self.with_pos_embed(img_feat_src, img_pos_embeds),
            key=self.with_pos_embed(word_feat, word_pos),
            value=word_feat,
            key_padding_mask=word_key_padding_mask
        )

        imgfeat_adapt_embed = self.adapt_proj(imgfeat_adapt)  # [bs, 1920, 256]
        imgfeat_orig_embed = self.orig_proj(img_feat_src)

        verify_score = (F.normalize(imgfeat_orig_embed, p=2, dim=-1) *
                        F.normalize(imgfeat_adapt_embed, p=2, dim=-1)).sum(dim=-1, keepdim=True)
        verify_score = self.tf_scale * \
                       torch.exp( - (1 - verify_score).pow(self.tf_pow) \
                        / (2 * self.tf_sigma**2))   # [12, 1920, 1]

        # For score of map-16 to upsample and downsample
        verify_score_16 = verify_score.reshape(bs, spatial_shapes[1][0], spatial_shapes[1][1], 1).squeeze(-1)
        verify_score_8 = self.upsample(verify_score_16.unsqueeze(1)).squeeze(1)
        verify_score_32 = self.downsample(verify_score_16)
        verify_score_64 = self.downsample(verify_score_32)
        verify_score_list = [verify_score_8.flatten(1), verify_score_16.flatten(1),verify_score_32.flatten(1), verify_score_64.flatten(1)]
        verify_score = torch.cat(verify_score_list, dim=1).unsqueeze(-1)

        q = k = img_feat_src + imgfeat_adapt   # second image feature

        # concat multi-scale image feature
        src = torch.cat([img_feat_src_list[0],q ,img_feat_src_list[2],img_feat_src_list[3]], 1)
        # the reference points of img is non-learnable meshgrid
        reference_points_input = self.get_reference_points(spatial_shapes, src_valid_ratios, src.device)
        
        text_cond_img_ctx, _ = self.img2img_msdeform_attn(
            hidden_states=src,
            attention_mask=orig_multiscale_masks,
            encoder_hidden_states=orig_multiscale_img_feat,
            position_embeddings=orig_multiscale_img_pos_embeds,
            reference_points=reference_points_input, 
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index, 
        )

        # adapted image feature
        adapt_img_feat = (self.norm_img(orig_multiscale_img_feat) + self.norm_text_cond_img(text_cond_img_ctx)) * verify_score

        # text-guided depth encoder
        depthfeat_adapt, _ = self.depth2textcross_attn(
            query=depth_pos_embed,
            key=self.with_pos_embed(word_feat, word_pos),
            value=word_feat, 
            key_padding_mask=word_key_padding_mask,
        )

        q = k = depth_pos_embed + depthfeat_adapt   # depth feature of second image
        text_cond_depth, _ = self.depth2depth_attn(
            query=q, 
            key=k, 
            value=depth_pos_embed, 
            key_padding_mask=mask_depth,
        )

        # adapted depth feature
        adapt_depth_feat = (self.norm_depth(depth_pos_embed) + self.norm_text_cond_depth(text_cond_depth)) * verify_score_16.flatten(1).unsqueeze(-1)
        
        return torch.cat([orig_multiscale_img_feat, adapt_img_feat], dim=-1), torch.cat([depth_pos_embed, adapt_depth_feat], dim=-1)


class Mono3DVGv2DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.d_model
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout

        # text cross attention
        self.cross_attn_text = nn.MultiheadAttention(self.embed_dim, config.decoder_attention_heads, dropout=config.dropout, batch_first=True)
        self.cross_attn_text_layer_norm = nn.LayerNorm(self.embed_dim)
        self.cross_attn_text_residual = config.decoder_text_residual

        # depth cross attention
        self.cross_attn_depth = nn.MultiheadAttention(self.embed_dim, config.decoder_attention_heads, dropout=config.dropout, batch_first=True)
        self.cross_attn_depth_layer_norm = nn.LayerNorm(self.embed_dim)
        self.cross_attn_depth_residual = config.decoder_depth_residual
        
        # self attention
        if config.decoder_self_attn:
            self.self_attn = nn.MultiheadAttention(self.embed_dim, config.decoder_attention_heads, dropout=config.dropout, batch_first=True)
            self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        else:
            self.self_attn = None

        # cross attention
        self.encoder_attn = MSDeformAttn(
            config,
            num_heads=config.decoder_attention_heads,
            n_points=config.decoder_n_points,
        )
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        # feedforward neural networks
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def with_pos_embed(self, tensor: torch.Tensor, position_embeddings: Optional[Tensor]):
        return tensor if position_embeddings is None else tensor + position_embeddings

    def forward(
        self,
        hidden_states: torch.Tensor, # bs, nq, d_model
        position_embeddings: Optional[torch.Tensor] = None, # bs, nq, d_model
        reference_points: Optional[Tensor] = None, # bs, nq, 2
        spatial_shapes: Optional[Tensor] = None, # num_levels, 2
        level_start_index: Optional[Tensor] = None, # num_levels
        encoder_hidden_states: Optional[torch.Tensor] = None, # bs, \sum{hxw}, d_model
        encoder_attention_mask: Optional[torch.Tensor] = None, # bs, \sum{hxw}
        # For text
        text_embeds: Optional[Tensor] = None, # bs, seq_len, d_model
        text_attention_mask: Optional[Tensor] = None, # bs, seq_len
        # For depth
        depth_embeds: Optional[Tensor] = None, # bs, H_1*W_1, d_model
        depth_attention_mask: Optional[Tensor] = None, # bs, H_1*W_1
        depth_adapt_k: Optional[Tensor] = None, # H_1*W_1, bs, d_model
        output_attentions: Optional[bool] = False,
    ):
        """
        Args:
            hidden_states (`torch.FloatTensor`):
                Input to the layer of shape `(seq_len, batch, embed_dim)`.
            position_embeddings (`torch.FloatTensor`, *optional*):
                Position embeddings that are added to the queries and keys in the self-attention layer.
            reference_points (`torch.FloatTensor`, *optional*):
                Reference points.
            spatial_shapes (`torch.LongTensor`, *optional*):
                Spatial shapes.
            level_start_index (`torch.LongTensor`, *optional*):
                Level start index.
            encoder_hidden_states (`torch.FloatTensor`):
                cross attention input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_attention_mask (`torch.FloatTensor`): encoder attention mask of size
                `(batch, 1, target_len, source_len)` where padding elements are indicated by very large negative values.
            text_embeds (`torch.FloatTensor`, *optional*):
                Text embeddings of shape `(batch, seq_len, embed_dim)`.
            text_attention_mask (`torch.FloatTensor`, *optional*):
                Text attention mask of shape `(batch, seq_len)`.
            depth_embeds (`torch.FloatTensor`, *optional*):
                Depth embeddings of shape `(batch, H_1*W_1, embed_dim)`.
            depth_attention_mask (`torch.FloatTensor`, *optional*):
                Depth attention mask of shape `(batch, H_1*W_1)`.
            depth_adapt_k (`torch.FloatTensor`, *optional*):
                Depth adapt key of shape `(H_1*W_1, batch, embed_dim)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers.
        """
        residual = hidden_states
        
        # Depth Cross Attention
        hidden_states, cross_attn_depth_weights = self.cross_attn_depth(
            query=hidden_states,
            key=depth_adapt_k,
            value=depth_embeds,
            key_padding_mask=depth_attention_mask,
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        if self.cross_attn_depth_residual:
            hidden_states = hidden_states + residual
        hidden_states = self.cross_attn_depth_layer_norm(hidden_states)
        if self.cross_attn_depth_residual:
            residual = hidden_states
        
        # Text Cross Attention
        hidden_states, cross_attn_text_weights = self.cross_attn_text(
            query=self.with_pos_embed(hidden_states, position_embeddings),
            key=text_embeds,
            value=text_embeds,
            key_padding_mask=text_attention_mask,
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        if self.cross_attn_text_residual:
            hidden_states = hidden_states + residual
        hidden_states = self.cross_attn_text_layer_norm(hidden_states)
        if self.cross_attn_text_residual:
            residual = hidden_states
        
        # Self-Attention
        self_attn_weights = None
        if self.self_attn is not None:
            hidden_states, self_attn_weights = self.self_attn(
                query=self.with_pos_embed(hidden_states, position_embeddings),
                key=self.with_pos_embed(hidden_states, position_embeddings),
                value=hidden_states,
            )
            hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
            hidden_states = hidden_states + residual
            hidden_states = self.self_attn_layer_norm(hidden_states)
            residual = hidden_states
        
        # Cross-Atrtention
        cross_attn_weights = None
        hidden_states, cross_attn_weights = self.encoder_attn(
            hidden_states=hidden_states,
            attention_mask=encoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            position_embeddings=position_embeddings,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            output_attentions=output_attentions,
        )
        
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = hidden_states + residual
        hidden_states = self.encoder_attn_layer_norm(hidden_states)
        
        # Fully Connected
        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)
        
        if output_attentions:
            outputs += (cross_attn_depth_weights, cross_attn_text_weights, self_attn_weights, cross_attn_weights)
        
        return outputs
        

class Mono3DVGv2PreTrainedModel(PreTrainedModel):
    config_class = Mono3DVGv2Config
    base_model_prefix = "model"
    main_input_name = "pixel_values"
    
    def _init_weights(self, module):
        std = self.config.init_std
        
        if isinstance(module, LearnedPositionEmbedding):
            nn.init.uniform_(module.row_embeddings.weight)
            nn.init.uniform_(module.column_embeddings.weight)
        elif isinstance(module, MSDeformAttn):
            module._reset_parameters()
        elif isinstance(module, (nn.Linear, nn.Conv2d, nn.BatchNorm2d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        if hasattr(module, "reference_points") and not self.config.two_stage:
            nn.init.xavier_uniform_(module.reference_points.weight.data, gain=1.0)
            nn.init.constant_(module.reference_points.bias.data, 0.0)
        if hasattr(module, "level_embed"):
            nn.init.normal_(module.level_embed)
    
        
class Mono3DVGv2Decoder(Mono3DVGv2PreTrainedModel):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a [`Mono3DVGv2DecoderLayer`].
    
    The decoder updates the query embeddings through multiple depth-aware cross-attention, text-aware cross-attention, self-attention, and cross-attention layers.
    
    Args:
        config: Mono3DVGv2Config
    """
    
    def __init__(self, config: Mono3DVGv2Config):
        super().__init__(config)
        
        self.dropout = config.dropout
        self.layers = nn.ModuleList([Mono3DVGv2DecoderLayer(config) for _ in range(config.decoder_layers)])
        self.gradient_checkpointing = False
        
        # hack implementation for iterative bounding box refinement and two-stage Deformable DETR
        self.bbox_embed = None
        self.dim_embed = None
        self.class_embed = None
        
        # Use DAB
        self.use_dab = config.use_dab
        if self.use_dab:
            self.d_model = config.d_model
            self.query_scale = MLP(config.d_model, config.d_model, config.d_model, 2)
            self.ref_point_head = MLP(3 * config.d_model, config.d_model, config.d_model, 2)
        
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        position_embeddings=None, # bs, nq, d_model
        reference_points=None, # bs, nq, 2
        spatial_shapes=None, # num_levels, 2
        level_start_index=None, # num_levels
        valid_ratios=None, 
        # For text
        text_embeds=None, # bs, seq_len, d_model
        text_attention_mask=None, # bs, seq_len
        # For depth
        depth_embeds=None, # bs, H_1*W_1, d_model
        depth_attention_mask=None, # bs, H_1*W_1
        depth_adapt_k=None, # H_1*W_1, bs, d_model
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        Args:
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, num_queries, hidden_size)`):
                The query embeddings that are passed into the decoder.
            encoder_hidden_states (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
                of the decoder.
            encoder_attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing cross-attention on padding pixel_values of the encoder. Mask values selected
                in `[0, 1]`:
                - 0 for pixels that are real (i.e. **not masked**),
                - 1 for pixels that are padding (i.e. **masked**).
            position_embeddings (`torch.FloatTensor` of shape `(batch_size, num_queries, hidden_size)`, *optional*):
                Position embeddings that are added to the queries and keys in each self-attention layer.
            reference_points (`torch.FloatTensor` of shape `(batch_size, num_queries, 4)` is `as_two_stage` else `(batch_size, num_queries, 2)` or , *optional*):
                Reference point in range `[0, 1]`, top-left (0,0), bottom-right (1, 1), including padding area.
            spatial_shapes (`torch.FloatTensor` of shape `(num_feature_levels, 2)`):
                Spatial shapes of the feature maps.
            level_start_index (`torch.LongTensor` of shape `(num_feature_levels)`, *optional*):
                Indexes for the start of each feature level. In range `[0, sequence_length]`.
            valid_ratios (`torch.FloatTensor` of shape `(batch_size, num_feature_levels, 2)`, *optional*):
                Ratio of valid area in each feature level.
            text_embeds (`torch.FloatTensor` of shape `(batch_size, seq_len, hidden_size)`, *optional*):
                Text embeddings of the input text.
            text_attention_mask (`torch.FloatTensor` of shape `(batch_size, seq_len)`, *optional*):
                Text attention mask of the input text.
            depth_embeds (`torch.FloatTensor` of shape `(batch_size, H_1*W_1, hidden_size)`, *optional*):
                Depth embeddings of the input depth map.
            depth_attention_mask (`torch.FloatTensor` of shape `(batch_size, H_1*W_1)`, *optional*):
                Depth attention mask of the input depth map.
            depth_adapt_k (`torch.FloatTensor` of shape `(H_1*W_1, batch_size, hidden_size)`, *optional*):
                Depth adapt key of the input depth map.

            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~file_utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        
        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None
        intermediate = ()
        intermediate_reference_points = ()
        intermediate_reference_dims = ()
        
        for idx, decoder_layer in enumerate(self.layers):
            if reference_points.shape[-1] == 6: # x, y, l, t, r, b
                reference_points_input = (
                    reference_points[:, :, None] * torch.stack([valid_ratios[..., 0], valid_ratios[..., 1], valid_ratios[..., 0], valid_ratios[..., 0], valid_ratios[..., 1], valid_ratios[..., 1]], -1)[:, None]
                )# bs, nq, 4, 6
            else:
                if reference_points.shape[-1] != 2:
                    raise ValueError("Reference points' last dimension must be of size 2")
                reference_points_input = reference_points[:, :, None] * valid_ratios[:, None] # bs, nq, 4, 2
            
            # DAB
            if self.use_dab:
                query_sine_embed = gen_sineembed_for_position(reference_points_input[:, :, 0, :]) # bs, nq, 256*3 
                raw_query_pos = self.ref_point_head(query_sine_embed) # bs, nq, 256
                pos_scale = self.query_scale(hidden_states) if idx != 0 else 1
                position_embeddings = pos_scale * raw_query_pos

            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    None,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states=hidden_states,
                    position_embeddings=position_embeddings,
                    reference_points=reference_points_input,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    text_embeds=text_embeds,
                    text_attention_mask=text_attention_mask,
                    depth_embeds=depth_embeds,
                    depth_attention_mask=depth_attention_mask,
                    depth_adapt_k=depth_adapt_k,
                    output_attentions=output_attentions,
                )
                
            hidden_states = layer_outputs[0] # bs, nq, d_model

            # hack implementation for iterative bounding box refinement
            if self.bbox_embed is not None:
                tmp = self.bbox_embed[idx](hidden_states)
                if reference_points.shape[-1] == 6:
                    new_reference_points = tmp + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                else:
                    if reference_points.shape[-1] != 2:
                        raise ValueError(
                            f"Reference points' last dimension must be of size 2, but is {reference_points.shape[-1]}"
                        )
                    new_reference_points = tmp
                    new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            if self.dim_embed is not None:
                reference_dims = self.dim_embed[idx](hidden_states)

            intermediate += (hidden_states,)
            intermediate_reference_points += (reference_points,)
            intermediate_reference_dims += (reference_dims,)
            
            if output_attentions:
                all_self_attns += (layer_outputs[3],)
                
                if encoder_hidden_states is not None:
                    all_cross_attentions += ((layer_outputs[1], layer_outputs[2], layer_outputs[4],),)
        
        # Keep batch_size as first dimension
        intermediate = torch.stack(intermediate, dim=1)
        intermediate_reference_points = torch.stack(intermediate_reference_points, dim=1)
        intermediate_reference_dims = torch.stack(intermediate_reference_dims, dim=1)
        
        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    intermediate,
                    intermediate_reference_points,
                    intermediate_reference_dims,
                    all_hidden_states,
                    all_self_attns,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return Mono3DVGv2DecoderOutput(
            last_hidden_state=hidden_states,
            intermediate_hidden_states=intermediate,
            intermediate_reference_points=intermediate_reference_points,
            intermediate_reference_dims=intermediate_reference_dims,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        )


class Mono3DVGv2Model(Mono3DVGv2PreTrainedModel):
    def __init__(self, config: Mono3DVGv2Config):
        super().__init__(config)
        self.d_model = config.d_model
        
        # Create vision backbone + positional encoding
        vision_backbone = build_vision_backbone(config)
        position_embeddings = build_position_encoding(config)
        self.vision_backbone = VisionBackboneModel(vision_backbone, position_embeddings)
        
        # Create language backbone
        language_backbone = build_language_backbone(config)
        tokenizer = RobertaTokenizerFast.from_pretrained(config.text_encoder_type)
        self.language_backbone = LanguageBackboneModel(language_backbone, tokenizer)
        
        # Create depth predictor
        self.depth_predictor = DepthPredictor(config)
        
        # Use TextGuidedAdapter
        self.use_text_guided_adapter = config.use_text_guided_adapter
        if self.use_text_guided_adapter:
            self.text_guided_adapter = TextGuidedAdapter(config)
        
        # Create vision input projection layers
        if config.num_feature_levels > 1:
            num_backbone_outs = len(vision_backbone.intermediate_channel_sizes)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = vision_backbone.intermediate_channel_sizes[_]
                input_proj_list.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, self.d_model, kernel_size=1),
                        nn.GroupNorm(32, self.d_model),
                    )
                )
            for _ in range(config.num_feature_levels - num_backbone_outs):
                input_proj_list.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, self.d_model, kernel_size=3, stride=2, padding=1),
                        nn.GroupNorm(32, self.d_model),
                    )
                )
                in_channels = self.d_model
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv2d(vision_backbone.intermediate_channel_sizes[-1], self.d_model, kernel_size=1),
                        nn.GroupNorm(32, self.d_model),
                    )
                ]
            )

        # Create language input projection layer
        language_proj = nn.Sequential(
            nn.Linear(language_backbone.hidden_size, self.d_model, bias=True),
            nn.LayerNorm(self.d_model, eps=1e-12),
            nn.Dropout(0.1),
        )
        self.language_proj = _get_clones(language_proj, config.num_text_output_layers)
        # self.language_proj = nn.ModuleList([language_proj for _ in range(config.num_text_output_layers)])
        
        self.encoder = build_vision_language_encoder(config)
        self.decoder = Mono3DVGv2Decoder(config)
        
        self.level_embed = nn.Parameter(torch.Tensor(config.num_feature_levels, self.d_model))
        
        # setting query
        self.use_dab = config.use_dab
        if self.use_dab:
            self.target_embeddings = nn.Embedding(config.num_queries, self.d_model)
            self.refpoint_embeddings = nn.Embedding(config.num_queries, 6) # cx, cy, l, r, t, b
        else:
            self.query_position_embeddings = nn.Embedding(config.num_queries, self.d_model * 2)
            self.reference_points = nn.Linear(self.d_model, 2) # cx, cy
        
        self.post_init()

    def get_valid_ratio(self, mask):
        """Get the valid ratio of all feature maps."""

        _, height, width = mask.shape
        valid_height = torch.sum(~mask[:, :, 0], 1)
        valid_width = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_heigth = valid_height.float() / height
        valid_ratio_width = valid_width.float() / width
        valid_ratio = torch.stack([valid_ratio_width, valid_ratio_heigth], -1)
        return valid_ratio

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        pixel_mask: Optional[torch.LongTensor] = None,
        text: Optional[List[str]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], Mono3DVGv2ForSingleObjectDetectionOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        batch_size, num_channels, height, width = pixel_values.shape
        device = pixel_values.device
        
        if pixel_mask is None:
            pixel_mask = torch.zeros(((batch_size, height, width)), dtype=torch.bool, device=device)
        
        # Extract multi-scale feature maps of same resolution `config.d_model` (cf Figure 4 in paper)
        # First, sent pixel_values + pixel_mask through Backbone to obtain the features which is a list of tuples
        features, position_embeddings_list = self.vision_backbone(pixel_values, pixel_mask)
        
        # Then, apply 1x1 convolution to reduce the channel dimension to d_model (256 by default)
        sources = []
        masks = []
        for level, (source, mask) in enumerate(features):
            sources.append(self.input_proj[level](source))
            masks.append(mask)
            if mask is None:
                raise ValueError("No attention mask was provided")
        
        # Lowest resolution feature maps are obtained via 3x3 stride 2 convolutions on the final stage
        if self.config.num_feature_levels > len(sources):
            _len_sources = len(sources)
            for level in range(_len_sources, self.config.num_feature_levels):
                if level == _len_sources:
                    source = self.input_proj[level](features[-1][0])
                else:
                    source = self.input_proj[level](sources[-1])
                mask = nn.functional.interpolate(pixel_mask[None].float(), size=source.shape[-2:]).to(torch.bool)[0]
                pos_l = self.vision_backbone.position_embedding(source, mask).to(source.dtype)
                sources.append(source)
                masks.append(mask)
                position_embeddings_list.append(pos_l)
        
        # Extract text features
        # First, sent text through the language backbone to obtain the features with is a list of tensors
        text_features, text_attention_mask = self.language_backbone(text, device)
        
        # Then, apply linear projection to reduce the channel dimension to d_model (256 by default)
        text_embeds = []
        for layer, text_feature in enumerate(text_features):
            text_embeds.append(self.language_proj[layer](text_feature))
        
        # Create queries
        if self.use_dab:
            target_embeds = self.target_embeddings.weight           # nq, 256
            refpoint_embeds = self.refpoint_embeddings.weight      # nq, 6
            query_embeds = torch.cat((target_embeds, refpoint_embeds), dim=1) # nq, 262
        else:
            query_embeds = self.query_position_embeddings.weight # (num_queries, d_model * 2)
        
        # Prepare encoder inputs (by flattening)
        source_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for level, (source, mask, pos_embed) in enumerate(zip(sources, masks, position_embeddings_list)):
            batch_size, num_channels, height, width = source.shape
            spatial_shape = (height, width)
            spatial_shapes.append(spatial_shape)
            source = source.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[level].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            source_flatten.append(source)
            mask_flatten.append(mask)
        source_flatten = torch.cat(source_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=source_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)
        valid_ratios = valid_ratios.float()
        
        # Fourth, sent source_flatten + mask_flatten + lvl_pos_embed_flatten (backbone + proj layer output) + text_embeds +text_attention_mask through encoder
        # Also provide spatial_shapes, level_start_index and valid_ratios
        encoder_outputs = self.encoder(
            vision_embeds=source_flatten,
            vision_attention_mask=mask_flatten,
            text_embeds=text_embeds if len(text_embeds) > 1 else text_embeds[-1],
            text_attention_mask=text_attention_mask,
            position_embeddings=lvl_pos_embed_flatten,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            output_attentions=output_attentions,
            return_dict=return_dict,
        )
        vision_embeds, text_embeds = encoder_outputs[0], encoder_outputs[1]
        
        # Fifth, prepare depth predictor inputs
        level_index_num = [H_ * W_ for H_, W_ in spatial_shapes]
        depth_logits, depth_embeds, weighted_depth = self.depth_predictor(
            [m.permute(0, 2, 1).reshape(-1, self.d_model, H_, W_) for m, (H_, W_) in zip(vision_embeds.split(level_index_num, dim=1), spatial_shapes)],
            mask_flatten.split(level_index_num, dim=1)[1].reshape(-1, *spatial_shapes[1]),
            lvl_pos_embed_flatten.split(level_index_num, dim=1)[1].permute(0, 2, 1).reshape(-1, self.d_model, *spatial_shapes[1]),
        )
        depth_embeds = depth_embeds.flatten(2).permute(0, 2, 1) # bs, H_1*W_1, d_model
        depth_attention_mask = masks[1].flatten(1) # bs, H_1*W_1
        depth_adapt_k = None
        
        # Sixth, prepare text-guided adapter 
        if self.use_text_guided_adapter:
            img_feat_orig2adapt, depth_feat_orig2adapt = self.text_guided_adapter(
                img_feat_src=vision_embeds, 
                masks=mask_flatten, 
                img_pos_embeds=lvl_pos_embed_flatten,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                src_valid_ratios=valid_ratios, 
                word_feat=text_embeds, 
                word_key_padding_mask=text_attention_mask, 
                depth_pos_embed=depth_embeds,
                mask_depth=depth_attention_mask, 
            )
            img_feat_srcs = img_feat_orig2adapt.chunk(2, dim=-1)
            vision_embeds = img_feat_srcs[1]
            depth_feat_srcs = depth_feat_orig2adapt.chunk(2, dim=-1)
            depth_adapt_k = depth_feat_srcs[1] # bs, H_1*W_1, d_model
        
        # Sixth, prepare decoder inputs
        batch_size, _, num_channels = vision_embeds.shape
        if self.use_dab:
            target = self.target_embeddings.weight.unsqueeze(0).expand(batch_size, -1, -1) # (bs, nq, d_model)
            reference_points = self.refpoint_embeddings.weight.unsqueeze(0).expand(batch_size, -1, -1).sigmoid() # (bs, nq, 6)
            init_reference_points = reference_points
        else:
            query_embed, target = torch.split(query_embeds, num_channels, dim=1)
            target = target.unsqueeze(0).expand(batch_size, -1, -1) # (bs, nq, d_model)
            reference_points = self.reference_points(query_embed).sigmoid() # (bs, nq, 2)
            init_reference_points = reference_points
        
        decoder_outputs = self.decoder(
            inputs_embeds=target,
            position_embeddings=None if self.use_dab else query_embed, # DAB will recalculate the position embedding
            encoder_hidden_states=vision_embeds,
            encoder_attention_mask=mask_flatten,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            text_embeds=text_embeds,
            text_attention_mask=text_attention_mask,
            depth_embeds=depth_embeds,
            depth_attention_mask=depth_attention_mask,
            depth_adapt_k=depth_embeds if depth_adapt_k is None else depth_adapt_k,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        if not return_dict:
            depth_outputs = tuple(value for value in [depth_logits, weighted_depth] if value is not None)
            tuple_outputs = (init_reference_points,) + decoder_outputs + encoder_outputs + depth_outputs
            
            return tuple_outputs
        
        return Mono3DVGv2ModelOutput(
            init_reference_points=init_reference_points,
            last_hidden_state=decoder_outputs.last_hidden_state,
            intermediate_hidden_states=decoder_outputs.intermediate_hidden_states,
            intermediate_reference_points=decoder_outputs.intermediate_reference_points,
            intermediate_reference_dims=decoder_outputs.intermediate_reference_dims,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_vision_hidden_state,
            encoder_attentions=encoder_outputs.attentions,
            depth_logits=depth_logits,
            weighted_depth=weighted_depth,
        )


class Mono3DVGv2ForSingleObjectDetection(Mono3DVGv2PreTrainedModel):
    
    def __init__(self, config: Mono3DVGv2Config):
        super().__init__(config)
        
        # Mono3DVG v2 model
        self.model = Mono3DVGv2Model(config)
        
        # Detection heads on top
        self.class_embed = nn.Linear(config.d_model, config.num_labels)
        self.bbox_embed = MLP(config.d_model, config.d_model, 6, 3)
        self.dim_embed_3d = MLP(config.d_model, config.d_model, 3, 2)
        self.angle_embed = MLP(config.d_model, config.d_model, 24, 2)
        self.depth_embed = MLP(config.d_model, config.d_model, 2, 2)  # depth and deviation
        
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(config.num_labels) * bias_value
        if config.init_box:
            nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        
        # if two-stage, the last class_embed and bbox_embed is for region proposal generation
        num_pred = config.decoder_layers
        if config.with_box_refine:
            self.class_embed = _get_clones(self.class_embed, num_pred)
            self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            # hack implementation for iterative bounding box refinement
            self.model.decoder.bbox_embed = self.bbox_embed
            self.dim_embed_3d = _get_clones(self.dim_embed_3d, num_pred)
            self.model.decoder.dim_embed = self.dim_embed_3d
            self.angle_embed = _get_clones(self.angle_embed, num_pred)
            self.depth_embed = _get_clones(self.depth_embed, num_pred)
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
            self.dim_embed_3d = nn.ModuleList([self.dim_embed_3d for _ in range(num_pred)])
            self.angle_embed = nn.ModuleList([self.angle_embed for _ in range(num_pred)])
            self.depth_embed = nn.ModuleList([self.depth_embed for _ in range(num_pred)])
            self.model.decoder.bbox_embed = None

        # Initialize weights and apply final processing
        self.post_init()
    
    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, outputs_3d_dim, outputs_angle, outputs_depth):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b, 
                 'pred_3d_dim': c, 'pred_angle': d, 'pred_depth': e}
                for a, b, c, d, e in zip(outputs_class[:-1], outputs_coord[:-1],
                                         outputs_3d_dim[:-1], outputs_angle[:-1], outputs_depth[:-1])]    
        
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        pixel_mask: Optional[torch.LongTensor] = None,
        calibs: Optional[torch.FloatTensor] = None,
        img_sizes: Optional[torch.FloatTensor] = None,
        captions: Optional[List[str]] = None,
        labels: Optional[List[dict]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], Mono3DVGv2ForSingleObjectDetectionOutput]:
        r"""
        labels (`List[Dict]` of len `(batch_size,)`, *optional*):
            Labels for computing the bipartite matching loss. List of dicts, each dictionary containing at least the
            following 6 keys: 'class_labels', 'boxes_3d', 'size_3d', 'depth', 'heading_bin' and 'heading_res' (the class labels, 3D bounding boxes, 3D sizes, depths and 3D angles of an 3D object in the batch respectively).
            The class labels themselves should be a `torch.LongTensor` of len `(number of bounding boxes in the image,)`, 
            The 3D bounding boxes a `torch.FloatTensor` of shape `(number of bounding boxes in the image, 6)`, 
            The 3D sizes a `torch.FloatTensor` of shape `(number of bounding boxes in the image, 3)`,
            The depths a `torch.FloatTensor` of shape `(number of bounding boxes in the image,)`,
            The 3D angles with bin a `torch.LongTensor` of shape `(number of bounding boxes in the image,)` and res a `torch.FloatTensor` of shape `(number of bounding boxes in the image,)`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # First, sent image-text pairs through DETR base model to obtain encoder + decoder outputs
        outputs = self.model(
            pixel_values,
            pixel_mask=pixel_mask,
            text=captions,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        hidden_states = outputs.intermediate_hidden_states if return_dict else outputs[2]
        init_reference = outputs.init_reference_points if return_dict else outputs[0]
        inter_references = outputs.intermediate_reference_points if return_dict else outputs[3]
        inter_references_dim = outputs.intermediate_reference_dims if return_dict else outputs[4]
        pred_depth_map_logits = outputs.depth_logits if return_dict else outputs[-2]
        weighted_depth = outputs.weighted_depth if return_dict else outputs[-1]
        
        # class logits + predicted bounding boxes + 3D dims + depths + angles
        outputs_classes = []
        outputs_coords = []
        outputs_3d_dims = []
        outputs_depths = []
        outputs_angles = []
        
        for level in range(hidden_states.shape[1]):
            if level == 0:
                reference = init_reference
            else:
                reference = inter_references[:, level - 1]
            reference = inverse_sigmoid(reference)
            delta_bbox = self.bbox_embed[level](hidden_states[:, level])
            if reference.shape[-1] == 6:
                outputs_coord_logits = delta_bbox + reference
            elif reference.shape[-1] == 2:
                delta_bbox[..., :2] += reference
                outputs_coord_logits = delta_bbox
            else:
                raise ValueError(f"reference.shape[-1] should be 6 or 2, but got {reference.shape[-1]}")

            # 3d center + 2d box
            outputs_coord = outputs_coord_logits.sigmoid()
            outputs_coords.append(outputs_coord)

            # classes
            outputs_class = self.class_embed[level](hidden_states[:, level])
            outputs_classes.append(outputs_class)

            # 3D sizes
            size3d = inter_references_dim[:, level]
            outputs_3d_dims.append(size3d)

            # depth_geo
            box2d_height_norm = outputs_coord[:, :, 4] + outputs_coord[:, :, 5]
            box2d_height = torch.clamp(box2d_height_norm * img_sizes[:, 1: 2], min=1.0)
            depth_geo = size3d[:, :, 0] / box2d_height * calibs[:, 0, 0].unsqueeze(1)

            # depth_reg
            depth_reg = self.depth_embed[level](hidden_states[:, level])

            # depth_map
            outputs_center3d = ((outputs_coord[..., :2] - 0.5) * 2).unsqueeze(2).detach()
            depth_map = F.grid_sample(
                weighted_depth.unsqueeze(1),
                outputs_center3d,
                mode='bilinear',
                align_corners=True
            ).squeeze(1)

            # depth average + sigma
            depth_ave = torch.cat([((1. / (depth_reg[:, :, 0: 1].sigmoid() + 1e-6) - 1.) + depth_geo.unsqueeze(-1) + depth_map) / 3,
                                    depth_reg[:, :, 1: 2]], -1)
            outputs_depths.append(depth_ave)

            # angles
            outputs_angle = self.angle_embed[level](hidden_states[:, level])
            outputs_angles.append(outputs_angle)

        outputs_coord = torch.stack(outputs_coords)
        outputs_class = torch.stack(outputs_classes)
        outputs_3d_dim = torch.stack(outputs_3d_dims)
        outputs_depth = torch.stack(outputs_depths)
        outputs_angle = torch.stack(outputs_angles)
        
        logits = outputs_class[-1]
        pred_boxes = outputs_coord[-1]
        pred_3d_dim = outputs_3d_dim[-1]
        pred_depth = outputs_depth[-1]
        pred_angle = outputs_angle[-1]

        loss, loss_dict, auxiliary_outputs = None, None, None
        if labels is not None:
            # First: create the matcher
            matcher = Mono3DVGv2HungarianMatcher(
                class_cost=self.config.class_cost, center3d_cost=self.config.center3d_cost, bbox_cost=self.config.bbox_cost, giou_cost=self.config.giou_cost
            )
            # Second: create the criterion
            losses = ['labels', 'boxes', 'cardinality', 'depths', 'dims', 'angles', 'center', 'depth_map']
            criterion = Mono3DVGv2Loss(
                matcher=matcher,
                num_classes=self.config.num_labels,
                focal_alpha=self.config.focal_alpha,
                losses=losses,
            )
            criterion.to(self.device)
            # Third: compute the losses, based on outputs and labels
            outputs_loss = {}
            outputs_loss["logits"] = logits
            outputs_loss["pred_boxes"] = pred_boxes
            outputs_loss['pred_3d_dim'] = pred_3d_dim
            outputs_loss['pred_depth'] = pred_depth
            outputs_loss['pred_angle'] = pred_angle
            outputs_loss['pred_depth_map_logits'] = pred_depth_map_logits
            if self.config.auxiliary_loss:
                auxiliary_outputs = self._set_aux_loss(outputs_class, outputs_coord, outputs_3d_dim, outputs_angle, outputs_depth)
                outputs_loss["auxiliary_outputs"] = auxiliary_outputs
                
            loss_dict = criterion(outputs_loss, labels)
            # Fourth: compute total loss, as a weighted sum of the various losses
            weight_dict = {"loss_ce": self.config.cls_loss_coefficient, "loss_bbox": self.config.bbox_loss_coefficient}
            weight_dict["loss_giou"] = self.config.giou_loss_coefficient
            weight_dict['loss_dim'] = self.config.dim_loss_coefficient
            weight_dict['loss_angle'] = self.config.angle_loss_coefficient
            weight_dict['loss_depth'] = self.config.depth_loss_coefficient
            weight_dict['loss_center'] = self.config.center3d_loss_coefficient
            weight_dict['loss_depth_map'] = self.config.depth_map_loss_coefficient
            if self.config.auxiliary_loss:
                aux_weight_dict = {}
                for i in range(self.config.decoder_layers - 1):
                    aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
                weight_dict.update(aux_weight_dict)
            loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        
        if not return_dict:
            if auxiliary_outputs is not None:
                output = (logits, pred_boxes, pred_3d_dim, pred_depth, pred_angle, pred_depth_map_logits) + auxiliary_outputs + outputs
            else:
                output = (logits, pred_boxes, pred_3d_dim, pred_depth, pred_angle, pred_depth_map_logits) + outputs
            tuple_outputs = ((loss, loss_dict) + output) if loss is not None else output

            return tuple_outputs

        dict_outputs = Mono3DVGv2ForSingleObjectDetectionOutput(
            loss=loss,
            loss_dict=loss_dict,
            logits=logits,
            pred_boxes=pred_boxes,
            pred_3d_dim=pred_3d_dim,
            pred_depth=pred_depth,
            pred_angle=pred_angle,
            pred_depth_map_logits=pred_depth_map_logits,
            auxiliary_outputs=auxiliary_outputs,
            last_hidden_state=outputs.last_hidden_state,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_attentions=outputs.encoder_attentions,
            intermediate_hidden_states=outputs.intermediate_hidden_states,
            intermediate_reference_points=outputs.intermediate_reference_points,
            init_reference_points=outputs.init_reference_points,
        )
        
        return dict_outputs

    @classmethod
    def _load_mono3dvg_pretrain_model(
        cls,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
        config: Optional[Union[Mono3DVGv2Config, dict]] = None,
        output_loading_info: bool = False,
        logger: Optional[logging.Logger] = None,
    ):
        logger = logger if logger is not None else logging.getLogger(__name__)
        
        # Load config if we don't provide a configuration
        if config is None:
            config = Mono3DVGv2Config()
        elif isinstance(config, dict):
            config = Mono3DVGv2Config(**config)
        else:
            assert isinstance(config, Mono3DVGv2Config), f"config: {config} has to be of type Mono3DVGv2Config"
        
        config = copy.deepcopy(config) # We do not want to modify the config inplace in from_pretrained.
        model = cls(config)
        
        logger.info(f"==> Loading from Mono3DVG '{pretrained_model_name_or_path}'")
        checkpoint = torch.load(pretrained_model_name_or_path, map_location="cpu", weights_only=False)
        
        new_state_dict = {}
        state_dict = checkpoint['model_state']
        model_state_dict = model.state_dict()
        for key, value in state_dict.items():
            if 'backbone.0' in key:
                new_key = key.replace('backbone.0', 'model.vision_backbone.backbone')
            elif 'text_encoder' in key:
                new_key = key.replace('text_encoder', 'model.language_backbone.backbone.model')
            elif 'mono3dvg_transformer' in key:
                new_key = key.replace('mono3dvg_transformer', 'model')
                if 'TextGuidedAdapter' in key:
                    new_key = new_key.replace('TextGuidedAdapter', 'text_guided_adapter')
                if 'mono3dvg_transformer.encoder' in key:
                    new_key = new_key.replace('msdeform_attn', 'self_attn')
                    new_key = new_key.replace('norm1', 'self_attn_layer_norm')
                    new_key = new_key.replace('ca_text', 'cross_attn_text')
                    new_key = new_key.replace('ca_img', 'cross_attn_img')
                    new_key = new_key.replace('catext_norm', 'cross_attn_text_layer_norm')
                    new_key = new_key.replace('caimg_norm', 'cross_attn_img_layer_norm')
                    new_key = new_key.replace('linear1', 'fc1')
                    new_key = new_key.replace('linear2', 'fc2')
                    new_key = new_key.replace('norm2', 'final_layer_norm')
                if 'mono3dvg_transformer.decoder' in key:
                    new_key = new_key.replace('cross_attn', 'encoder_attn')
                    new_key = new_key.replace('norm1', 'encoder_attn_layer_norm')
                    new_key = new_key.replace('ca_text', 'cross_attn_text')
                    new_key = new_key.replace('catext_norm', 'cross_attn_text_layer_norm')
                    new_key = new_key.replace('ca_depth', 'cross_attn_depth')
                    new_key = new_key.replace('cadepth_norm', 'cross_attn_depth_layer_norm')
                    new_key = new_key.replace('linear1', 'fc1')
                    new_key = new_key.replace('linear2', 'fc2')
                    new_key = new_key.replace('norm3', 'final_layer_norm')
            elif 'input_proj' in key:
                new_key = key.replace('input_proj', 'model.input_proj')
            elif 'resizer' in key:
                new_key = key.replace('resizer', 'model.language_proj')
                # All language_proj layers share the same weights
                for layer in range(config.num_text_output_layers):
                    if 'fc' in key:
                        language_proj_key = new_key.replace('fc', f'{layer}.0')
                    if 'layer_norm' in key:
                        language_proj_key = new_key.replace('layer_norm', f'{layer}.1')
                    new_state_dict[language_proj_key] = value
            elif 'tgt_embed' in key:
                new_key = key.replace('tgt_embed', 'model.target_embeddings')
            elif 'refpoint_embed' in key:
                new_key = key.replace('refpoint_embed', 'model.refpoint_embeddings')
            else:
                new_key = key
            
            if new_key in model_state_dict and value.shape != model_state_dict[new_key].shape:
                logger.info(f"Skip loading parameter: {new_key}, "
                            f"required shape: {model_state_dict[new_key].shape}, "
                            f"loaded shape: {state_dict[key].shape}")
                continue
            new_state_dict[new_key] = value

        if checkpoint['model_state'] is not None:
            missing_keys, unexpected_keys = model.load_state_dict(new_state_dict,strict=False)
        logger.info("==> Done")
        
        if output_loading_info:
            return model, {"missing_keys": missing_keys, "unexpected_keys": unexpected_keys}
        
        return model
        

# Copied from transformers.models.detr.modeling_detr.sigmoid_focal_loss
def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.

    Args:
        inputs (`torch.FloatTensor` of arbitrary shape):
            The predictions for each example.
        targets (`torch.FloatTensor` with the same shape as `inputs`)
            A tensor storing the binary classification label for each element in the `inputs` (0 for the negative class
            and 1 for the positive class).
        alpha (`float`, *optional*, defaults to `0.25`):
            Optional weighting factor in the range (0,1) to balance positive vs. negative examples.
        gamma (`int`, *optional*, defaults to `2`):
            Exponent of the modulating factor (1 - p_t) to balance easy vs hard examples.

    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    # add modulating factor
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes


class Mono3DVGv2Loss(nn.Module):
    """ This class computes the loss for Mono3DVG-TR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
        
    Args:
        matcher (`Mono3DVGv2HungarianMatcher`):
            Module able to compute a matching between targets and proposals.
        num_classes (`int`):
            Number of object categories, omitting the special no-object category.
        focal_alpha (`float`):
            Alpha parameter in focal loss.
        losses (`List[str]`):
            List of all the losses to be applied. See `get_loss` for a list of all available losses.
    """
    
    def __init__(self, matcher, num_classes, focal_alpha, losses):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.losses = losses
        self.focal_alpha = focal_alpha
        self.ddn_loss = DDNLoss()  # for depth map

    # removed logging parameter, which was part of the original implementation
    def loss_labels(self, outputs, targets, indices, num_boxes):
        """
        Classification loss (Binary focal loss) targets dicts must contain the key "class_labels" containing a tensor
        of dim [nb_target_boxes]
        """
        if "logits" not in outputs:
            raise KeyError("No logits were found in the outputs")
        source_logits = outputs["logits"]

        idx = self._get_source_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            source_logits.shape[:2], self.num_classes, dtype=torch.int64, device=source_logits.device
        )
        target_classes[idx] = target_classes_o.squeeze().long()

        target_classes_onehot = torch.zeros(
            [source_logits.shape[0], source_logits.shape[1], source_logits.shape[2] + 1],
            dtype=source_logits.dtype,
            layout=source_logits.layout,
            device=source_logits.device,
        )
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:, :, :-1]
        loss_ce = (
            sigmoid_focal_loss(source_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2)
            * source_logits.shape[1]
        )
        losses = {"loss_ce": loss_ce}

        return losses

    @torch.no_grad()
    # Copied from transformers.models.detr.modeling_detr.DetrLoss.loss_cardinality
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """
        Compute the cardinality error, i.e. the absolute error in the number of predicted non-empty boxes.

        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients.
        """
        logits = outputs["logits"]
        device = logits.device
        target_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (logits.argmax(-1) != logits.shape[-1] - 1).sum(1)
        card_err = nn.functional.l1_loss(card_pred.float(), target_lengths.float())
        losses = {"cardinality_error": card_err}
        return losses

    def loss_3dcenter(self, outputs, targets, indices, num_boxes):
        """
        Compute the losses related to the 3D center coordinates, the L1 regression loss.
        
        Targets dicts must contain the key "boxes_3d" containing a tensor of dim [nb_target_boxes, 6]. The target boxes are expected in format (cx, cy, l, r, t, b), normalized by the image size.
        """
        
        if "pred_boxes" not in outputs:
            raise KeyError("No predicted boxes found in outputs")
        idx = self._get_source_permutation_idx(indices)
        source_center3d = outputs['pred_boxes'][..., :2][idx]
        target_center3d = torch.cat([t['boxes_3d'][..., :2][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_center3d = nn.functional.l1_loss(source_center3d, target_center3d, reduction='none')
        losses = {}
        losses['loss_center'] = loss_center3d.sum() / num_boxes
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """
        Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss.

        Targets dicts must contain the key "boxes_3d" containing a tensor of dim [nb_target_boxes, 6]. The target boxes are expected in format (cx, cy, l, r, t, b), normalized by the image size.
        """
        if "pred_boxes" not in outputs:
            raise KeyError("No predicted boxes found in outputs")
        idx = self._get_source_permutation_idx(indices)
        source_boxes3d = outputs['pred_boxes'][idx]
        target_boxes3d = torch.cat([t['boxes_3d'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = nn.functional.l1_loss(source_boxes3d[..., 2:], target_boxes3d[..., 2:], reduction="none")

        losses = {}
        losses["loss_bbox"] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(
            generalized_box_iou(box_cxcylrtb_to_xyxy(source_boxes3d), box_cxcylrtb_to_xyxy(target_boxes3d))
        )
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_depths(self, outputs, targets, indices, num_boxes):  
        """
        Compute the losses related to the depth, the Laplacian algebraic
uncertainty loss.

        Targets dicts must contain the key "depth" containing a tensor of dim [nb_target_boxes, 1].
        """
        if "pred_depth" not in outputs:
            raise KeyError("No predicted depth found in outputs")
        idx = self._get_source_permutation_idx(indices)
        source_depths = outputs['pred_depth'][idx]
        target_depths = torch.cat([t['depth'][i] for t, (_, i) in zip(targets, indices)], dim=0).squeeze()

        depth_input, depth_log_variance = source_depths[:, 0], source_depths[:, 1] 
        # depth_log_variance should be positive
        depth_log_variance = abs(depth_log_variance)
        # heteroscedastic aleatoric uncertainty
        depth_loss = 1.4142 * torch.exp(-depth_log_variance) * torch.abs(depth_input - target_depths) + depth_log_variance  
        losses = {}
        losses['loss_depth'] = depth_loss.sum() / num_boxes 
        return losses  
    
    def loss_dims(self, outputs, targets, indices, num_boxes):  
        """
        Compute the losses related to the 3D size, the 3D IoU oriented loss.
        
        Targets dicts must contain the key "size_3d" containing a tensor of dim [nb_target_boxes, 3]. The target sizes are expected in format (h, w, l).
        """
        
        if "pred_3d_dim" not in outputs:
            raise KeyError("No predicted 3d_dim found in outputs")
        idx = self._get_source_permutation_idx(indices)
        source_dims = outputs['pred_3d_dim'][idx]
        target_dims = torch.cat([t['size_3d'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        dimension = target_dims.clone().detach()
        dim_loss = torch.abs(source_dims - target_dims)
        dim_loss /= dimension
        with torch.no_grad():
            compensation_weight = F.l1_loss(source_dims, target_dims) / dim_loss.mean()
        dim_loss *= compensation_weight
        losses = {}
        losses['loss_dim'] = dim_loss.sum() / num_boxes
        return losses

    def loss_angles(self, outputs, targets, indices, num_boxes):  
        """
        Compute the losses related to the angle, the MultiBin loss.
        
        Targets dicts must contain the key "heading_bin" containing a tensor of dim [nb_target_boxes, 1] and "heading_res" containing a tensor of dim [nb_target_boxes, 1].
        """
        if "pred_angle" not in outputs:
            raise KeyError("No predicted angle found in outputs")
        idx = self._get_source_permutation_idx(indices)
        heading_input = outputs['pred_angle'][idx]
        target_heading_cls = torch.cat([t['heading_bin'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        target_heading_res = torch.cat([t['heading_res'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        heading_input = heading_input.view(-1, 24)
        heading_target_cls = target_heading_cls.view(-1).long()
        heading_target_res = target_heading_res.view(-1)

        # classification loss
        heading_input_cls = heading_input[:, 0:12]
        cls_loss = F.cross_entropy(heading_input_cls, heading_target_cls, reduction='none')

        # regression loss
        heading_input_res = heading_input[:, 12:24]
        cls_onehot = torch.zeros(heading_target_cls.shape[0], 12).cuda().scatter_(dim=1, index=heading_target_cls.view(-1, 1), value=1)
        heading_input_res = torch.sum(heading_input_res * cls_onehot, 1)
        reg_loss = F.l1_loss(heading_input_res, heading_target_res, reduction='none')
        
        angle_loss = cls_loss + reg_loss
        losses = {}
        losses['loss_angle'] = angle_loss.sum() / num_boxes 
        return losses

    def loss_depth_map(self, outputs, targets, indices, num_boxes):
        """
        Compute the losses related to the depth map, the DDN loss.
        
        Targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4] and "depth" containing a tensor of dim [nb_target_boxes, 1]. The target boxes are expected in format (cx, cy, w, h), normalized by the image size.
        """
        if "pred_depth_map_logits" not in outputs:
            raise KeyError("No predicted depth map found in outputs")
        depth_map_logits = outputs['pred_depth_map_logits']

        num_gt_per_img = [len(t['boxes']) for t in targets]
        gt_boxes2d = torch.cat([t['boxes'] for t in targets], dim=0) * torch.tensor([80, 24, 80, 24], device='cuda')
        gt_boxes2d = box_cxcywh_to_xyxy(gt_boxes2d)
        gt_center_depth = torch.cat([t['depth'] for t in targets], dim=0).squeeze(dim=1)

        losses = dict()
        losses["loss_depth_map"] = self.ddn_loss(
            depth_map_logits, gt_boxes2d, num_gt_per_img, gt_center_depth)
        return losses

    def _get_source_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(source, i) for i, (source, _) in enumerate(indices)])
        source_idx = torch.cat([source for (source, _) in indices])
        return batch_idx, source_idx

    def _get_target_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(target, i) for i, (_, target) in enumerate(indices)])
        target_idx = torch.cat([target for (_, target) in indices])
        return batch_idx, target_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'depths': self.loss_depths,
            'dims': self.loss_dims,
            'angles': self.loss_angles,
            'center': self.loss_3dcenter,
            'depth_map': self.loss_depth_map,
        }
        if loss not in loss_map:
            raise ValueError(f"Loss {loss} not supported")
        return loss_map[loss](outputs, targets, indices, num_boxes)

    def forward(self, outputs, targets):
        """ 
        This performs the loss computation.
        
        Args:
             outputs (`dict`, *optional*):
                Dictionary of tensors, see the output specification of the model for the format.
             targets (`List[dict]`, *optional*):
                List of dicts, such that `len(targets) == batch_size`. The expected keys in each dict depends on the
                losses applied, see each loss' doc.
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "auxiliary_outputs"}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        # (Niels): comment out function below, distributed training to be added
        # if is_dist_avail_and_initialized():
        #     torch.distributed.all_reduce(num_boxes)
        # (Niels) in original implementation, num_boxes is divided by get_world_size()
        num_boxes = torch.clamp(num_boxes, min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "auxiliary_outputs" in outputs:
            for i, auxiliary_outputs in enumerate(outputs["auxiliary_outputs"]):
                indices = self.matcher(auxiliary_outputs, targets)
                for loss in self.losses:
                    if loss == 'depth_map':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    l_dict = self.get_loss(loss, auxiliary_outputs, targets, indices, num_boxes)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses


# Copied from transformers.models.detr.modeling_detr.DetrMLPPredictionHead
class MLP(nn.Module):
    """
    Very simple multi-layer perceptron (MLP, also called FFN), used to predict the normalized center coordinates,
    height and width of a bounding box w.r.t. an image.

    Copied from https://github.com/facebookresearch/detr/blob/master/models/detr.py

    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = nn.functional.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class Mono3DVGv2HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    
    Args:
        class_cost: 
            The relative weight of the classification error in the matching cost.
        center3d_cost:
            The relative weight of the L1 error of the center3d coordinates in the matching cost.
        bbox_cost: 
            The relative weight of the L1 error of the bounding box coordinates in the matching cost.
        giou_cost: 
        The relative weight of the giou loss of the bounding box in the matching cost.
    """

    def __init__(self, class_cost: float = 1, center3d_cost: float = 1, bbox_cost: float = 1, giou_cost: float = 1):
        super().__init__()
        requires_backends(self, ["scipy"])
        
        self.class_cost = class_cost
        self.center3d_cost = center3d_cost
        self.bbox_cost = bbox_cost
        self.giou_cost = giou_cost
        if class_cost == 0 and bbox_cost == 0 and giou_cost == 0 and center3d_cost == 0:
            raise ValueError("All costs of the Matcher can't be 0")

    @torch.no_grad()
    def forward(self, outputs, targets):
        """
        Args:
            outputs (`dict`):
                A dictionary that contains at least these entries:
                * "logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                * "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates.
            targets (`List[dict]`):
                A list of targets (len(targets) = batch_size), where each target is a dict containing:
                * "class_labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of
                  ground-truth
                 objects in the target) containing the class labels
                * "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates.

        Returns:
            `List[Tuple]`: A list of size `batch_size`, containing tuples of (index_i, index_j) where:
            - index_i is the indices of the selected predictions (in order)
            - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds: len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        batch_size, num_queries = outputs["logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["logits"].flatten(0, 1).sigmoid()  # [batch_size * num_queries, num_classes]
        out_bbox3d = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 6]
        
        # Also concat the target labels and boxes
        target_ids = torch.cat([v["labels"] for v in targets]).long()
        tgt_bbox3d = torch.cat([v["boxes_3d"] for v in targets])

        # Compute the classification cost.
        alpha = 0.25
        gamma = 2.0
        neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
        pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
        class_cost = pos_cost_class[:, target_ids] - neg_cost_class[:, target_ids]

        # Compute the center3d L1 cost between boxes
        center3d_cost = torch.cdist(out_bbox3d[..., :2], tgt_bbox3d[..., :2], p=1)

        # Compute the L1 cost between boxes
        bbox_cost = torch.cdist(out_bbox3d[..., 2:], tgt_bbox3d[..., 2:], p=1)

        # Compute the giou cost betwen boxes
        giou_cost = -generalized_box_iou(box_cxcylrtb_to_xyxy(out_bbox3d), box_cxcylrtb_to_xyxy(tgt_bbox3d))

        # Final cost matrix
        cost_matrix = self.bbox_cost * bbox_cost + self.center3d_cost * center3d_cost + self.class_cost * class_cost + self.giou_cost * giou_cost
        cost_matrix = cost_matrix.view(batch_size, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(cost_matrix.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def gen_sineembed_for_position(pos_tensor):
    # n_query, bs, _ = pos_tensor.size()
    # sineembed_tensor = torch.zeros(n_query, bs, 256)
    scale = 2 * math.pi
    dim_t = torch.arange(128, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / 128)
    x_embed = pos_tensor[:, :, 0] * scale
    y_embed = pos_tensor[:, :, 1] * scale
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
    if pos_tensor.size(-1) == 2:
        pos = torch.cat((pos_y, pos_x), dim=2)
    elif pos_tensor.size(-1) == 4:
        w_embed = pos_tensor[:, :, 2] * scale
        pos_w = w_embed[:, :, None] / dim_t
        pos_w = torch.stack((pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3).flatten(2)

        h_embed = pos_tensor[:, :, 3] * scale
        pos_h = h_embed[:, :, None] / dim_t
        pos_h = torch.stack((pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()), dim=3).flatten(2)

        pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=2)
    elif pos_tensor.size(-1) == 6:
        for i in range(2, 6):         # Compute sine embeds for l, r, t, b
            embed = pos_tensor[:, :, i] * scale
            pos_embed = embed[:, :, None] / dim_t
            pos_embed = torch.stack((pos_embed[:, :, 0::2].sin(), pos_embed[:, :, 1::2].cos()), dim=3).flatten(2)
            if i == 2:  # Initialize pos for the case of size(-1)=6
                pos = pos_embed
            else:       # Concatenate embeds for l, r, t, b
                pos = torch.cat((pos, pos_embed), dim=2)
        pos = torch.cat((pos_y, pos_x, pos), dim=2)
    else:
        raise ValueError("Unknown pos_tensor shape(-1):{}".format(pos_tensor.size(-1)))
    return pos

