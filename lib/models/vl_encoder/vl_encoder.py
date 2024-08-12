from torch import nn

from ..configuration_mono3dvg_v2 import Mono3DVGv2Config
from .encoder_layer import VisionCrossEncoderLayer, get_reference_points, VisionLanguageModelOutput


class VisionLanguageEncoder(nn.Module):
    def __init__(self, config: Mono3DVGv2Config):
        super().__init__()
        self.layers = nn.ModuleList([VisionCrossEncoderLayer(config) for _ in range(config.encoder_layers)])
    
    def forward(
        self, 
        vision_embeds=None,
        vision_attention_mask=None,
        text_embeds=None,
        text_attention_mask=None,
        position_embeddings=None,
        spatial_shapes=None,
        level_start_index=None,
        valid_ratios=None,
        output_attentions=None,
        return_dict=None,
    ):
        """
        Args:
            vision_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Flattened feature map (output of the backbone + projection layer) that is passed to the encoder.
            vision_attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding pixel features. Mask values selected in `[0, 1]`:
                - 1 for pixel features that are real (i.e. **not masked**),
                - 0 for pixel features that are padding (i.e. **masked**).
                [What are attention masks?](../glossary#attention-mask)
            text_embeds (`torch.FloatTensor` of shape `(batch_size, text_length, hidden_size)`):
                Input to the layer.
            text_attention_mask (`torch.FloatTensor` of shape `(batch_size, text_length)`):
                Text attention mask.
            position_embeddings (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Position embeddings that are added to the queries and keys in each self-attention layer.
            spatial_shapes (`torch.LongTensor` of shape `(num_feature_levels, 2)`):
                Spatial shapes of each feature map.
            level_start_index (`torch.LongTensor` of shape `(num_feature_levels)`):
                Starting index of each feature map.
            valid_ratios (`torch.FloatTensor` of shape `(batch_size, num_feature_levels, 2)`):
                Ratio of valid area in each feature level.
            output_attentions (`bool`, *optional*):
                Whether or not to return the cross attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~file_utils.ModelOutput`] instead of a plain tuple.
        """
        hidden_states = vision_embeds
        
        reference_points = get_reference_points(
            spatial_shapes, valid_ratios, device=vision_embeds.device
        )
        
        all_attentions = () if output_attentions else None
        for i, layer in enumerate(self.layers):
            layer_outputs = layer(
                vision_embeds=hidden_states, 
                vision_attention_mask=vision_attention_mask,
                text_embeds=text_embeds,
                text_attention_mask=text_attention_mask,
                position_embeddings=position_embeddings,
                reference_points=reference_points,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                output_attentions=output_attentions,
            )
            
            hidden_states = layer_outputs[0]
            
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)
        
        if not return_dict:
            return tuple(v for v in [hidden_states, text_embeds, all_attentions] if v is not None)
        return VisionLanguageModelOutput(
            last_vision_hidden_state=hidden_states, last_text_hidden_state=text_embeds, attentions=all_attentions,
        )

