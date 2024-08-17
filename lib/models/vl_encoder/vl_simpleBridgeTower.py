import torch
from torch import nn

from ..configuration_mono3dvg_v2 import Mono3DVGv2Config
from .encoder_layer import (
    VisionCrossEncoderLayer, 
    LanguageCrossEncoderLayer, 
    DeformableDetrEncoderLayer, 
    get_reference_points, 
    VisionLanguageModelOutput,
)


class LinkTower(nn.Module):
    def __init__(self, hidden_size: int = 256, link_tower_type: str = 'add'):
        super().__init__()
        self.link_tower_type = link_tower_type
        if link_tower_type in ["add", "scaled_add", "interpolate"]:
            if link_tower_type == "scaled_add":
                self.scaled_factor = nn.Parameter(torch.tensor(1.0))
            elif link_tower_type == "interpolate":
                self.beta = nn.Parameter(torch.tensor(0.5))
            self.LayerNorm = nn.LayerNorm(hidden_size)
        else:
            raise NotImplementedError(f"link_tower_type {link_tower_type} is not implemented")

    def forward(self, hidden_states, cross_modal_hidden_states):
        if self.link_tower_type == "add":
            return self.LayerNorm(hidden_states + cross_modal_hidden_states)
        elif self.link_tower_type == "scaled_add":
            return self.LayerNorm(hidden_states * self.scaled_factor + cross_modal_hidden_states)
        elif self.link_tower_type == "interpolate":
            return self.LayerNorm(hidden_states * (1 - self.beta) + cross_modal_hidden_states * self.beta)
        else:
            raise NotImplementedError(f"link_tower_type {self.link_tower_type} is not implemented")


def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()


class VisionLanguageSimpleBridgeEncoder(nn.Module):
    def __init__(self, config: Mono3DVGv2Config):
        super().__init__()

        self.num_layers = config.encoder_layers
        self.embed_dim = config.d_model
        
        self.v_encoder_layers = nn.ModuleList([DeformableDetrEncoderLayer(config) for _ in range(self.num_layers)])
        self.v_ln_post = nn.LayerNorm(self.embed_dim)
        
        self.token_type_embeddings = nn.Embedding(2, self.embed_dim)
        self.token_type_embeddings.apply(init_weights)
        
        self.cross_modal_image_layers = nn.ModuleList([VisionCrossEncoderLayer(config) for _ in range(self.num_layers)])
        self.cross_modal_text_layers = nn.ModuleList([LanguageCrossEncoderLayer(config) for _ in range(self.num_layers)])
        
        # ===================== Initialize BT Components ===================== #
        # just for first layer
        self.cross_modal_text_layernorm = nn.LayerNorm(self.embed_dim)
        self.cross_modal_text_layernorm.apply(init_weights)
        self.cross_modal_image_layernorm = nn.LayerNorm(self.embed_dim)
        self.cross_modal_image_layernorm.apply(init_weights)
        
        self.cross_modal_text_link_tower = nn.ModuleList([LinkTower(self.embed_dim) for _ in range(self.num_layers - 1)])
        self.cross_modal_image_link_tower = nn.ModuleList([LinkTower(self.embed_dim) for _ in range(self.num_layers - 1)])
        
        self.cross_modal_text_link_tower.apply(init_weights)
        self.cross_modal_image_link_tower.apply(init_weights)

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
        reference_points = get_reference_points(
            spatial_shapes, valid_ratios, device=vision_embeds.device
        )
        vision_encoder_outputs = self.v_encoder_layers[0](
            hidden_states=vision_embeds,
            attention_mask=vision_attention_mask,
            position_embeddings=position_embeddings,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            output_attentions=output_attentions,
        )
        vision_embeds = vision_encoder_outputs[0]
        
        # first layer is a special case because we don't have the output from the cross-encoder yet
        cross_modal_text = text_embeds[0]
        
        text_token_type_embeddings = self.token_type_embeddings(
            torch.zeros(1, dtype=torch.long, device=vision_embeds.device)
        ).expand_as(cross_modal_text)
        
        cross_modal_text = self.cross_modal_text_layernorm(cross_modal_text + text_token_type_embeddings)
        
        image_embeds_with_ln = self.v_ln_post(vision_embeds)
        image_token_type_embeddings = self.token_type_embeddings(
            torch.full((1,), 1, dtype=torch.long, device=vision_embeds.device)
        ).expand_as(image_embeds_with_ln)
        image_embeds_with_ln = image_embeds_with_ln + image_token_type_embeddings
        cross_modal_image = self.cross_modal_image_layernorm(image_embeds_with_ln)
        
        all_attentions = () if output_attentions else None
        cross_text_outputs = self.cross_modal_text_layers[0](
            vision_embeds=cross_modal_image,
            vision_attention_mask=vision_attention_mask,
            text_embeds=cross_modal_text,
            text_attention_mask=text_attention_mask,
            position_embeddings=position_embeddings,
            output_attentions=output_attentions,
        )
        cross_text_features = cross_text_outputs[0]

        cross_image_outputs = self.cross_modal_image_layers[0](
            vision_embeds=cross_modal_image,
            vision_attention_mask=vision_attention_mask,
            text_embeds=cross_modal_text,
            text_attention_mask=text_attention_mask,
            position_embeddings=position_embeddings,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            output_attentions=output_attentions,
        )
        cross_image_features = cross_image_outputs[0]
        
        if output_attentions:
            all_attentions = all_attentions + ((cross_text_outputs[1], cross_image_outputs[1]),)
        
        link_layer_index = 0
        for i in range(1, self.num_layers):
            vision_encoder_outputs = self.v_encoder_layers[i](
                hidden_states=vision_embeds,
                attention_mask=vision_attention_mask,
                position_embeddings=position_embeddings,
                reference_points=reference_points,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                output_attentions=output_attentions,
            )
            vision_embeds = vision_encoder_outputs[0]
            image_embeds_with_ln = self.v_ln_post(vision_embeds) + image_token_type_embeddings
            
            text_link_tower = self.cross_modal_text_link_tower[link_layer_index]
            image_link_tower = self.cross_modal_image_link_tower[link_layer_index]
            
            # Bridge layers for textual and visual encoders
            cross_text_features_ = text_link_tower(text_embeds[i] + text_token_type_embeddings,
                cross_text_features,
            )
            cross_image_features_ = image_link_tower(image_embeds_with_ln, cross_image_features)
            
            # Cross-modal encoder via bridge layers of textual and visual encoders
            cross_text_outputs = self.cross_modal_text_layers[link_layer_index + 1](
                vision_embeds=cross_image_features_,
                vision_attention_mask=vision_attention_mask,
                text_embeds=cross_text_features_,
                text_attention_mask=text_attention_mask,
                position_embeddings=position_embeddings,
                output_attentions=output_attentions,
            )
            cross_text_features = cross_text_outputs[0]
            
            cross_image_outputs = self.cross_modal_image_layers[link_layer_index + 1](
                vision_embeds=cross_image_features_,
                vision_attention_mask=vision_attention_mask,
                text_embeds=cross_text_features_,
                text_attention_mask=text_attention_mask,
                position_embeddings=position_embeddings,
                reference_points=reference_points,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                output_attentions=output_attentions,
            )
            cross_image_features = cross_image_outputs[0]
            
            if output_attentions:
                all_attentions = all_attentions + ((cross_text_outputs[1], cross_image_outputs[1]),)
            
            link_layer_index += 1
        
        if not return_dict:
            return tuple(v for v in [cross_image_features, cross_text_features, all_attentions] if v is not None)
        return VisionLanguageModelOutput(
            last_vision_hidden_state=cross_image_features, last_text_hidden_state=cross_text_features, attentions=all_attentions,
        )
        
