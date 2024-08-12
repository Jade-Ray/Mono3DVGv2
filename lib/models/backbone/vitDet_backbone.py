import torch
from torch import nn

from transformers.utils import requires_backends, logging, is_timm_available
from transformers.models.auto import AutoBackbone


logger = logging.get_logger(__name__)

if is_timm_available():
    from timm import create_model

logger = logging.get_logger(__name__)


class VitDetVisionBackbone(nn.Module):
    """
    VitDet Vision backbone, using either the AutoBackbone API or one from the timm library.
    """
    
    def __init__(self, config):
        super().__init__()

        self.config = config

        if config.use_timm_backbone:
            requires_backends(self, ["timm"])
            kwargs = {
                'img_size': (384, 1280),
            }
            backbone = create_model(
                config.backbone,
                pretrained=config.use_pretrained_backbone,
                features_only=True,
                out_indices=1, # last 1 feature layer
                in_chans=config.num_channels,
                **kwargs,
            )
        else:
            backbone = AutoBackbone.from_config(config.backbone_config)
        
        self.model = backbone
        
        reduction = self.model.feature_info.reduction()[0]
        assert reduction == 16, f"Vit reduction {reduction} is not 16"
        num_channels = self.model.feature_info.channels()[0] if config.use_timm_backbone else self.model.channels
        
        # Simple feature pyramid with 3 levels
        output_proj_list = [
            # upsample feature map with 2x stride
            nn.ConvTranspose2d(num_channels, num_channels, kernel_size=4, stride=2, padding=1),
            nn.Identity(),
            # downsample feature map with 2x stride
            nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=2, padding=1),
        ]
        self.out_proj = nn.ModuleList(output_proj_list)
        self.strides = [reduction // 2, reduction, reduction * 2]
        self.intermediate_channel_sizes = [num_channels, num_channels, num_channels] if config.num_feature_levels > 1 else [num_channels]
        
        backbone_model_type = config.backbone if config.use_timm_backbone else config.backbone_config.model_type
        if "vit" in backbone_model_type:
            for name, parameter in self.model.named_parameters():
                if "blocks.23" not in name: # large vit model with 24 blocks
                    parameter.requires_grad_(False)
    
    def forward(self, pixel_values: torch.Tensor, pixel_mask: torch.Tensor=None):
        # send pixel_values through the model to get list of feature maps
        features = self.model(pixel_values) if self.config.use_timm_backbone else self.model(pixel_values).feature_maps
        feature = features[0] # last feature layer only [B, C, H/16, W/16]
        features = feature[None].repeat(3, 1, 1, 1, 1) # repeat feature map 3 times

        out = []
        for i, feature_map in enumerate(features):
            feature_map = self.out_proj[i](feature_map)
            mask = nn.functional.interpolate(pixel_mask[None].float(), size=feature_map.shape[-2:]).to(torch.bool)[0]
            out.append((feature_map, mask))
        return out