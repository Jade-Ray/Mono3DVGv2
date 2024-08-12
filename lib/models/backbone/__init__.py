from .resnet_backbone import ResNetVisionBackbone
from .swin_backbone import SwinVisionBackbone
from .vitDet_backbone import VitDetVisionBackbone
from .roberta_backbone import RobertaLanguageBackbone

from .position_encoding import build_position_encoding, LearnedPositionEmbedding
from .vision_backbone import VisionBackboneModel
from .language_backbone import LanguageBackboneModel, LanguageBackboneOutput


def build_vision_backbone(config):
    backbone_model_type = config.backbone if config.use_timm_backbone else config.backbone_config.model_type
    
    if backbone_model_type.startswith("resnet"):
        vision_backbone = ResNetVisionBackbone(config)
    elif backbone_model_type.startswith("swin"):
        vision_backbone = SwinVisionBackbone(config)
    elif backbone_model_type.startswith("vit"):
        vision_backbone = VitDetVisionBackbone(config)
    else:
        raise ValueError(f"Not supported {backbone_model_type}")

    return vision_backbone


def build_language_backbone(config):
    if "roberta" in config.text_encoder_type:
        language_backbone = RobertaLanguageBackbone(config)
    else:
        raise ValueError(f"Not supported {config.text_encoder_type}")
    
    return language_backbone