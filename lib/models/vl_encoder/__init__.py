from .vl_encoder import VisionLanguageEncoder
from .vl_bi_encoder import VisionLanguageBiEncoder
from .vl_simpleBridgeTower import VisionLanguageSimpleBridgeEncoder
from .encoder_layer import VisionLanguageModelOutput


def build_vision_language_encoder(config):
    if config.vl_encoder_type == 'simple-bridge-tower':
        vision_language_encoder = VisionLanguageSimpleBridgeEncoder(config)
    elif config.vl_encoder_type == 'bi-direction':
        vision_language_encoder = VisionLanguageBiEncoder(config)
    elif config.vl_encoder_type == 'fusion':
        vision_language_encoder = VisionLanguageEncoder(config)
    else:
        raise ValueError(f"Not supported {config.vl_encoder_type}")
    
    return vision_language_encoder
