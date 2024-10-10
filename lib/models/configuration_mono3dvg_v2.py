"""Mono3DVGv2 model configuration."""

from transformers import PretrainedConfig, logging, CONFIG_MAPPING


logger = logging.get_logger(__name__)


class Mono3DVGv2Config(PretrainedConfig):
    r"""
    This is the cocnfiguration class to store the configuration of a [`Mono3DVGv2Model`]. It is used to instantiate
    a Mono3DVGv2 model according to the specified arguments, defining the model architecture. 
    
    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    
    Args:
        use_timm_backbone (`bool`, *optional*, defaults to `True`):
            Whether or not to use the `timm` library for the backbone. If set to `False`, will use the [`AutoBackbone`]
            API.
        backbone_config (`PretrainedConfig` or `dict`, *optional*):
            The configuration of the backbone model. Only used in case `use_timm_backbone` is set to `False` in which
            case it will default to `ResNetConfig()`.
        num_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        num_queries (`int`, *optional*, defaults to 300):
            Number of object queries, i.e. detection slots. This is the maximal number of objects
            [`DeformableDetrModel`] can detect in a single image. In case `two_stage` is set to `True`, we use
            `two_stage_num_proposals` instead.
        d_model (`int`, *optional*, defaults to 256):
            Dimension of the layers.
        use_dab (`bool`, *optional*, defaults to `False`):
            Whether or not to use the DAB Deformable trick.
        use_text_guided_adapter (`bool`, *optional*, defaults to `True`):
            Whether or not to use the text-guided adapter.
        encoder_layers (`int`, *optional*, defaults to 3):
            Number of encoder layers.
        decoder_layers (`int`, *optional*, defaults to 3):
            Number of decoder layers.
        encoder_attention_heads (`int`, *optional*, defaults to 8):
            Number of attention heads in the encoder.
        decoder_attention_heads (`int`, *optional*, defaults to 8):
            Number of attention heads in the decoder.
        encoder_ffn_dim (`int`, *optional*, defaults to 256):
            Dimension of the feedforward network in the encoder.
        decoder_ffn_dim (`int`, *optional*, defaults to 256):
            Dimension of the feedforward network in the decoder.
        encoder_n_points (`int`, *optional*, defaults to 4):
            Number of points in the encoder.
        decoder_n_points (`int`, *optional*, defaults to 4):
            Number of points in the decoder.
        decoder_self_attn (`bool`, *optional*, defaults to `False`):
            Whether or not to use self-attention in the decoder.
        decoder_depth_residual (`bool`, *optional*, defaults to `False`):
            Whether or not to use depth residual in the decoder.
        decoder_text_residual (`bool`, *optional*, defaults to `False`):
            Whether or not to use text residual in the decoder.
        activation_function (`str` or `function`, *optional*, defaults to `"relu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        dropout (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        activation_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for activations inside the fully connected layer.
        init_std (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        init_xavier_std (`float`, *optional*, defaults to 1):
            The scaling factor used for the Xavier initialization gain in the HM Attention map module.
        init_box (`bool`, *optional*, defaults to `False`):
            Whether or not to initialize the bounding box embed last layer.
        encoder_layerdrop (`float`, *optional*, defaults to 0.0):
            The LayerDrop probability for the encoder. See the [LayerDrop paper](see https://arxiv.org/abs/1909.11556)
            for more details.
        auxiliary_loss (`bool`, *optional*, defaults to `False`):
            Whether auxiliary decoding losses (loss at each decoder layer) are to be used.
        position_embedding_type (`str`, *optional*, defaults to `"sine"`):
            Type of position embeddings to be used on top of the image features. One of `"sine"` or `"learned"`.
        backbone (`str`, *optional*, defaults to `"resnet50"`):
            Name of convolutional backbone to use in case `use_timm_backbone` = `True`. Supports any convolutional
            backbone from the timm package. For a list of all available models, see [this
            page](https://rwightman.github.io/pytorch-image-models/#load-a-pretrained-model).
        use_pretrained_backbone (`bool`, *optional*, defaults to `True`):
            Whether to use pretrained weights for the backbone. Only supported when `use_timm_backbone` = `True`.
        num_feature_levels (`int`, *optional*, defaults to 4):
            The number of input feature levels.
        text_encoder_type (`str`, *optional*, defaults to `"roberta-base"`):
            The type of text encoder to use.
        num_text_output_layers (`int`, *optional*, defaults to 1):
            The number of text output layers.
        num_depth_bins (`int`, *optional*, defaults to 80):
            The number of depth bins.
        depth_min (`float`, *optional*, defaults to 1e-3):
            The minimum depth value.
        depth_max (`float`, *optional*, defaults to 60.0):
            The maximum depth value.
        vl_encoder_type (`str`, *optional*, defaults to `"simple-bridge-tower"`):
            The type of vision-language encoder to use.
        two_stage (`bool`, *optional*, defaults to `False`):
            Whether to apply a two-stage deformable DETR, where the region proposals are also generated by a variant of
            Deformable DETR, which are further fed into the decoder for iterative bounding box refinement.
        with_box_refine (`bool`, *optional*, defaults to `False`):
            Whether to apply iterative bounding box refinement, where each decoder layer refines the bounding boxes
            based on the predictions from the previous layer.
        class_cost (`float`, *optional*, defaults to 2):
            Relative weight of the classification error in the Hungarian matching cost.
        bbox_cost (`float`, *optional*, defaults to 5):
            Relative weight of the L1 error of the bounding box coordinates in the Hungarian matching cost.
        giou_cost (`float`, *optional*, defaults to 2):
            Relative weight of the generalized IoU loss of the bounding box in the Hungarian matching cost.
        center3d_cost (`float`, *optional*, defaults to 10):
            Relative weight of the L1 error of the 3D center coordinates in the Hungarian matching cost.
        cls_loss_coefficient (`float`, *optional*, defaults to 2):
            Relative weight of the classification loss in the object detection loss.
        bbox_loss_coefficient (`float`, *optional*, defaults to 5):
            Relative weight of the L1 bounding box loss in the object detection loss.
        giou_loss_coefficient (`float`, *optional*, defaults to 2):
            Relative weight of the generalized IoU loss in the object detection loss.
        center3d_loss_coefficient (`float`, *optional*, defaults to 10):
            Relative weight of the L1 3D center loss in the object detection loss.
        dim_loss_coefficient (`float`, *optional*, defaults to 1):
            Relative weight of the 3D IoU oriented loss in the object detection loss.
        angle_loss_coefficient (`float`, *optional*, defaults to 1):
            Relative weight of the MultiBin loss in the object detection loss.
        depth_loss_coefficient (`float`, *optional*, defaults to 1):
            Relative weight of the Laplacian algebraic
uncertainty loss in the object detection loss.
        depth_map_loss_coefficient (`float`, *optional*, defaults to 1):
            Relative weight of the DDN loss in the object detection loss.
        focal_alpha (`float`, *optional*, defaults to 0.25):
            Alpha parameter in the focal loss.
        disable_custom_kernels (`bool`, *optional*, defaults to `False`):
            Disable the use of custom CUDA and CPU kernels. This option is necessary for the ONNX export, as custom
            kernels are not supported by PyTorch ONNX export.
    """
    
    model_type = "mono3dvgv2"
    attribute_map = {
        "hidden_size": "d_model",
        "num_attention_heads": "encoder_attention_heads",
    }
    
    def __init__(
        self,
        use_timm_backbone=True,
        backbone_config=None,
        num_channels=3,
        num_queries=50,
        use_dab=False,
        use_text_guided_adapter=True,
        encoder_layers=3,
        encoder_ffn_dim=256,
        encoder_attention_heads=8,
        encoder_n_points=4,
        decoder_layers=3,
        decoder_ffn_dim=256,
        decoder_attention_heads=8,
        decoder_n_points=4,
        decoder_self_attn=False,
        decoder_depth_residual=False,
        decoder_text_residual=False,
        activation_function="relu",
        d_model=256,
        dropout=0.1,
        attention_dropout=0.0,
        activation_dropout=0.0,
        init_std=0.02,
        init_xavier_std=1.0,
        init_box=False,
        return_intermediate=True,
        auxiliary_loss=False,
        position_embedding_type="sine",
        backbone="resnet50",
        use_pretrained_backbone=True,
        freeze_backbone=True,
        num_feature_levels=4,
        text_encoder_type="roberta-base",
        num_text_output_layers=1,
        num_depth_bins=80,
        depth_min=1e-3,
        depth_max=60.0,
        vl_encoder_type='simple-bridge-tower',
        two_stage=False,
        with_box_refine=True,
        class_cost=2,
        bbox_cost=5,
        giou_cost=2,
        center3d_cost=10,
        cls_loss_coefficient=2,
        bbox_loss_coefficient=5,
        giou_loss_coefficient=2,
        center3d_loss_coefficient=10,
        dim_loss_coefficient=1,
        angle_loss_coefficient=1,
        depth_loss_coefficient=1,
        depth_map_loss_coefficient=1,
        focal_alpha=0.25,
        disable_custom_kernels=False,
        **kwargs,
    ):
        if backbone_config is not None and use_timm_backbone:
            raise ValueError("You can't specify both `backbone_config` and `use_timm_backbone`.")

        if not use_timm_backbone:
            if backbone_config is None:
                logger.info("`backbone_config` is `None`. Initializing the config with the default `ResNet` backbone.")
                backbone_config = CONFIG_MAPPING["resnet"](out_features=["stage4"])
            elif isinstance(backbone_config, dict):
                backbone_model_type = backbone_config.get("model_type")
                config_class = CONFIG_MAPPING[backbone_model_type]
                backbone_config = config_class.from_dict(backbone_config)
        self.use_timm_backbone = use_timm_backbone
        self.backbone_config = backbone_config
        self.freeze_backbone = freeze_backbone
        self.num_channels = num_channels
        self.num_queries = num_queries
        self.d_model = d_model
        self.encoder_ffn_dim = encoder_ffn_dim
        self.encoder_layers = encoder_layers
        self.encoder_attention_heads = encoder_attention_heads
        self.decoder_ffn_dim = decoder_ffn_dim
        self.decoder_layers = decoder_layers
        self.decoder_attention_heads = decoder_attention_heads
        self.decoder_self_attn = decoder_self_attn
        self.decoder_depth_residual = decoder_depth_residual
        self.decoder_text_residual = decoder_text_residual
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.activation_function = activation_function
        self.init_std = init_std
        self.init_xavier_std = init_xavier_std
        self.init_box = init_box
        self.auxiliary_loss = auxiliary_loss
        self.position_embedding_type = position_embedding_type
        self.backbone = backbone
        self.use_pretrained_backbone = use_pretrained_backbone
        # text attributes
        self.text_encoder_type = text_encoder_type
        self.num_text_output_layers = num_text_output_layers
        self.use_text_guided_adapter = use_text_guided_adapter
        # depth attributes
        self.num_depth_bins = num_depth_bins
        self.depth_min = depth_min
        self.depth_max = depth_max
        # vision-language encoder attributes
        self.vl_encoder_type = vl_encoder_type
        # deformable attributes
        self.use_dab = use_dab
        self.num_feature_levels = num_feature_levels
        self.encoder_n_points = encoder_n_points
        self.decoder_n_points = decoder_n_points
        self.two_stage = two_stage
        self.with_box_refine = with_box_refine
        if two_stage is True and with_box_refine is False:
            raise ValueError("If two_stage is True, with_box_refine must be True.")
        # Hungarian matcher
        self.class_cost = class_cost
        self.bbox_cost = bbox_cost
        self.giou_cost = giou_cost
        self.center3d_cost = center3d_cost
        # Loss coefficients
        self.cls_loss_coefficient = cls_loss_coefficient
        self.bbox_loss_coefficient = bbox_loss_coefficient
        self.giou_loss_coefficient = giou_loss_coefficient
        self.center3d_loss_coefficient = center3d_loss_coefficient
        self.dim_loss_coefficient = dim_loss_coefficient
        self.angle_loss_coefficient = angle_loss_coefficient
        self.depth_loss_coefficient = depth_loss_coefficient
        self.depth_map_loss_coefficient = depth_map_loss_coefficient
        self.focal_alpha = focal_alpha
        self.disable_custom_kernels = disable_custom_kernels
        if 'is_encoder_decoder' not in kwargs:
            kwargs['is_encoder_decoder'] = True
        super().__init__(**kwargs)
    
    @property
    def num_attention_heads(self) -> int:
        return self.encoder_attention_heads

    @property
    def hidden_size(self) -> int:
        return self.d_model
