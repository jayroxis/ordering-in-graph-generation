
import warnings
import torch.nn as nn

from .misc import build_model
from timm.models.registry import register_model


@register_model
def vision_gpt(
    in_chans: int = 3,
    emb_dim: int = 1024,
    output_dim: int = 4,
    dtype: str = "float",
    stop_token: int = -1.0,
    gpt_name: str = "gpt_medium",
    conv_backbone: str = "resnet50",
    vit_backbone: str = None,
    pretrained: bool =False,
    vis_enc_cfg: dict = {},
    seq_gen_cfg: dict = {},
    seq_enc_cfg: dict = {},
    stop_detector_cfg: dict = {},
    **kwargs,
) -> nn.Module:
    """
    Constructs a GPT-based framework for visual sequence generation tasks such as object detection
    and visual graph generation.

    Args:
        in_chans    (int):          Number of input channels.
        emb_dim     (int):          Dimension of the embedding used to represent visual 
                                        features and generate sequences.
        output_dim  (int):          Dimension of the output sequences.
        dtype       (str):          Data type of the generated sequence. Can be "float" or "categorical".
        stop_token  (float or int): Value used to indicate the end of a generated sequence.
        gpt_name    (str):          Setting of the GPT model to use.
        conv_backbone (str):        Name of the convolutional backbone model to use for 
                                        visual feature extraction.
        vit_backbone (str):         Name of the Vision Transformer backbone model to use for 
                                        visual feature extraction.
        pretrained  (bool):         Whether to use pretrained weights for the convolutional 
                                        backbone model.
        vis_enc_cfg (dict):         Additional configuration options for the visual encoder model.
        seq_gen_cfg (dict):         Additional configuration options for the sequence generator model.
        seq_enc_cfg (dict):         Additional configuration options for the sequence encoder model.
        stop_detector_cfg (dict):   Additional configuration options for the stop token detector model.
        **kwargs:                   Additional arguments to be passed to the `build_model` function.

    Returns:
        A `torch.nn.Module` object representing the GPT-based framework.
    """
    # Visual Encoder
    if vit_backbone is None and conv_backbone is None:
        raise ValueError("Either `vit_backbone` or `conv_backbone` has to be specified.")
    if vit_backbone is not None and conv_backbone is not None:
        warnings.warn("`conv_backbone` is ignored when `vit_backbone` is specified.")
    if vit_backbone is not None:
        vis_enc = dict(
            model_name="custom_visual_encoder",
            backbone=vit_backbone,
            in_chans=in_chans,
            output_dim=emb_dim, 
            pretrained=pretrained,
            **vis_enc_cfg
        )
    else:
        vis_enc = dict(
            model_name="custom_conv_encoder",
            backbone=conv_backbone,
            in_chans=in_chans,
            output_dim=emb_dim, 
            pretrained=pretrained,
            **vis_enc_cfg
        )

    # Sequence Encoder
    seq_enc = dict(
        model_name="MLPEncoder",
        input_dim=output_dim, 
        output_dim=emb_dim, 
        hidden_dim=int(emb_dim // 2), 
        num_layers=output_dim,
        **seq_enc_cfg
    )

    # Sequence Generator
    seq_gen = dict(
        model_name=gpt_name,
        input_dim=emb_dim,
        output_dim=output_dim,
        **seq_gen_cfg
    )

    # Stop Token Encoder
    if dtype == "categorical":
        stop_detector = dict(
            model_name="stop_token_dector",
            dtype=dtype,
            stop_idx=int(stop_token), 
            threshold=0.5,
            **stop_detector_cfg
        )
    else:
        stop_detector = dict(
            model_name="stop_token_dector",
            dtype=dtype,
            stop_value=float(stop_token), 
            threshold=0.5,
            **stop_detector_cfg
        )
    
    # build final model from all configs
    model_config = dict(
        vis_enc=vis_enc,
        seq_gen=seq_gen,
        stop_detector=stop_detector,
        seq_enc=seq_enc,
    )
    model = build_model(
        model_name="conditional_sequence_generator",
        modality="image",
        model_config=model_config,
    )
    return model