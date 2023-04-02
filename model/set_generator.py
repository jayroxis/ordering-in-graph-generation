import torch
import torch.nn as nn

from .misc import build_model, get_params_group
from timm.models.registry import register_model



@register_model
def conditional_set_generator(modality, model_config, **kwargs):
    if modality == "set":
        model = Set2Set(**model_config)
    elif modality == "image":
        model = Visual2Set(**model_config)
    else:
        raise NotImplementedError(f"'{modality}' is not supported yet.")
    return model


class Set2Set(nn.Module):
    def __init__(
        self, 
        prompt_set_enc: dict,
        set_gen: dict,
        stop_detector: dict,
        correction: dict = {},
        **kwargs,
    ):
        """
        Base model for Conditional Set-to-Set generation.

        Args:
            prompt_set_enc : dict
                Configuration for the prompt set encoder.
            set_gen : dict
                Configuration for the set generator.
            stop_detector : dict
                Configuration for the stop token detector.
            correction : dict (optional)
                Configuration for the correction model to improve generated set.
        """
        super().__init__()

        # Prompt Encoder
        self.prompt_set_enc = build_model(**prompt_set_enc)

        # Set Generator
        self.set_gen = build_model(**set_gen)

        # Stop Token Detector
        self.stop_detector = build_model(**stop_detector)

        # Correction Model to Improve Generated Set
        if correction != {}:
            self.correction = build_model(**correction)
        else:
            self.correction = nn.Identity()
    
    def forward(self, prompt, set=None):
        """
        Forward pass through the model.

        Args:
        - prompt: the input prompt set
        - set: the input set tensor (will be ignored) 
               

        Returns:
        - generated set
        """
        assert set.ndim == 3, f"Input set expect 3D tensor but got {set.ndim}-D."

        # encode prompt set
        prompt_emb = self.prompt_set_enc(prompt)
        token = prompt_emb

        # feed into gpt for causal modeling
        output = self.set_gen(token)

        # encode set and concatenate with visual embeddings
        if set is not None:
            num_out_token = set.shape[1] + 1
            pred = output[:, -num_out_token:]
            return pred
        else:
            return output
        
    def get_params_group(self, lr=2e-4, weight_decay=0, **kwargs):
        """
        Collect the params_group in the PyTorch optimizer input format 
        from each of the three modules and return the combined params_group.
        """
        if hasattr(self, "lr"):
            lr = self.lr
        if hasattr(self, "weight_decay"):
            weight_decay = self.weight_decay

        params_group = []
        
        # Get prompt encoder parameters group
        params_group += get_params_group(
            self.prompt_set_enc,
            lr=lr,
            weight_decay=weight_decay,
            **kwargs
        )

        # Get set Generator parameters group
        params_group += get_params_group(
            self.set_gen,
            lr=lr,
            weight_decay=weight_decay,
            **kwargs
        )

        # Get Stop Token Detector parameters group
        params_group += get_params_group(
            self.stop_detector,
            lr=lr,
            weight_decay=weight_decay,
            **kwargs
        )
        
        # Get correction model parameters group
        params_group += get_params_group(
            self.correction,
            lr=lr,
            weight_decay=weight_decay,
            **kwargs
        )
        return params_group
    


class Visual2Set(nn.Module):
    def __init__(
        self, 
        vis_enc: dict,
        set_gen: dict,
        stop_detector: dict,
        correction: dict = {},
        **kwargs,
    ):
        """
        Base model for Vision-to-Set generation.

        Args:
            vis_enc : dict
                Configuration for the visual encoder.
            set_gen : dict
                Configuration for the set generator.
            stop_detector : dict
                Configuration for the stop token detector.
            correction : dict (optional)
                Configuration for the correction model to improve generated set.
        """
        super().__init__()

        # Visual Encoder
        self.vis_enc = build_model(**vis_enc)

        # Set Generator
        self.set_gen = build_model(**set_gen)

        # Stop Token Detector
        self.stop_detector = build_model(**stop_detector)

        # Correction Model to Improve Generated Set
        if correction != {}:
            self.correction = build_model(**correction)
        else:
            self.correction = nn.Identity()
    
    def forward(self, img, set=None):
        """
        Forward pass through the model.

        Args:
        - img: the input image tensor
        - set: the input set tensor (will be ignored)

        Returns:
        - generated set
        """
        assert img.ndim == 4, f"Input image expect 4D tensor but got {img.ndim}-D."
        if set is not None:
            assert set.ndim == 3, f"Input set expect 3D tensor but got {set.ndim}-D."

        # encode visual features
        visual_emb = self.vis_enc(img)
        token = visual_emb

        # feed into gpt for causal modeling
        output = self.set_gen(token)

        # encode set and concatenate with visual embeddings
        if set is not None:
            num_out_token = set.shape[1] + 1
            pred = output[:, -num_out_token:]
            return pred
        else:
            return output

    def get_params_group(self, lr=2e-4, weight_decay=0, **kwargs):
        """
        Collect the params_group in the PyTorch optimizer input format 
        from each of the three modules and return the combined params_group.
        """
        if hasattr(self, "lr"):
            lr = self.lr
        if hasattr(self, "weight_decay"):
            weight_decay = self.weight_decay

        params_group = []
        
        # Get visual encoder parameters group
        params_group += get_params_group(
            self.vis_enc,
            lr=lr,
            weight_decay=weight_decay,
            **kwargs
        )

        # Get set Generator parameters group
        params_group += get_params_group(
            self.set_gen,
            lr=lr,
            weight_decay=weight_decay,
            **kwargs
        )

        # Get Stop Token Detector parameters group
        params_group += get_params_group(
            self.stop_detector,
            lr=lr,
            weight_decay=weight_decay,
            **kwargs
        )
        
        # Get correction model parameters group
        params_group += get_params_group(
            self.correction,
            lr=lr,
            weight_decay=weight_decay,
            **kwargs
        )
        return params_group
    
