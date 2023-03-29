import torch
import torch.nn as nn

from .misc import build_model, get_params_group
from timm.models.registry import register_model



@register_model
def conditional_sequence_generator(modality, model_config, **kwargs):
    if modality == "sequence":
        model = Sequence2Sequence(**model_config)
    elif modality == "image":
        model = Visual2Sequence(**model_config)
    else:
        raise NotImplementedError(f"'{modality}' is not supported yet.")
    return model


class Sequence2Sequence(nn.Module):
    def __init__(
        self, 
        prompt_seq_enc: dict,
        seq_enc: dict,
        seq_gen: dict,
        stop_detector: dict,
        correction: dict = {},
        **kwargs,
    ):
        """
        Base model for Conditional Sequence-to-Sequence generation.

        Args:
            prompt_seq_enc : dict
                Configuration for the prompt sequence encoder.
            seq_enc : dict
                Configuration for the sequence encoder.
            seq_gen : dict
                Configuration for the sequence generator.
            stop_detector : dict
                Configuration for the stop token detector.
            correction : dict (optional)
                Configuration for the correction model to improve generated sequence.
        """
        super().__init__()

        # Prompt Encoder
        self.prompt_seq_enc = build_model(**prompt_seq_enc)
        
        # Sequence Encoder
        self.seq_enc = build_model(**seq_enc)

        # Sequence Generator
        self.seq_gen = build_model(**seq_gen)

        # Stop Token Detector
        self.stop_detector = build_model(**stop_detector)

        # Correction Model to Improve Generated Sequence
        if correction != {}:
            self.correction = build_model(**correction)
        else:
            self.correction = nn.Identity()
    
    def forward(self, prompt, seq=None):
        """
        Forward pass through the model.

        Args:
        - prompt: the input prompt sequence
        - seq: the input sequence tensor 
               (None for first token prediction)

        Returns:
        - generated sequence
        """
        assert seq.ndim == 3, f"Input sequence expect 3D tensor but got {seq.ndim}-D."

        # encode prompt sequence
        prompt_emb = self.prompt_seq_enc(prompt)

        # encode sequence and concatenate with prompt embeddings
        if seq is not None:
            seq_emb = self.seq_enc(seq)

            # concatenate prompt and positional embeddings
            token = torch.cat([prompt_emb, seq_emb], dim=1)
            
            num_out_token = seq.shape[1] + 1
        else:
            token = prompt_emb
            num_out_token = 1
            
        # feed into gpt for causal modeling
        output = self.seq_gen(token)
        pred = output[:, -num_out_token:]
        return pred
    

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
            self.prompt_seq_enc,
            lr=lr,
            weight_decay=weight_decay,
            **kwargs
        )

        # Get sequence encoder parameters group
        params_group += get_params_group(
            self.seq_enc,
            lr=lr,
            weight_decay=weight_decay,
            **kwargs
        )

        # Get sequence Generator parameters group
        params_group += get_params_group(
            self.seq_gen,
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
    
    def generate(
        self, 
        prompt, 
        seq_len=100, 
    ):
        """
        Perform iterative forward pass through the model.

        Args:
        - prompt: the input prompt sequence
        - seq_len: the desired sequence length
        - stop_token_value: the value of the stop token
        - stop_threshold: the maximum absolute difference to consider a token as a stop token

        Returns:
        - generated sequence
        """
        # encode prompt sequence
        prompt_emb = self.prompt_seq_enc(prompt)
        self.seq_gen._init_buffer_()
        
        # initial forward
        token = prompt_emb
        output_seq = self.seq_gen.predict_next(token)
        
        # iterative forward
        for _ in range(seq_len):
            seq_emb = self.seq_enc(output_seq)
            token = torch.cat([
                token,
                seq_emb,
            ], dim=1)
            next_token = self.seq_gen.predict_next(token)
            output_seq = torch.cat([
                output_seq,
                next_token
            ], dim=1)
            stop_flag = self.stop_detector(output_seq)
            if stop_flag:
                break
        output_seq = self.correction(output_seq)
        return output_seq




class Visual2Sequence(nn.Module):
    def __init__(
        self, 
        vis_enc: dict,
        seq_enc: dict,
        seq_gen: dict,
        stop_detector: dict,
        correction: dict = {},
        **kwargs,
    ):
        """
        Base model for Vision-to-Sequence generation.

        Args:
            vis_enc : dict
                Configuration for the visual encoder.
            seq_enc : dict
                Configuration for the sequence encoder.
            seq_gen : dict
                Configuration for the sequence generator.
            stop_detector : dict
                Configuration for the stop token detector.
            correction : dict (optional)
                Configuration for the correction model to improve generated sequence.
        """
        super().__init__()

        # Visual Encoder
        self.vis_enc = build_model(**vis_enc)
        
        # Sequence Encoder
        self.seq_enc = build_model(**seq_enc)

        # Sequence Generator
        self.seq_gen = build_model(**seq_gen)

        # Stop Token Detector
        self.stop_detector = build_model(**stop_detector)

        # Correction Model to Improve Generated Sequence
        if correction != {}:
            self.correction = build_model(**correction)
        else:
            self.correction = nn.Identity()
    
    def forward(self, img, seq=None):
        """
        Forward pass through the model.

        Args:
        - img: the input image tensor
        - seq: the input sequence tensor 
               (None for first token prediction)

        Returns:
        - generated sequence
        """
        assert img.ndim == 4, f"Input image expect 4D tensor but got {img.ndim}-D."
        assert seq.ndim == 3, f"Input sequence expect 3D tensor but got {seq.ndim}-D."

        # encode visual features
        visual_emb = self.vis_enc(img)

        # encode sequence and concatenate with visual embeddings
        if seq is not None:
            seq_emb = self.seq_enc(seq)

            # concatenate visual and positional embeddings
            token = torch.cat([visual_emb, seq_emb], dim=1)
            
            num_out_token = seq.shape[1] + 1
        else:
            token = visual_emb
            num_out_token = 1
            
        # feed into gpt for causal modeling
        output = self.seq_gen(token)
        pred = output[:, -num_out_token:]
        return pred
    

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

        # Get sequence encoder parameters group
        params_group += get_params_group(
            self.seq_enc,
            lr=lr,
            weight_decay=weight_decay,
            **kwargs
        )

        # Get sequence Generator parameters group
        params_group += get_params_group(
            self.seq_gen,
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
    
    def generate(
        self, 
        img, 
        seq_len=100, 
    ):
        """
        Perform iterative forward pass through the model.

        Args:
        - img: the input image tensor
        - seq_len: the desired sequence length
        - stop_token_value: the value of the stop token
        - stop_threshold: the maximum absolute difference to consider a token as a stop token

        Returns:
        - generated sequence
        """
        # encode visual features
        visual_emb = self.vis_enc(img)
        self.seq_gen._init_buffer_()
        
        # initial forward
        token = visual_emb
        output_seq = self.seq_gen.predict_next(token)
        
        # iterative forward
        for _ in range(seq_len):
            seq_emb = self.seq_enc(output_seq)
            token = torch.cat([
                token,
                seq_emb,
            ], dim=1)
            next_token = self.seq_gen.predict_next(token)
            output_seq = torch.cat([
                output_seq,
                next_token
            ], dim=1)
            stop_flag = self.stop_detector(output_seq)
            if stop_flag:
                break
        output_seq = self.correction(output_seq)
        return output_seq
