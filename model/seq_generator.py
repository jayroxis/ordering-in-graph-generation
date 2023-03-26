import torch
import torch.nn as nn

from .misc import build_model
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
        seq_enc: dict,
        seq_gen: dict,
        stop_detector: dict,
        correction: dict,
        **kwargs,
    ):
        """
        Base model for Sequence-to-Sequence generation.

        Args:
            seq_enc : dict
                Configuration for the sequence encoder.
            seq_gen : dict
                Configuration for the sequence generator.
            stop_detector : dict
                Configuration for the stop token detector.
            correction : dict
                Configuration for the correction model to improve generated sequence.
        """
        super().__init__()
        
        # Sequence Encoder
        self.seq_enc = build_model(**seq_enc)

        # Sequence Generator
        self.seq_gen = build_model(**seq_gen)

        # Stop Token Detector
        self.stop_detector = build_model(**stop_detector)

        # Correction Model to Improve Generated Sequence
        self.correction = build_model(**correction)
    
    def forward(self, seq):
        """
        Forward pass through the model.

        Args:
        - seq: the input sequence tensor 

        Returns:
        - generated sequence
        """
        token = self.seq_enc(seq)

        num_out_token = seq.shape[1] + 1
            
        # feed into gpt for causal modeling
        output = self.seq_gen(token)
        pred = output[:, -num_out_token:]
        
        return pred
    
    def get_params_group(self, *args, **kwargs):
        """
        Collect the params_group in the PyTorch optimizer input format 
        from each of the three modules and return the combined params_group.
        """
        params_group = []

        # Get visual encoder parameters group
        vis_params_group = self.vis_enc.get_params_group(*args, **kwargs)
        params_group.extend(vis_params_group)

        # Get positional encoder parameters group
        pos_params_group = self.seq_enc.get_params_group(*args, **kwargs)
        params_group.extend(pos_params_group)

        # Get GPT parameters group
        gpt_params_group = self.seq_gen.get_params_group(*args, **kwargs)
        params_group.extend(gpt_params_group)

        return params_group
    
    def generate(
        self,
        input_seq, 
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
        token = self.seq_enc(input_seq)
        self.seq_gen._init_buffer_()
        
        # initial forward
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
    
    def get_params_group(self, *args, **kwargs):
        """
        Collect the params_group in the PyTorch optimizer input format 
        from each of the three modules and return the combined params_group.
        """
        params_group = []

        # Get visual encoder parameters group
        if hasattr(self.vis_enc, "get_params_group"):
            vis_enc_params = self.vis_enc.get_params_group(*args, **kwargs)
        else:
            vis_enc_params = [{"params": self.vis_enc.parameters(), }]
        params_group.extend(vis_enc_params)

        # Get positional encoder parameters group
        if hasattr(self.seq_enc, "get_params_group"):
            seq_enc_params = self.seq_enc.get_params_group(*args, **kwargs)
        else:
            seq_enc_params = [{"params": self.seq_enc.parameters(), }]
        params_group.extend(seq_enc_params)

        # Get sequence Generator parameters group
        if hasattr(self.seq_gen, "get_params_group"):
            seq_gen_params = self.seq_gen.get_params_group(*args, **kwargs)
        else:
            seq_gen_params = [{"params": self.seq_gen.parameters(), }]
        params_group.extend(seq_gen_params)

        # Get correction model parameters group
        if hasattr(self.correction, "get_params_group"):
            correction_params = self.correction.get_params_group(*args, **kwargs)
        else:
            correction_params = [{"params": self.correction.parameters(), }]
        params_group.extend(correction_params)

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
