import torch
import torch.nn as nn

from .gpt import GPT
from .visual import VisualEncoder
from .correction import GraphTransformer
from .positional import SinusoidalMLPEncoder as PositionalEncoder



class GraphGPT(nn.Module):
    def __init__(
        self, 
        vis_enc_name="efficientnet_b0", 
        img_size=256, 
        embed_dim=512, 
        max_freq=10, 
        gpt_output_size=4, 
        gpt_d_model=512, 
        gpt_num_layers=8,
        use_correction_model=False,
        **kwargs,
    ):
        super().__init__()

        # Visual encoder
        self.vis_enc = VisualEncoder(
            vis_enc_name, 
            img_size=img_size, 
            embed_dim=embed_dim
        )

        # Positional encoder
        self.pos_enc = PositionalEncoder(
            d_model=embed_dim, 
            max_freq=max_freq
        )

        # GPT backbone
        self.gpt = GPT(
            input_size=embed_dim, 
            output_size=gpt_output_size, 
            d_model=gpt_d_model, 
            num_layers=gpt_num_layers
        )
        
        self.output_size = gpt_output_size
        self.d_model = gpt_d_model
        self.use_correction_model = use_correction_model

        if use_correction_model:
            # Full-attention correction model
            self._init_correction_model()

    def _init_correction_model(self):
        """
        Initialize post generation correction model.
        """
        self.correction = GraphTransformer(
            input_size=self.output_size, 
            output_size=self.output_size, 
            d_model=self.d_model, 
            num_layers=3
        )

    def forward(self, img, node_pair=None):
        """
        Forward pass through the model.

        Args:
        - img: the input image tensor
        - node_pair: the input node pair tensor (None for inference)

        Returns:
        - the output tensor of the model
        """
        # encode visual features
        visual_emb = self.vis_enc(img)

        # encode node pair features
        if node_pair is not None:
            edge_emb = self.pos_enc(node_pair)

            # concatenate visual and positional embeddings
            token = torch.cat([visual_emb, edge_emb], dim=1)
            
            
            num_out_token = node_pair.shape[1] + 1
        else:
            token = visual_emb
            num_out_token = 1
            
        # feed into gpt for causal modeling
        output = self.gpt(token)
        pred = output[:, -num_out_token:]

        # Post-generation correction
        if self.use_correction_model:
            pred =  (
                pred, 
                self.correction(visual_emb, pred)
            )
            

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
        pos_params_group = self.pos_enc.get_params_group(*args, **kwargs)
        params_group.extend(pos_params_group)

        # Get GPT parameters group
        gpt_params_group = self.gpt.get_params_group(*args, **kwargs)
        params_group.extend(gpt_params_group)

        # Get Correction model parameters group
        if self.use_correction_model:
            lr = kwargs.get("lr")
            weight_decay = kwargs.get("weight_decay")
            params_group = [
                {
                    "params": self.correction.parameters(), 
                    "lr": lr, 
                    "weight_decay": weight_decay,
                },
            ]
        return params_group
    
    def iterative_forward(
        self, img, 
        seq_len=1000, 
        stop_token_value=-1.0, 
        stop_threshold=0.5,
    ):
        """
        Perform iterative forward pass through the model.

        Args:
        - img: the input image tensor
        - seq_len: the maximum output sequence length
        - stop_token_value: the value of the stop token
        - stop_threshold: the maximum absolute difference to consider a token as a stop token

        Returns:
        - the final output tensor of the model
        """
        
        with torch.no_grad():
            # encode visual features
            visual_emb = self.vis_enc(img)
            self.gpt._init_buffer_()
        
            # initial forward
            token = visual_emb
            output_seq = self.gpt.predict_next(token)
            
            # iterative forward
            for _ in range(seq_len - 1):
                edge_emb = self.pos_enc(output_seq)
                token = torch.cat([
                    token,
                    edge_emb,
                ], dim=1)
                next_token = self.gpt.predict_next(token)
                output_seq = torch.cat([
                    output_seq,
                    next_token,
                ], dim=1)

                # stop generation if all tokens are stop tokens
                if stop_token_value is not None and stop_threshold is not None:
                    stop_flag = torch.abs(next_token - stop_token_value)
                    stop_flag = (stop_flag < stop_threshold).all()
                    if stop_flag:
                        break
        
        # Post-generation correction
        if self.use_correction_model:
            output_seq = self.correction(visual_emb, output_seq)
        return output_seq
    
    @torch.no_grad()
    def predict(self, img, seq_len=1000):
        """
        Predict a sequence of tokens for a given input image.

        Args:
        - img: the input image tensor
        - seq_len: the desired sequence length

        Returns:
        - the final output tensor of the model
        """
        return self.iterative_forward(
            img=img, 
            seq_len=seq_len, 
            stop_token_value=-1.0, 
            stop_threshold=0.5,
        )