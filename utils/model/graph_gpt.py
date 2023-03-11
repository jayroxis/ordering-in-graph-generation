import torch
import torch.nn as nn
from .gpt import GPT
from .visual import VisualEncoder
from .positional import SinusoidalMLPEncoder as PositionalEncoder



class GraphGPT(nn.Module):
    def __init__(
        self, 
        vis_enc_name="efficientnet_b0", 
        image_size=256, 
        embed_dim=512, 
        max_freq=10, 
        gpt_output_size=4, 
        gpt_d_model=512, 
        gpt_num_layers=8
    ):
        super().__init__()

        # Visual encoder
        self.vis_enc = VisualEncoder(
            vis_enc_name, 
            image_size=image_size, 
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

        return params_group
    
    def iterative_forward(self, img, seq_len=100, stop_token_value=-1.0, stop_threshold=1e-2):
        """
        Perform iterative forward pass through the model.

        Args:
        - img: the input image tensor
        - seq_len: the desired sequence length
        - stop_token_value: the value of the stop token
        - stop_threshold: the maximum absolute difference to consider a token as a stop token

        Returns:
        - the final output tensor of the model
        """

        # Initialize the output sequence with the first token
        output_seq = self.forward(img)

        for i in range(1, seq_len):

            # Get the last token of the output sequence
            last_token = output_seq[:, -1:]

            # Perform a single forward pass with the last token as the input node pair
            pred = self.forward(img, node_pair=last_token)

            # Append the predicted token to the output sequence
            output_seq = torch.cat([output_seq, pred], dim=1)

            # Check if the predicted token is a stop token
            stop_token = torch.tensor([stop_token_value]).to(pred.device)
            stop_token_mask = torch.isclose(pred, stop_token, rtol=0, atol=stop_threshold)
            if torch.any(stop_token_mask):
                stop_idx = torch.where(stop_token_mask)[0][0]
                output_seq = output_seq[:, :(i + stop_idx + 1)]
                break

        return output_seq
    
    @torch.no_grad()
    def predict(self, img, seq_len=100):
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
            stop_threshold=1e-2
        )