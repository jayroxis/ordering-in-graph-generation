import torch
import torch.nn as nn

class ConstractiveDiscriminator(nn.Module):
    def __init__(
        self,
        input_dim,
        num_layers=2,
        num_heads=8,
        hidden_dim=256,
        dropout_prob=0.0,
        activation='leaky_relu',
        **kwargs,
    ):
        """
        Initializes the Discriminator module that estimates the difference of two input sequences.
        
        Args:
            input_dim (int): The dimensionality of the input sequence.
            num_layers (int): The number of Transformer encoder layers in the Discriminator.
            num_heads (int): The number of attention heads in each Transformer encoder layer.
            hidden_dim (int): The number of hidden units in each Transformer encoder layer.
            dropout_prob (float): The probability of dropping out units in each Transformer 
                                  encoder layer.
            activation (str): The activation function to use in each Transformer encoder layer.
                Must be one of ['leaky_relu', 'gelu'].
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.dropout_prob = dropout_prob
        
        # Define the activation function
        if activation == 'leaky_relu':
            self.activation = nn.LeakyReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            raise ValueError(f"Unknown activation function: {activation}")
        
        # Define the input layer
        self.input_layer = nn.Linear(
            in_features=input_dim,
            out_features=hidden_dim
        )
        
        # Define the Transformer encoder layers
        self.encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim,
                dropout=dropout_prob,
                batch_first=True,
                activation=self.activation
            ) for _ in range(num_layers)
        ])
        
        # Define the final binary classification layer
        self.classifier = nn.Linear(
            in_features=hidden_dim,
            out_features=1
        )

        # Identifier for sequence 1
        self.seq_1_id = nn.Parameter(
            torch.randn(1, 1, self.input_dim),
            requires_grad=True
        )

        # Identifier for sequence 2
        self.seq_2_id = nn.Parameter(
            torch.randn(1, 1, self.input_dim),
            requires_grad=True
        )
        
    def forward(self, seq_1, seq_2):
        """
        Performs a forward pass through the Discriminator module.
        
        Args:
            seq_1 (torch.Tensor): A tensor of shape (batch_size, seq_len, input_dim)
                containing the first input sequence.
            seq_2 (torch.Tensor): A tensor of shape (batch_size, seq_len, input_dim)
                containing the second input sequence.
                
        Returns:
            logits (torch.Tensor): A tensor of shape (batch_size, 1) containing the 
            logits for each input pair, where logits[i] is the logit for the pair 
            (seq_1[i], seq_2[i]).
        """
        # Add the sequence identifier to each input sequence
        seq_1 = self.seq_1_id + seq_1
        seq_2 = self.seq_2_id + seq_2
        
        # Concatenate the two input sequences
        seq = torch.cat([seq_1, seq_2], dim=1)

        # Transform the input sequence to the hidden dimension
        emb = self.input_layer(seq)
        
        # Run the sequence through the Transformer encoder layers
        for layer in self.encoder_layers:
            emb = layer(emb)
        
        # Classify the mean of all token embeddings
        logits = self.classifier(emb.mean(dim=1))
        
        return logits
