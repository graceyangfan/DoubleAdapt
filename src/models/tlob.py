from torch import nn
import torch
from einops import rearrange
from .bin import BiN
from .mlplob import MLP
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class ComputeQKV(nn.Module):
    """
    Computes Query, Key and Value projections for transformer attention.
    
    Args:
        hidden_dim (int): Dimension of input features
        num_heads (int): Number of attention heads
    """
    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.q = nn.Linear(hidden_dim, hidden_dim*num_heads)
        self.k = nn.Linear(hidden_dim, hidden_dim*num_heads)
        self.v = nn.Linear(hidden_dim, hidden_dim*num_heads)
        
    def forward(self, x):
        """
        Computes query, key, and value projections from input.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_dim)
            
        Returns:
            tuple: Query, key, and value projections
        """
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        return q, k, v


class TransformerLayer(nn.Module):
    """
    Custom transformer layer with multi-head attention and MLP components.
    
    Args:
        hidden_dim (int): Dimension of input features
        num_heads (int): Number of attention heads
        final_dim (int): Output dimension after MLP
        device (torch.device): Device to run computations on
    """
    def __init__(
        self, 
        hidden_dim: int, 
        num_heads: int, 
        final_dim: int,
        device: torch.device,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.norm = nn.LayerNorm(hidden_dim)
        self.qkv = ComputeQKV(hidden_dim, num_heads)
        self.attention = nn.MultiheadAttention(hidden_dim*num_heads, num_heads, batch_first=True, device=device)
        self.mlp = MLP(hidden_dim, hidden_dim*4, final_dim)
        self.w0 = nn.Linear(hidden_dim*num_heads, hidden_dim)
        
    def forward(self, x):
        """
        Forward pass through the transformer layer.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_dim)
            
        Returns:
            tuple: Processed tensor and attention weights
        """
        # Store residual connection
        residual = x
        
        # Compute QKV and apply attention
        q, k, v = self.qkv(x)
        attention_output, attention_weights = self.attention(
            q, k, v, 
            average_attn_weights=False, 
            need_weights=True
        )
        
        # Project attention output back to hidden dimension
        projected_output = self.w0(attention_output)
        
        # First residual connection and normalization
        normalized_output = self.norm(projected_output + residual)
        
        # Apply MLP block
        mlp_output = self.mlp(normalized_output)
        
        # Second residual connection if dimensions match
        final_output = mlp_output
        if final_output.shape[-1] == residual.shape[-1]:
            final_output = mlp_output + residual
            
        return final_output, attention_weights


class TLOB(nn.Module):
    """
    Transformer-based Limit Order Book (TLOB) model for financial time series prediction.
    
    This model processes sequential financial data through alternating transformer layers
    that attend to temporal and feature dimensions separately.
    
    Args:
        hidden_dim (int): Dimension of hidden layers
        num_layers (int): Number of transformer layer pairs
        seq_size (int): Sequence length of input data
        num_features (int): Number of features per time step
        num_heads (int): Number of attention heads
        output_dim (int): Dimension of output prediction
        is_sin_emb (bool): Whether to use sinusoidal positional embedding
        device (torch.device): Device to run computations on
    """
    def __init__(self, 
                 hidden_dim: int,
                 num_layers: int,
                 seq_size: int,
                 num_features: int,
                 num_heads: int,
                 output_dim: int,
                 is_sin_emb: bool,
                 device: torch.device,
                 ) -> None:
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.is_sin_emb = is_sin_emb
        self.seq_size = seq_size
        self.num_heads = num_heads
        self.device = device
        
        # Initialize model components
        self.layers = nn.ModuleList()
        self.first_branch = nn.ModuleList()
        self.second_branch = nn.ModuleList()
        self.norm_layer = BiN(num_features, seq_size, device)
        self.emb_layer = nn.Linear(num_features, hidden_dim)
        
        # Positional encoding
        if is_sin_emb:
            self.pos_encoder = sinusoidal_positional_embedding(seq_size, hidden_dim, device)
        else:
            self.pos_encoder = nn.Parameter(torch.randn(1, seq_size, hidden_dim))
        
        # Build transformer layers
        for i in range(num_layers):
            if i != num_layers-1:
                # Intermediate layers maintain dimension size
                self.layers.append(TransformerLayer(hidden_dim, num_heads, hidden_dim, device))
                self.layers.append(TransformerLayer(seq_size, num_heads, seq_size, device))
            else:
                # Final layer pair reduces dimensions
                self.layers.append(TransformerLayer(hidden_dim, num_heads, hidden_dim//4, device))
                self.layers.append(TransformerLayer(seq_size, num_heads, seq_size//4, device))
        
        # Initialize attention tracking containers
        self.att_temporal = []
        self.att_feature = []
        self.mean_att_distance_temporal = []
        
        # Build final classification layers
        total_dim = (hidden_dim//4)*(seq_size//4)
        self.final_layers = nn.ModuleList()
        while total_dim > 128:
            self.final_layers.append(nn.Linear(total_dim, total_dim//4))
            self.final_layers.append(nn.GELU())
            total_dim = total_dim//4
        self.final_layers.append(nn.Linear(total_dim, output_dim))
        
    def forward(self, input_tensor, store_att=False):
        """
        Forward pass through the TLOB network.
        
        Args:
            input_tensor: Input data with shape (batch_size, seq_size, num_features)
            store_att (bool): Whether to store attention weights for analysis
            
        Returns:
            Tensor: Output predictions with shape (batch_size, output_dim)
        """
        # Apply BiN normalization (requires feature-first format)
        x_transposed = rearrange(input_tensor, 'b s f -> b f s')
        normalized_x = self.norm_layer(x_transposed)
        x = rearrange(normalized_x, 'b f s -> b s f')
        
        # Apply embedding and add positional encoding
        embedded_x = self.emb_layer(x)
        x_with_pos = embedded_x + self.pos_encoder
        
        # Initialize attention tracking arrays if needed
        mean_att_distance_temporal = np.zeros((self.num_layers, self.num_heads))
        att_max_temporal = np.zeros((self.num_layers, 2, self.num_heads, self.seq_size))
        att_max_feature = np.zeros((self.num_layers-1, 2, self.num_heads, self.hidden_dim))
        att_temporal = np.zeros((self.num_layers, self.num_heads, self.seq_size, self.seq_size))
        att_feature = np.zeros((self.num_layers-1, self.num_heads, self.hidden_dim, self.hidden_dim))
        
        # Process through transformer layers
        current_x = x_with_pos
        for i in range(len(self.layers)):
            current_x, att = self.layers[i](current_x)
            
            # Detach attention weights to avoid memory leaks
            att_detached = att.detach()
            
            # Transpose for alternating between feature and temporal attention
            current_x = current_x.permute(0, 2, 1)
            
            # Store attention information if requested
            if store_att:
                if i % 2 == 0:  # Temporal attention
                    att_temporal[i//2] = att_detached[0].cpu().numpy()
                    values, indices = att_detached[0].max(dim=2)
                    mean_att_distance_temporal[i//2] = compute_mean_att_distance(att_detached[0])
                    att_max_temporal[i//2, 0] = indices.cpu().numpy()
                    att_max_temporal[i//2, 1] = values.cpu().numpy()
                elif i % 2 == 1 and i != len(self.layers)-1:  # Feature attention
                    att_feature[i//2] = att_detached[0].cpu().numpy()
                    values, indices = att_detached[0].max(dim=2)
                    att_max_feature[i//2, 0] = indices.cpu().numpy()
                    att_max_feature[i//2, 1] = values.cpu().numpy()
        
        # Store attention statistics
        self.mean_att_distance_temporal.append(mean_att_distance_temporal)
        if store_att:
            self.att_temporal.append(att_max_temporal)
            self.att_feature.append(att_max_feature)
        
        # Reshape and flatten for final processing
        flattened_x = rearrange(current_x, 'b s f -> b (f s) 1')
        flattened_x = flattened_x.reshape(flattened_x.shape[0], -1)
        
        # Process through final layers
        output = flattened_x
        for layer in self.final_layers:
            output = layer(output)
            
        return output


def sinusoidal_positional_embedding(token_sequence_size, token_embedding_dim, device, n=10000.0):
    """
    Generate sinusoidal positional embeddings as described in 'Attention Is All You Need'.
    
    Args:
        token_sequence_size (int): Length of the sequence
        token_embedding_dim (int): Dimension of the embedding
        device (torch.device): Device to place the embeddings on
        n (float): Base for the sinusoidal functions
        
    Returns:
        Tensor: Positional embeddings of shape (token_sequence_size, token_embedding_dim)
    """
    if token_embedding_dim % 2 != 0:
        raise ValueError(f"Sinusoidal positional embedding cannot apply to odd token embedding dim (got dim={token_embedding_dim})")

    T = token_sequence_size
    d = token_embedding_dim

    positions = torch.arange(0, T).unsqueeze_(1)
    embeddings = torch.zeros(T, d)

    # Calculate denominators for sinusoidal functions
    denominators = torch.pow(n, 2*torch.arange(0, d//2)/d)  # 10000^(2i/d_model)
    
    # Apply sin to even indices and cos to odd indices
    embeddings[:, 0::2] = torch.sin(positions/denominators)  # sin(pos/10000^(2i/d_model))
    embeddings[:, 1::2] = torch.cos(positions/denominators)  # cos(pos/10000^(2i/d_model))

    return embeddings.to(device, non_blocking=True)


def count_parameters(layer):
    """
    Count and print the number of trainable parameters in a model.
    
    Args:
        layer: PyTorch model or layer
    """
    param_count = sum(p.numel() for p in layer.parameters() if p.requires_grad)
    print(f"Number of parameters: {param_count}")


def compute_mean_att_distance(att):
    """
    Compute the mean attention distance for each attention head.
    
    This measures how far each token attends on average, which helps
    understand the attention span of different heads.
    
    Args:
        att: Attention weights tensor of shape (num_heads, seq_len, seq_len)
        
    Returns:
        ndarray: Mean attention distances for each head
    """
    att_distances = np.zeros((att.shape[0], att.shape[1]))
    
    for h in range(att.shape[0]):  # For each head
        for key in range(att.shape[2]):  # For each key position
            for query in range(att.shape[1]):  # For each query position
                distance = abs(query-key)
                # Weight the distance by the attention probability
                att_distances[h, key] += torch.abs(att[h, query, key]).cpu().item() * distance
                
    # Average across sequence positions
    mean_distances = att_distances.mean(axis=1)
    return mean_distances
