from torch import nn
import torch
from .bin import BiN

class GRULOB(nn.Module):
    """
    GRU-based Limit Order Book (GRULOB) model.
    
    This model processes sequential financial data through GRU layers,
    with normalization and dimension reduction techniques.
    
    Args:
        hidden_dim (int): Dimension of hidden layers
        num_layers (int): Number of GRU layers
        seq_size (int): Sequence length of input data
        num_features (int): Number of features per time step
        output_dim (int): Dimension of output prediction
        device (torch.device): Device to run computations on
        dropout (float): Dropout rate for GRU layers
    """
    def __init__(self, 
                 hidden_dim: int,
                 num_layers: int,
                 seq_size: int,
                 num_features: int,
                 output_dim: int,
                 device: torch.device,
                 dropout: float = 0.1
                 ) -> None:
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.seq_size = seq_size
        self.num_features = num_features
        self.output_dim = output_dim
        
        # Initialize model components
        self.norm_layer = BiN(num_features, seq_size, device)
        self.input_proj = nn.Linear(num_features, hidden_dim)
        
        # GRU layer
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True  # Use bidirectional GRU for better context capture
        )
        
        # Calculate dimensions for final layers
        # Due to bidirectional GRU, hidden_dim is doubled
        total_dim = hidden_dim * 2 * seq_size
        
        # Build final classification layers with progressive dimension reduction
        self.final_layers = nn.ModuleList()
        while total_dim > 128:
            self.final_layers.append(nn.Linear(total_dim, total_dim//4))
            self.final_layers.append(nn.GELU())
            total_dim = total_dim//4
            
        # Output projection layer
        self.final_layers.append(nn.Linear(total_dim, output_dim))
        
    def forward(self, input_tensor):
        """
        Forward pass through the GRULOB network.
        
        Args:
            input_tensor: Input data with shape (batch_size, seq_size, num_features)
            
        Returns:
            Tensor: Output predictions with shape (batch_size, output_dim)
        """
        # Apply BiN normalization (requires feature-first format)
        x_transposed = input_tensor.permute(0, 2, 1)
        normalized_x = self.norm_layer(x_transposed)
        x = normalized_x.permute(0, 2, 1)
        
        # Project input to hidden dimension
        x = self.input_proj(x)
        
        # Process through GRU
        output, _ = self.gru(x)  # output shape: (batch_size, seq_size, hidden_dim*2)
        
        # Flatten the output for final processing
        batch_size = output.size(0)
        x_flattened = output.reshape(batch_size, -1)
        
        # Process through final classification layers
        output = x_flattened
        for layer in self.final_layers:
            output = layer(output)
            
        return output