from torch import nn
import torch
from .bin import BiN

class MLPLOB(nn.Module):
    """
    Multi-Layer Perceptron for Limit Order Book (MLPLOB) model.
    
    This model processes sequential financial data through a series of MLP layers,
    with normalization and dimension reduction techniques.
    
    Args:
        hidden_dim (int): Dimension of hidden layers
        num_layers (int): Number of MLP layer pairs
        seq_size (int): Sequence length of input data
        num_features (int): Number of features per time step
        output_dim (int): Dimension of output prediction
        device (torch.device): Device to run computations on
    """
    def __init__(self, 
                 hidden_dim: int,
                 num_layers: int,
                 seq_size: int,
                 num_features: int,
                 output_dim: int,
                 device: torch.device,
                 ) -> None:
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.seq_size = seq_size
        self.num_features = num_features
        self.output_dim = output_dim
        
        # Initialize model components
        self.layers = nn.ModuleList()
        self.first_layer = nn.Linear(num_features, hidden_dim)
        self.norm_layer = BiN(num_features, seq_size, device)
        
        # Build initial layers
        self.layers.append(self.first_layer)
        self.layers.append(nn.GELU())
        
        # Construct alternating MLP layers for feature and sequence dimensions
        for i in range(num_layers):
            if i != num_layers-1:
                # Intermediate layers maintain dimension size
                self.layers.append(MLP(hidden_dim, hidden_dim*4, hidden_dim))
                self.layers.append(MLP(seq_size, seq_size*4, seq_size))
            else:
                # Final layer pair reduces dimensions
                self.layers.append(MLP(hidden_dim, hidden_dim*2, hidden_dim//4))
                self.layers.append(MLP(seq_size, seq_size*2, seq_size//4))
                
        # Calculate flattened dimension after processing
        total_dim = (hidden_dim//4)*(seq_size//4)
        
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
        Forward pass through the MLPLOB network.
        
        Args:
            input_tensor: Input data with shape (batch_size, seq_size, num_features)
            
        Returns:
            Tensor: Output predictions with shape (batch_size, output_dim)
        """
        # Input dimension: (batch_size, seq_size, num_features)
        
        # Apply BiN normalization (requires feature-first format)
        x_transposed = input_tensor.permute(0, 2, 1)
        normalized_x = self.norm_layer(x_transposed)
        x = normalized_x.permute(0, 2, 1)
        
        # Process through main layers with alternating transpositions
        for layer in self.layers:
            x = layer(x)
            # Create a new tensor with transposed dimensions instead of in-place operation
            x = x.permute(0, 2, 1)
        
        # Flatten the output for final processing
        batch_size = x.size(0)
        x_flattened = x.reshape(batch_size, -1)
        
        # Process through final classification layers
        output = x_flattened
        for layer in self.final_layers:
            output = layer(output)
            
        return output
        
        
class MLP(nn.Module):
    """
    Multi-Layer Perceptron block with residual connection and normalization.
    
    Args:
        start_dim (int): Input dimension
        hidden_dim (int): Hidden layer dimension
        final_dim (int): Output dimension
    """
    def __init__(self, 
                 start_dim: int,
                 hidden_dim: int,
                 final_dim: int
                 ) -> None:
        super().__init__()
        
        self.layer_norm = nn.LayerNorm(final_dim)
        self.fc = nn.Linear(start_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, final_dim)
        self.gelu = nn.GELU()
        
    def forward(self, x):
        """
        Forward pass through the MLP block.
        
        Args:
            x: Input tensor
            
        Returns:
            Tensor: Processed tensor with normalization and activation
        """
        # Store input for residual connection
        residual = x
        
        # Two-layer transformation with GELU activation
        transformed = self.fc(x)
        transformed = self.gelu(transformed)
        transformed = self.fc2(transformed)
        
        # Apply residual connection if dimensions match
        if transformed.shape[2] == residual.shape[2]:
            output = transformed + residual
        else:
            output = transformed
            
        # Apply layer normalization and final activation
        normalized = self.layer_norm(output)
        result = self.gelu(normalized)
        
        return result
