import torch
from torch import nn

class BiN(nn.Module):
    """
    Bi-directional Normalization (BiN) module that performs normalization along both 
    temporal and feature dimensions of the input tensor.
    
    This module applies a learnable normalization process that:
    1. Normalizes along the temporal dimension
    2. Normalizes along the feature dimension
    3. Combines both normalizations using learnable weights
    
    Args:
        d1 (int): Number of features/channels
        t1 (int): Sequence length/temporal dimension
        device (torch.device): Device to place the tensors on
    """
    def __init__(self, d1, t1, device):
        super().__init__()
        self.t1 = t1
        self.d1 = d1
        self.device = device 

        # Parameters for temporal dimension normalization
        # B1: Bias term for temporal normalization [t1, 1]
        bias1 = torch.Tensor(t1, 1)
        self.B1 = nn.Parameter(bias1)
        nn.init.constant_(self.B1, 0)

        # l1: Scale factor for temporal normalization [t1, 1]
        l1 = torch.Tensor(t1, 1)
        self.l1 = nn.Parameter(l1)
        nn.init.xavier_normal_(self.l1)

        # Parameters for feature dimension normalization
        # B2: Bias term for feature normalization [d1, 1]
        bias2 = torch.Tensor(d1, 1)
        self.B2 = nn.Parameter(bias2)
        nn.init.constant_(self.B2, 0)

        # l2: Scale factor for feature normalization [d1, 1]
        l2 = torch.Tensor(d1, 1)
        self.l2 = nn.Parameter(l2)
        nn.init.xavier_normal_(self.l2)

        # Learnable weights to balance temporal and feature normalization
        # y1, y2: Weights for temporal and feature normalization respectively
        self.y1 = nn.Parameter(torch.tensor(0.5))  # Weight for temporal normalization
        self.y2 = nn.Parameter(torch.tensor(0.5))  # Weight for feature normalization

    def forward(self, x):
        """
        Forward pass of BiN module.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, features, time]
            
        Returns:
            torch.Tensor: Normalized tensor of same shape as input
        """
        # Ensure weights are positive using clamp instead of in-place operations
        y1_clamped = torch.clamp(self.y1, min=0.01)
        y2_clamped = torch.clamp(self.y2, min=0.01)

        # Step 1: Temporal Dimension Normalization
        # Create ones tensor for broadcasting [t1, 1]
        T2 = torch.ones([self.t1, 1], device=self.device)
        
        # Calculate mean along temporal dimension
        x2 = torch.mean(x, dim=2)
        x2 = torch.reshape(x2, (x2.shape[0], x2.shape[1], 1))
        
        # Calculate standard deviation along temporal dimension
        std = torch.std(x, dim=2)
        std = torch.reshape(std, (std.shape[0], std.shape[1], 1))
        
        # Handle zero standard deviation case safely
        std_safe = torch.where(std < 1e-4, torch.ones_like(std), std)

        # Compute temporal normalization
        diff = x - (x2 @ (T2.T))  # Center the data
        Z2 = diff / (std_safe @ (T2.T))  # Scale the data
        
        # Apply learnable parameters
        X2 = self.l2 @ T2.T  # Scale factor
        X2 = X2 * Z2  # Apply scaling
        X2 = X2 + (self.B2 @ T2.T)  # Add bias

        # Step 2: Feature Dimension Normalization
        # Create ones tensor for broadcasting [d1, 1]
        T1 = torch.ones([self.d1, 1], device=self.device)
        
        # Calculate mean along feature dimension
        x1 = torch.mean(x, dim=1)
        x1 = torch.reshape(x1, (x1.shape[0], x1.shape[1], 1))

        # Calculate standard deviation along feature dimension
        std = torch.std(x, dim=1)
        std = torch.reshape(std, (std.shape[0], std.shape[1], 1))

        # Compute feature normalization
        op1 = x1 @ T1.T  # Broadcast mean
        op1 = torch.permute(op1, (0, 2, 1))
        
        op2 = std @ T1.T  # Broadcast standard deviation
        op2 = torch.permute(op2, (0, 2, 1))

        # Normalize features
        z1 = (x - op1) / op2  # Standardize
        X1 = (T1 @ self.l1.T)  # Scale factor
        X1 = X1 * z1  # Apply scaling
        X1 = X1 + (T1 @ self.B1.T)  # Add bias

        # Step 3: Combine both normalizations using learnable weights
        # Weight and combine both normalizations
        x_out = y1_clamped * X1 + y2_clamped * X2  # Weighted sum of both normalizations

        return x_out
