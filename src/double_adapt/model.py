import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Any

class LabelAdaptHeads(nn.Module):
    """
    Label Adaptation Head implementing hi(y) = γiy + βi from the paper
    Each head learns a set of scale (γ) and bias (β) parameters
    """
    def __init__(self, num_head):
        super().__init__()
        # γi: scale parameter [1, num_head]
        self.weight = nn.Parameter(torch.empty(1, num_head))
        # βi: bias parameter [1, num_head]
        self.bias = nn.Parameter(torch.ones(1, num_head) / 8)
        # Initialize weight parameters between [0.75, 1.25]
        nn.init.uniform_(self.weight, 0.75, 1.25)

    def forward(self, y, inverse=False):
        """
        Args:
            y: [batch_size] or [batch_size, num_head] input labels
            inverse: bool, whether to perform inverse transformation
        Returns:
            [batch_size, num_head] transformed labels
        """
        if inverse:
            # Ensure y has shape [batch_size, num_head]
            if y.dim() == 1:
                y = y.view(-1, 1).expand(-1, self.weight.size(1))
            # y = (ŷ - βi) / γi
            return (y - self.bias) / (self.weight + 1e-9)
        else:
            # Transform y from [batch_size] to [batch_size, 1]
            y = y.view(-1, 1)
            # ŷ = γiy + βi
            return (self.weight + 1e-9) * y + self.bias
        
class CosineSimilarityAttention(nn.Module):
    """Compute cosine similarity between features and prototype vectors"""
    def __init__(self, num_heads, feature_dim):
        super().__init__()
        self.num_heads = num_heads
        self.feature_dim = feature_dim
        # Prototype vector P
        self.P = nn.Parameter(torch.empty(num_heads, feature_dim))
        # Initialize parameter P
        nn.init.kaiming_uniform_(self.P, a=math.sqrt(5))

    def forward(self, v):
        """
        Compute cosine similarity.
        Args:
            v: [batch_size, feature_dim] or [batch_size, seq_length, feature_dim]
        Returns:
            [batch_size, num_heads] or [batch_size, seq_length, num_heads]
        """
        if len(v.shape) == 2:  # [batch_size, feature_dim]
            batch_size, feature_dim = v.shape
            assert feature_dim == self.feature_dim, "Input feature_dim must match the configured feature_dim!"

            v_expanded = v.unsqueeze(1)  # [batch_size, 1, feature_dim]
            P_expanded = self.P.unsqueeze(0)  # [1, num_heads, feature_dim]
            gate = F.cosine_similarity(v_expanded, P_expanded, dim=2)  # [batch_size, num_heads]

        elif len(v.shape) == 3:  # [batch_size, seq_length, feature_dim]
            batch_size, seq_length, feature_dim = v.shape
            assert feature_dim == self.feature_dim, "Input feature_dim must match the configured feature_dim!"

            v_expanded = v.unsqueeze(2)  # [batch_size, seq_length, 1, feature_dim]
            P_expanded = self.P.unsqueeze(0).unsqueeze(0)  # [1, 1, num_heads, feature_dim]
            gate = F.cosine_similarity(v_expanded, P_expanded, dim=3)  # [batch_size, seq_length, num_heads]

        else:
            raise ValueError("Input tensor shape must be [batch_size, feature_dim] or [batch_size, seq_length, feature_dim]")

        return gate

class LabelAdapter(nn.Module):
    """Label adapter combining feature projection, attention mechanism, and label transformation"""
    def __init__(self, x_dim, num_head=4, temperature=4, hid_dim=32, decay_rate=0.5):
        super().__init__()
        self.num_head = num_head
        self.temperature = temperature
        self.decay_rate = decay_rate
        
        # Feature projection layer
        self.linear = nn.Linear(x_dim, hid_dim, bias=False)
        # Cosine similarity attention
        self.attention = CosineSimilarityAttention(num_head, hid_dim)
        # Label transformation heads
        self.heads = LabelAdaptHeads(num_head)

    def _compute_temporal_weights(self, seq_length, device):
        """Compute geometric decay temporal weights"""
        exponents = torch.arange(seq_length, device=device).float()
        weights = torch.pow(self.decay_rate, seq_length - 1 - exponents)
        return weights / weights.sum()

    def forward(self, x, y, inverse=False):
        """
        Forward propagation function
        Args:
            x: [batch_size, feature_dim] or [batch_size, sequence_length, feature_dim]
            y: [batch_size] labels
            inverse: bool, whether to perform inverse transformation
        Returns:
            [batch_size] transformed labels
        """
        original_shape = x.shape
        
        if len(original_shape) == 3:
            batch_size, seq_length, feature_dim = original_shape
            
            # Project to hidden space
            v = self.linear(x)  # [batch_size, seq_length, hid_dim]
            
            # Compute attention [batch_size, seq_length, num_head]
            gate = self.attention(v)
            
            # Compute geometric decay temporal weights
            temporal_weights = self._compute_temporal_weights(seq_length, x.device)
            
            # Apply temporal weights to similarities
            gate = (gate * temporal_weights.view(1, -1, 1)).sum(dim=1)  # [batch_size, num_head]
            
        else:
            # 2D input
            v = self.linear(x)  # [batch_size, hid_dim]
            gate = self.attention(v)  # [batch_size, num_head]
            
        # Apply temperature scaling and softmax
        gate = torch.softmax(gate / self.temperature, -1)  # [batch_size, num_head]
        
        # Apply label transformation and weighted sum by attention weights
        transformed_labels = self.heads(y, inverse)  # [batch_size, num_head]
        return (gate * transformed_labels).sum(-1)  # [batch_size]

class FeatureAdapter(nn.Module):
    """
    Feature Adapter implementing G(x) as defined in Eq. (6)
    x̃ = G(x) := x + Σ(si * gi(x))
    where:
    - si is computed using cosine similarity with temperature scaling
    - gi(x) = Wix + bi is implemented as a simple dense layer
    """
    def __init__(self, feature_dim, num_head=4, temperature=4.0):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_head = num_head
        self.temperature = temperature
        
        # Cosine similarity attention for computing si
        self.attention = CosineSimilarityAttention(
            num_heads=num_head,
            feature_dim=feature_dim
        )
        
        # Feature transformation heads gi(x) = Wix + bi
        self.transform_heads = nn.ModuleList([
            nn.Linear(feature_dim, feature_dim, bias=True)
            for _ in range(num_head)
        ])

    def forward(self, x):
        """
        Args:
            x: [batch_size, feature_dim] or [batch_size, sequence_length, feature_dim]
        Returns:
            x̃: Same shape as input
        """
        original_shape = x.shape
        
        if len(original_shape) == 2:
            # 2D input [batch_size, feature_dim]
            # Compute attention weights
            attention_scores = self.attention(x)  # [batch_size, num_head]
            # Apply temperature scaling and softmax
            attention_weights = torch.softmax(attention_scores / self.temperature, dim=-1)
            
            # Apply feature transformation and aggregate
            transformed = []
            for i, head in enumerate(self.transform_heads):
                # head(x): [batch_size, feature_dim]
                # attention_weights[:, i:i+1]: [batch_size, 1]
                transformed.append(attention_weights[:, i:i+1] * head(x))
            
            # Sum and add residual connection
            return x + sum(transformed)
            
        elif len(original_shape) == 3:
            # 3D input [batch_size, sequence_length, feature_dim]
            batch_size, seq_length, _ = original_shape
            
            # Compute attention weights [batch_size, sequence_length, num_head]
            attention_scores = self.attention(x)
            # Apply temperature scaling and softmax
            attention_weights = torch.softmax(attention_scores / self.temperature, dim=-1)
            
            # Apply feature transformation and aggregate
            transformed = []
            for i, head in enumerate(self.transform_heads):
                # head(x): [batch_size, sequence_length, feature_dim]
                # attention_weights[:, :, i:i+1]: [batch_size, sequence_length, 1]
                transformed.append(
                    attention_weights[:, :, i:i+1] * head(x)
                )
            
            # Sum and add residual connection
            return x + sum(transformed)
            
        else:
            raise ValueError("Input tensor shape must be [batch_size, feature_dim] or [batch_size, sequence_length, feature_dim]")

class ForecastModel(nn.Module):
    """Base forecasting model"""
    def __init__(
        self,
        model: nn.Module, 
        device: torch.device = None
    ):
        super().__init__()
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model
        # Move model to device
        self.model.to(self.device)

    def forward(self, x: torch.Tensor, model: nn.Module = None) -> torch.Tensor:
        if not model:
            model = self.model
        # Ensure input is on correct device
        x = x.to(self.device)
        # Generate prediction
        predictions = model(x)
        return predictions

class DoubleAdapt(ForecastModel):
    """
    Double adaptation model combining feature and label adaptation.
    
    Args:
        model (nn.Module): Model adapter model.
        feature_dim (int): The feature dimension.
        num_head (int): The number of heads in the adaptation modules.
        temperature (float): The temperature parameter for the attention mechanism.
        device (torch.device): The device to use for the model.
    """
    def __init__(
        self,
        model: nn.Module,
        feature_dim: int,
        num_head: int = 8,
        temperature: float = 10.0,
        device: torch.device = None
    ):
        super().__init__(model, device)
        
        # Initialize adapters
        self.feature_adapter = FeatureAdapter(
            feature_dim=feature_dim,
            num_head=num_head,
            temperature=temperature
        )
        
        self.label_adapter = LabelAdapter(
            x_dim=feature_dim,
            num_head=num_head,
            temperature=temperature
        )
        
        # Move adapters to device
        self.feature_adapter.to(self.device)
        self.label_adapter.to(self.device)
        
        # Collect meta-parameters
        self.meta_parameters = list(self.feature_adapter.parameters()) + \
                             list(self.label_adapter.parameters())

    def forward(
        self, 
        x: torch.Tensor,
        model: nn.Module = None,
        transform: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Ensure input is on correct device
        x = x.to(self.device)
        
        # Apply feature adaptation if requested
        if transform:
            x = self.feature_adapter(x)
        # Generate prediction
        return super().forward(x, model),  x