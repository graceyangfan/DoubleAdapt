import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class TradeEventFusionModel(nn.Module):
    def __init__(
        self,
        continuous_feature_dim,  # Feature dimension of continuous data
        event_feature_dim,       # Feature dimension of event data
        lstm_hidden_dim=128,     # LSTM hidden layer dimension
        fusion_hidden_dim=64,    # Fusion layer dimension
        dropout=0.2,             # Dropout rate
        output_dim=1,            # Output dimension (number of prediction targets)
        num_layers=2,            # Number of LSTM layers
        num_heads=4              # Number of attention heads
    ):
        super(TradeEventFusionModel, self).__init__()
        
        # Store dimensions as class attributes for later use
        self.lstm_hidden_dim = lstm_hidden_dim
        self.fusion_hidden_dim = fusion_hidden_dim
        
        # LSTM for processing continuous data
        self.continuous_lstm = nn.LSTM(
            input_size=continuous_feature_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=False
        )
        
        # Fully connected network for processing event data
        self.event_encoder = nn.Sequential(
            nn.Linear(event_feature_dim, fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, fusion_hidden_dim),
            nn.ReLU()
        )
        
        # Query projection layer for attention mechanism
        self.event_query_proj = nn.Linear(fusion_hidden_dim, lstm_hidden_dim)
        
        # Attention layer
        self.attention = nn.MultiheadAttention(
            embed_dim=lstm_hidden_dim, 
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(lstm_hidden_dim + fusion_hidden_dim, fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, fusion_hidden_dim),
            nn.ReLU()
        )
        
        # Output layer
        self.output_layer = nn.Linear(fusion_hidden_dim, output_dim)
    
    def forward(self, continuous_data, event_data=None):
        """
        Inputs:
          - continuous_data: Continuous data [batch_size, seq_len, continuous_feature_dim]
          - event_data: Event data [batch_size, event_feature_dim] or None
        Outputs:
          - output: Predictions [batch_size, output_dim] or None (if no event data is provided)
        """
        batch_size = continuous_data.size(0)
        
        # If no event data is provided, do not make predictions
        if event_data is None:
            return None
        
        # Process continuous data
        lstm_output, (h_n, _) = self.continuous_lstm(continuous_data)
        # lstm_output: [batch_size, seq_len, lstm_hidden_dim]
        
        # Process event data
        event_repr = self.event_encoder(event_data)  # [batch_size, fusion_hidden_dim]
        
        # Use event representation as query, LSTM output as key/value, compute attention weights
        event_query = event_repr.unsqueeze(1)  # [batch_size, 1, fusion_hidden_dim]
        
        # Project event representation to the same dimension as LSTM output
        event_query_proj = self.event_query_proj(event_query)  # [batch_size, 1, lstm_hidden_dim]
        
        # Use attention mechanism to fuse continuous data and event data
        attn_output, _ = self.attention(
            query=event_query_proj,  
            key=lstm_output,
            value=lstm_output
        )
        # attn_output: [batch_size, 1, lstm_hidden_dim]
        
        # Concatenate attention output with event representation
        attn_output = attn_output.squeeze(1)  # [batch_size, lstm_hidden_dim]
        combined = torch.cat([attn_output, event_repr], dim=1)  # [batch_size, lstm_hidden_dim + fusion_hidden_dim]
        
        # Pass through fusion layer
        fusion_output = self.fusion_layer(combined)  # [batch_size, fusion_hidden_dim]
        
        # Generate predictions
        output = self.output_layer(fusion_output)  # [batch_size, output_dim]
        
        return output