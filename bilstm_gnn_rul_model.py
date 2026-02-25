"""
BiLSTM-GNN Model for Battery RUL Prediction

Architecture:
1. GNN Layer: Captures spatial relationships between features
2. BiLSTM Layer: Captures temporal patterns (forward + backward)
3. Forecasting Head: Predicts future feature values
4. RUL Head: Predicts remaining useful life
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv
from torch_geometric.data import Data
import numpy as np


class BiLSTM_GNN_RUL(nn.Module):
    """
    Hybrid BiLSTM-GNN model for battery RUL prediction
    
    Args:
        num_features: Number of input features (22 currently)
        gnn_hidden_dim: Hidden dimension for GNN layers
        lstm_hidden_dim: Hidden dimension for BiLSTM
        num_gnn_layers: Number of GNN layers
        num_lstm_layers: Number of BiLSTM layers
        num_heads: Number of attention heads (for GAT)
        dropout: Dropout rate
        forecast_horizon: Number of future timesteps to predict
    """
    
    def __init__(
        self,
        num_features=22,
        gnn_hidden_dim=64,
        lstm_hidden_dim=128,
        num_gnn_layers=2,
        num_lstm_layers=2,
        num_heads=4,
        dropout=0.2,
        forecast_horizon=50
    ):
        super(BiLSTM_GNN_RUL, self).__init__()
        
        self.num_features = num_features
        self.gnn_hidden_dim = gnn_hidden_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.forecast_horizon = forecast_horizon
        
        # GNN Layers (Graph Attention Network)
        self.gnn_layers = nn.ModuleList()
        
        # First GNN layer
        self.gnn_layers.append(
            GATConv(num_features, gnn_hidden_dim, heads=num_heads, dropout=dropout)
        )
        
        # Additional GNN layers
        for _ in range(num_gnn_layers - 1):
            self.gnn_layers.append(
                GATConv(gnn_hidden_dim * num_heads, gnn_hidden_dim, 
                       heads=num_heads, dropout=dropout)
            )
        
        # Final GNN output dimension
        gnn_output_dim = gnn_hidden_dim * num_heads
        
        # BiLSTM Layer
        self.bilstm = nn.LSTM(
            input_size=gnn_output_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_lstm_layers > 1 else 0
        )
        
        # BiLSTM output is 2x hidden_dim (forward + backward)
        bilstm_output_dim = lstm_hidden_dim * 2
        
        # Forecasting Head (predicts future feature values)
        self.forecast_head = nn.Sequential(
            nn.Linear(bilstm_output_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, forecast_horizon * num_features)
        )
        
        # RUL Prediction Head
        self.rul_head = nn.Sequential(
            nn.Linear(bilstm_output_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)  # Single RUL value
        )
        
        # Attention mechanism for feature importance
        self.feature_attention = nn.Sequential(
            nn.Linear(bilstm_output_dim, num_features),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, x, edge_index):
        """
        Forward pass
        
        Args:
            x: Input features [batch_size, time_steps, num_features]
            edge_index: Graph edges [2, num_edges]
            
        Returns:
            forecasted_features: [batch_size, forecast_horizon, num_features]
            rul: [batch_size, 1]
            attention_weights: [batch_size, num_features]
        """
        batch_size, time_steps, num_features = x.shape
        
        # Step 1: Apply GNN at each timestep
        gnn_outputs = []
        
        for t in range(time_steps):
            # Get features at timestep t: [batch_size, num_features]
            x_t = x[:, t, :]
            
            # Apply GNN layers
            h = x_t
            for gnn_layer in self.gnn_layers:
                h = gnn_layer(h, edge_index)
                h = F.elu(h)
            
            gnn_outputs.append(h)
        
        # Stack GNN outputs: [batch_size, time_steps, gnn_output_dim]
        gnn_outputs = torch.stack(gnn_outputs, dim=1)
        
        # Step 2: Apply BiLSTM
        lstm_out, (h_n, c_n) = self.bilstm(gnn_outputs)
        # lstm_out: [batch_size, time_steps, lstm_hidden_dim * 2]
        
        # Use the last timestep output for predictions
        last_hidden = lstm_out[:, -1, :]  # [batch_size, lstm_hidden_dim * 2]
        
        # Step 3: Feature attention (interpretability)
        attention_weights = self.feature_attention(last_hidden)
        
        # Step 4: Forecasting
        forecast_flat = self.forecast_head(last_hidden)
        forecasted_features = forecast_flat.view(
            batch_size, self.forecast_horizon, self.num_features
        )
        
        # Step 5: RUL Prediction
        rul = self.rul_head(last_hidden)
        
        return forecasted_features, rul, attention_weights


class BatteryGraphBuilder:
    """
    Builds graph structure for battery features
    """
    
    def __init__(self):
        # Define feature names (22 features)
        self.features = [
            'Battery_Current', 'Battery_Voltage', 'Battery_Temp', 
            'Battery_SoH', 'Estimated_SoE', 'Estimated_Soc',
            'Estimated_Battery_Capacity', 'estimated_range', 'Vehicle_speed',
            'Distance_Travelled', 'LED_OverCurrent', 'LED_UnderTemp',
            'LED_OverTemp', 'LED_UnderVoltage', 'LED_OverVoltage',
            'Pack_Current', 'Pack_Voltage', 'SoP',
            'Charging_Status', 'Charge_Discharge_Cycles', 'Time_To_Charge',
            'timestamp'
        ]
        
        # Note: Vehicle_speed and Distance_Travelled may not be in your database
        # They will be filled with 0.0 if missing
        
        self.feature_to_idx = {f: i for i, f in enumerate(self.features)}
        
    def build_edge_index(self):
        """
        Build edge index based on physical/causal relationships
        
        Returns:
            edge_index: [2, num_edges] tensor
        """
        edges = []
        
        # Helper function to add edge
        def add_edge(src, dst):
            if src in self.feature_to_idx and dst in self.feature_to_idx:
                edges.append([self.feature_to_idx[src], self.feature_to_idx[dst]])
        
        # Electrical → Health relationships
        add_edge('Battery_Current', 'Battery_SoH')
        add_edge('Battery_Voltage', 'Battery_SoH')
        add_edge('Battery_Current', 'Estimated_Battery_Capacity')
        add_edge('Pack_Current', 'Battery_SoH')
        add_edge('Pack_Voltage', 'Battery_SoH')
        
        # Thermal → Health relationships
        add_edge('Battery_Temp', 'Battery_SoH')
        add_edge('Battery_Temp', 'Estimated_Battery_Capacity')
        
        # Thermal → Electrical relationships
        add_edge('Battery_Temp', 'Battery_Voltage')
        add_edge('Battery_Temp', 'SoP')
        add_edge('Battery_Temp', 'Pack_Voltage')
        
        # Operational → Electrical relationships
        add_edge('Vehicle_speed', 'Battery_Current')
        add_edge('Vehicle_speed', 'Pack_Current')
        add_edge('Charging_Status', 'Battery_Current')
        add_edge('Charging_Status', 'Battery_Voltage')
        
        # State → Health relationships
        add_edge('Estimated_Soc', 'Battery_SoH')
        add_edge('Charge_Discharge_Cycles', 'Battery_SoH')
        add_edge('Charge_Discharge_Cycles', 'Estimated_Battery_Capacity')
        
        # Faults → Health relationships
        add_edge('LED_OverCurrent', 'Battery_SoH')
        add_edge('LED_OverTemp', 'Battery_SoH')
        add_edge('LED_OverVoltage', 'Battery_SoH')
        add_edge('LED_UnderTemp', 'Battery_SoH')
        add_edge('LED_UnderVoltage', 'Battery_SoH')
        
        # Bidirectional relationships (mutual influence)
        bidirectional = [
            ('Battery_Current', 'Battery_Voltage'),
            ('Pack_Current', 'Pack_Voltage'),
            ('Estimated_Soc', 'Estimated_SoE'),
            ('Battery_SoH', 'Estimated_Battery_Capacity'),
        ]
        
        for src, dst in bidirectional:
            add_edge(src, dst)
            add_edge(dst, src)  # Reverse direction
        
        # Convert to tensor
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        
        return edge_index
    
    def visualize_graph(self):
        """
        Print graph structure for visualization
        """
        edge_index = self.build_edge_index()
        
        print("Battery Feature Graph Structure:")
        print(f"Number of nodes (features): {len(self.features)}")
        print(f"Number of edges: {edge_index.shape[1]}")
        print("\nEdge connections:")
        
        for i in range(edge_index.shape[1]):
            src_idx = edge_index[0, i].item()
            dst_idx = edge_index[1, i].item()
            src_name = self.features[src_idx]
            dst_name = self.features[dst_idx]
            print(f"  {src_name} → {dst_name}")


# Example usage and training setup
if __name__ == "__main__":
    # Model parameters
    NUM_FEATURES = 22
    WINDOW_SIZE = 100  # Past timesteps
    FORECAST_HORIZON = 50  # Future timesteps
    BATCH_SIZE = 32
    
    # Initialize model
    model = BiLSTM_GNN_RUL(
        num_features=NUM_FEATURES,
        gnn_hidden_dim=64,
        lstm_hidden_dim=128,
        num_gnn_layers=2,
        num_lstm_layers=2,
        num_heads=4,
        dropout=0.2,
        forecast_horizon=FORECAST_HORIZON
    )
    
    # Build graph structure
    graph_builder = BatteryGraphBuilder()
    edge_index = graph_builder.build_edge_index()
    
    # Visualize graph
    graph_builder.visualize_graph()
    
    # Example forward pass
    # Input: [batch_size, window_size, num_features]
    x = torch.randn(BATCH_SIZE, WINDOW_SIZE, NUM_FEATURES)
    
    # Forward pass
    forecasted_features, rul, attention_weights = model(x, edge_index)
    
    print(f"\nModel Output Shapes:")
    print(f"Forecasted features: {forecasted_features.shape}")
    print(f"RUL prediction: {rul.shape}")
    print(f"Attention weights: {attention_weights.shape}")
    
    # Model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
