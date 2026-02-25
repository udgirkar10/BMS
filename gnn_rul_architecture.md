# GNN-Based RUL Prediction Architecture

## Why GNN for Battery RUL?

### Key Advantages
✅ **Captures Feature Dependencies**: Models relationships between features (e.g., temp affects voltage, current affects SoH)
✅ **Handles Complex Interactions**: Non-linear relationships between battery parameters
✅ **Scalable**: Easy to add new features as nodes
✅ **Interpretable**: Graph structure shows which features influence each other
✅ **Spatial-Temporal**: Combine GNN (spatial) + RNN/Transformer (temporal)

### Challenges
⚠️ **Graph Construction**: Need to define meaningful edges between features
⚠️ **Temporal Modeling**: Pure GNN doesn't handle time-series well (need hybrid)
⚠️ **Computational Cost**: More complex than standard RNN/Transformer

---

## Graph Construction Strategies

### Option 1: Feature Dependency Graph
Model physical/causal relationships between battery parameters

```
Nodes: Each feature is a node (22 nodes currently)

Edges: Connect features that influence each other

Example Graph Structure:
┌─────────────────────────────────────────────────────────┐
│                                                         │
│  Battery_Temp ──→ Battery_Voltage                      │
│       │                    │                            │
│       ↓                    ↓                            │
│  Battery_SoH ←── Battery_Current ──→ Pack_Current      │
│       │                    │                            │
│       ↓                    ↓                            │
│  Estimated_Capacity   Charge_Cycles                    │
│       │                                                 │
│       ↓                                                 │
│     RUL                                                 │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

**Edge Types**:
- **Causal**: Temperature → Voltage (temp affects voltage)
- **Degradation**: Current → SoH (current causes degradation)
- **Derived**: SoH → Capacity (capacity derived from SoH)
- **Correlation**: Speed → Current (driving affects current draw)

### Option 2: Correlation-Based Graph
Learn edges from data using correlation/mutual information

```python
# Compute correlation matrix
correlation_matrix = compute_correlation(all_features)

# Create edges where |correlation| > threshold
edges = [(i, j) for i, j in correlation_matrix if |corr| > 0.5]
```

### Option 3: Fully Connected Graph
Let the model learn which connections matter

```
All features connected to all features
GNN learns edge weights through attention mechanism
```

### Option 4: Hierarchical Graph (Recommended)
Group features by category with intra/inter-group connections

```
Level 1: Feature Categories
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Electrical  │────→│   Thermal    │────→│    Health    │
│   Features   │     │   Features   │     │  Indicators  │
└──────────────┘     └──────────────┘     └──────────────┘
       │                    │                     │
       ↓                    ↓                     ↓
Level 2: Individual Features
[Current, Voltage] [Temp, Flags]  [SoH, Capacity, Cycles]
       │                    │                     │
       └────────────────────┴─────────────────────┘
                            ↓
                          RUL
```

---

## Hybrid Architecture: Spatial-Temporal GNN

### Architecture: GNN + Temporal Module

```
Input: Time-series of all features
       [batch, time_steps, num_features]
              ↓
┌─────────────────────────────────────────┐
│  For each time step:                    │
│                                         │
│  Feature Vector → Graph Nodes           │
│         ↓                               │
│  Graph Convolution Layers (GCN/GAT)    │
│         ↓                               │
│  Node Embeddings (captures spatial)    │
│                                         │
└─────────────────────────────────────────┘
              ↓
       [batch, time_steps, embedding_dim]
              ↓
┌─────────────────────────────────────────┐
│  Temporal Module:                       │
│                                         │
│  LSTM / GRU / Transformer               │
│         ↓                               │
│  Captures temporal evolution            │
│                                         │
└─────────────────────────────────────────┘
              ↓
       Hidden State
              ↓
┌─────────────────────────────────────────┐
│  Forecasting Head:                      │
│  Predict future feature values          │
└─────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│  RUL Prediction Head:                   │
│  Estimate remaining useful life         │
└─────────────────────────────────────────┘
              ↓
           RUL Output
```

---

## Specific GNN Variants

### 1. Graph Convolutional Network (GCN)
**Best for**: Static graph structure with known dependencies

```python
# Message passing
h_i^(l+1) = σ(Σ(W^(l) * h_j^(l) / sqrt(deg(i) * deg(j))))

# Aggregates neighbor information
# Simple and efficient
```

### 2. Graph Attention Network (GAT)
**Best for**: Learning which features are most important

```python
# Attention mechanism
α_ij = attention(h_i, h_j)  # Learn edge importance
h_i^(l+1) = σ(Σ(α_ij * W * h_j^(l)))

# Automatically learns feature importance
# More flexible than GCN
```

### 3. GraphSAGE
**Best for**: Large graphs, inductive learning

```python
# Sample and aggregate
h_i^(l+1) = σ(W * CONCAT(h_i^(l), AGGREGATE({h_j^(l)})))

# Scalable to many features
# Good for adding new features later
```

### 4. Temporal Graph Networks (TGN)
**Best for**: Your use case - time-evolving graphs

```python
# Combines graph structure + temporal dynamics
# Memory module stores historical patterns
# Updates graph embeddings over time
```

---

## Recommended Architecture for Your Case

### Spatial-Temporal Graph Attention Network (ST-GAT)

```python
class BatteryRUL_STGAT(nn.Module):
    def __init__(self, num_features, hidden_dim, num_heads):
        # Graph structure
        self.adjacency_matrix = build_feature_graph()
        
        # Spatial: Graph Attention Layers
        self.gat1 = GATConv(num_features, hidden_dim, heads=num_heads)
        self.gat2 = GATConv(hidden_dim * num_heads, hidden_dim)
        
        # Temporal: LSTM/Transformer
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=2)
        # OR
        self.transformer = nn.TransformerEncoder(...)
        
        # Forecasting head
        self.forecast_head = nn.Linear(hidden_dim, num_features)
        
        # RUL prediction head
        self.rul_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Output: RUL value
        )
    
    def forward(self, x, edge_index):
        # x: [batch, time_steps, num_features]
        batch_size, time_steps, num_features = x.shape
        
        # Process each time step through GNN
        graph_embeddings = []
        for t in range(time_steps):
            # Graph convolution at time t
            h = self.gat1(x[:, t, :], edge_index)
            h = F.elu(h)
            h = self.gat2(h, edge_index)
            graph_embeddings.append(h)
        
        # Stack: [batch, time_steps, hidden_dim]
        graph_embeddings = torch.stack(graph_embeddings, dim=1)
        
        # Temporal modeling
        temporal_out, (h_n, c_n) = self.lstm(graph_embeddings)
        
        # Forecasting
        future_features = self.forecast_head(temporal_out)
        
        # RUL prediction
        rul = self.rul_head(h_n[-1])  # Use final hidden state
        
        return future_features, rul
```

---

## Graph Construction for Battery Features

### Current 22 Features Graph

```python
# Define feature groups
ELECTRICAL = ['Battery_Current', 'Battery_Voltage', 'Pack_Current', 
              'Pack_Voltage', 'SoP']
THERMAL = ['Battery_Temp', 'LED_UnderTemp', 'LED_OverTemp']
HEALTH = ['Battery_SoH', 'Estimated_Battery_Capacity', 'Charge_Discharge_Cycles']
STATE = ['Estimated_SoC', 'Estimated_SoE', 'estimated_range']
OPERATIONAL = ['Vehicle_speed', 'Distance_Travelled', 'Charging_Status', 
               'Time_To_Charge']
FAULTS = ['LED_OverCurrent', 'LED_UnderVoltage', 'LED_OverVoltage']

# Define edges (feature relationships)
edges = [
    # Electrical → Health
    ('Battery_Current', 'Battery_SoH'),
    ('Battery_Voltage', 'Battery_SoH'),
    ('Battery_Current', 'Estimated_Battery_Capacity'),
    
    # Thermal → Health
    ('Battery_Temp', 'Battery_SoH'),
    ('Battery_Temp', 'Estimated_Battery_Capacity'),
    
    # Thermal → Electrical
    ('Battery_Temp', 'Battery_Voltage'),
    ('Battery_Temp', 'SoP'),
    
    # Operational → Electrical
    ('Vehicle_speed', 'Battery_Current'),
    ('Charging_Status', 'Battery_Current'),
    ('Charging_Status', 'Battery_Voltage'),
    
    # State → Health
    ('Estimated_SoC', 'Battery_SoH'),
    ('Charge_Discharge_Cycles', 'Battery_SoH'),
    ('Charge_Discharge_Cycles', 'Estimated_Battery_Capacity'),
    
    # Faults → Health
    ('LED_OverCurrent', 'Battery_SoH'),
    ('LED_OverTemp', 'Battery_SoH'),
    ('LED_OverVoltage', 'Battery_SoH'),
    
    # Health → RUL (target)
    ('Battery_SoH', 'RUL'),
    ('Estimated_Battery_Capacity', 'RUL'),
    ('Charge_Discharge_Cycles', 'RUL'),
]

# Bidirectional edges (mutual influence)
bidirectional_edges = [
    ('Battery_Current', 'Battery_Voltage'),
    ('Pack_Current', 'Pack_Voltage'),
    ('Estimated_SoC', 'Estimated_SoE'),
]
```

---

## Training Strategy

### Multi-Task Learning
```python
# Loss function
total_loss = α * forecast_loss + β * rul_loss + γ * graph_regularization

forecast_loss = MSE(predicted_features, actual_features)
rul_loss = MSE(predicted_rul, actual_rul)
graph_regularization = L1(edge_weights)  # Sparse graph
```

### Data Preparation
```python
# Sliding window
window_size = 100  # Past time steps
forecast_horizon = 50  # Future time steps

# Create graph data
for i in range(len(data) - window_size - forecast_horizon):
    # Input: [window_size, num_features]
    x = data[i:i+window_size]
    
    # Target features: [forecast_horizon, num_features]
    y_features = data[i+window_size:i+window_size+forecast_horizon]
    
    # Target RUL: scalar
    y_rul = calculate_rul(data[i+window_size])
    
    # Graph structure
    edge_index = build_edge_index(edges)
```

---

## Advantages for Your Expanding Feature Set

### When Adding New Features:

**Environmental Features** (weather, altitude, etc.)
```
Add nodes: Ambient_Temp, Humidity, Altitude
Add edges: Ambient_Temp → Battery_Temp
          Altitude → Vehicle_speed → Battery_Current
```

**Operational Features** (driving patterns)
```
Add nodes: Driving_Mode, Acceleration_Pattern
Add edges: Driving_Mode → Battery_Current
          Acceleration_Pattern → Battery_Temp
```

**Technical Features** (cell-level data)
```
Add nodes: Cell_Voltage_Delta, Internal_Resistance
Add edges: Cell_Voltage_Delta → Battery_SoH
          Internal_Resistance → Battery_SoH
```

**Vehicle Features** (system loads)
```
Add nodes: Motor_Power_Demand, HVAC_Power
Add edges: Motor_Power_Demand → Battery_Current
          HVAC_Power → Battery_Temp
```

The GNN automatically learns how these new features interact with existing ones!

---

## Implementation Libraries

### PyTorch Geometric (Recommended)
```python
import torch
from torch_geometric.nn import GATConv, GCNConv, TransformerConv
from torch_geometric.data import Data, DataLoader

# Easy to implement GNN layers
# Good documentation
# Active community
```

### DGL (Deep Graph Library)
```python
import dgl
from dgl.nn import GATConv, GraphConv

# Efficient for large graphs
# Good for temporal graphs
```

### Spektral (Keras/TensorFlow)
```python
from spektral.layers import GCNConv, GATConv

# If you prefer TensorFlow
# Simpler API
```

---

## Comparison: GNN vs Traditional Approaches

| Aspect | GNN | LSTM/Transformer | Advantage |
|--------|-----|------------------|-----------|
| Feature Relationships | Explicit graph | Implicit | GNN |
| Temporal Modeling | Needs hybrid | Native | LSTM/Transformer |
| Scalability | Good | Good | Tie |
| Interpretability | High (graph structure) | Low | GNN |
| Training Complexity | Higher | Lower | LSTM/Transformer |
| New Features | Easy to add | Retrain needed | GNN |

---

## Recommendation

**Use Hybrid ST-GAT (Spatial-Temporal Graph Attention Network)**:
- GNN captures feature dependencies (spatial)
- LSTM/Transformer captures time evolution (temporal)
- Attention mechanism learns feature importance
- Scalable for your expanding feature set

This gives you the best of both worlds: graph structure for feature relationships + temporal modeling for time-series prediction.

---

## Next Steps

1. Define initial graph structure (22 features)
2. Implement ST-GAT architecture
3. Train on historical battery data
4. Validate RUL predictions
5. Iteratively add new feature categories
6. Refine graph structure based on learned attention weights
