# EV Battery RUL Prediction using GAT + BiLSTM

## Table of Contents
- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [RUL Definition](#rul-definition)
- [Installation](#installation)
- [Usage](#usage)
- [Model Details](#model-details)
- [Training](#training)
- [Inference](#inference)
- [Extending the Model](#extending-the-model)
- [Performance](#performance)

## Overview

This project implements a hybrid deep learning model combining **Graph Attention Networks (GAT)** and **Bidirectional LSTM (BiLSTM)** for predicting the Remaining Useful Life (RUL) of Electric Vehicle batteries using multi-variate time-series data.

### Why GAT + BiLSTM?

**Graph Attention Network (GAT)**
- Learns feature importance automatically via attention mechanism
- Captures non-linear relationships between battery parameters
- Models physical/causal dependencies (e.g., temperature → voltage, current → SoH)
- Interpretable through attention weights
- Scalable - easy to add new features as graph nodes

**Bidirectional LSTM (BiLSTM)**
- Captures temporal dependencies in both directions (past and future)
- Learns degradation patterns over time
- Better context understanding than unidirectional LSTM
- Robust to noise in time-series data
- Handles long-term dependencies effectively

**Combined Power**
- Spatial-Temporal Learning: GAT captures feature interactions, BiLSTM captures time evolution
- Multi-Scale Patterns: GAT learns instantaneous relationships, BiLSTM learns long-term trends
- Adaptive: Attention weights adjust based on battery state and usage patterns
- Predictive: Forecasts future degradation by understanding both structure and dynamics

## Architecture

```
Input Time-Series → GAT (Feature Relationships) → BiLSTM (Temporal Patterns) → Dual Output
                                                                                  ├─ Feature Forecasting
                                                                                  └─ RUL Prediction
```

### Detailed Architecture Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    INPUT TIME-SERIES DATA                       │
│              [batch, time_steps, num_features]                  │
│                   Example: [32, 100, 22]                        │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│              GRAPH ATTENTION NETWORK (GAT)                      │
│                                                                 │
│  For each timestep:                                             │
│    • GAT Layer 1: Multi-Head Attention (4 heads)               │
│      - Computes attention: α_ij = attention(h_i, h_j)          │
│      - Aggregates neighbors: h_i' = Σ(α_ij × W × h_j)          │
│      - Output: [batch, 256]                                     │
│                                                                 │
│    • GAT Layer 2: Multi-Head Attention (4 heads)               │
│      - Refines representations                                  │
│      - Output: [batch, 256]                                     │
│                                                                 │
│  Result: Node embeddings capturing feature relationships        │
│  Output: [batch, time_steps, 256]                              │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│              BIDIRECTIONAL LSTM (BiLSTM)                        │
│                                                                 │
│  Forward LSTM:  Processes t=0 → t=99 (past to present)        │
│  Backward LSTM: Processes t=99 → t=0 (future to past)         │
│                                                                 │
│  • Layer 1: BiLSTM(256 → 128)                                  │
│    - Concatenated output: [batch, time, 256]                   │
│                                                                 │
│  • Layer 2: BiLSTM(256 → 128)                                  │
│    - Concatenated output: [batch, time, 256]                   │
│                                                                 │
│  Last hidden state: [batch, 256]                               │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    PREDICTION HEADS                             │
│                                                                 │
│  1. FORECASTING HEAD                                            │
│     Input: Last hidden [batch, 256]                             │
│     → Linear(256 → 256) + ReLU + Dropout                        │
│     → Linear(256 → forecast_horizon × num_features)             │
│     → Reshape to [batch, 50, 22]                                │
│     Output: Future feature values                               │
│                                                                 │
│  2. RUL PREDICTION HEAD                                         │
│     Input: Last hidden [batch, 256]                             │
│     → Linear(256 → 128) + ReLU + Dropout                        │
│     → Linear(128 → 64) + ReLU + Dropout                         │
│     → Linear(64 → 1)                                            │
│     Output: RUL value (cycles remaining)                        │
│                                                                 │
│  3. ATTENTION WEIGHTS (Interpretability)                        │
│     Input: Last hidden [batch, 256]                             │
│     → Linear(256 → 22) + Softmax                                │
│     Output: Feature importance [batch, 22]                      │
└─────────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                         OUTPUTS                                 │
│                                                                 │
│  1. Forecasted Features: [batch, 50, 22]                       │
│  2. RUL Prediction: [batch, 1]                                 │
│  3. Attention Weights: [batch, 22]                             │
└─────────────────────────────────────────────────────────────────┘
```

## Features

### Current Features (22)

#### Direct Health Indicators (Critical for RUL)
- `Battery_SoH`: State of Health (1 = 100%)
- `Estimated_Battery_Capacity`: Current capacity in Ah
- `Charge_Discharge_Cycles`: Cumulative cycle count

#### Electrical Parameters
- `Battery_Current`: Current flow in Amps
- `Battery_Voltage`: Voltage level in Volts
- `Pack_Current`: Total pack current
- `Pack_Voltage`: Total pack voltage
- `SoP`: State of Power (available power in Watts)

#### Thermal Parameters
- `Battery_Temp`: Temperature in °C
- `LED_UnderTemp`: Low temperature warning flag
- `LED_OverTemp`: High temperature warning flag

#### State Estimation
- `Estimated_SoC`: State of Charge (current charge level %)
- `Estimated_SoE`: State of Energy (available energy)
- `estimated_range`: Predicted driving range

#### Operational Context
- `Vehicle_speed`: Current speed in km/h
- `Distance_Travelled`: Trip distance in km
- `Charging_Status`: Boolean indicating if charging
- `Time_To_Charge`: Remaining charge time

#### Fault Indicators
- `LED_OverCurrent`: Overcurrent fault flag
- `LED_UnderVoltage`: Undervoltage fault flag
- `LED_OverVoltage`: Overvoltage fault flag

#### Temporal
- `timestamp`: Time-series index

### Planned Feature Expansion

#### Environmental Features
Ambient_Temperature, Humidity, Altitude, Weather_Conditions, Road_Grade, Air_Pressure

#### Operational & Usage Features
Driving_Mode, Acceleration_Pattern, Braking_Pattern, Trip_Type, Idle_Time, Average_Speed, Speed_Variance, Daily_Usage_Hours, Charging_Frequency, Fast_Charge_Ratio

#### Technical & Internal Battery Features
Cell_Voltage_Min/Max/Delta, Cell_Temp_Min/Max/Delta, Internal_Resistance, Self_Discharge_Rate, Balancing_Activity, Coulombic_Efficiency, Energy_Efficiency, Power_Fade, Calendar_Age, Storage_Time

#### Vehicle & Hardware Features
Motor_Power_Demand, HVAC_Power_Consumption, Auxiliary_Load, Regenerative_Braking_Energy, Battery_Cooling_System_Status, Thermal_Management_Power, Vehicle_Weight, Tire_Pressure, Aerodynamic_Drag

### Example Data Point

```python
{
    'timestamp': '2025-10-06 16:04:31',
    'Battery_Current': 5.714285597000488,
    'Battery_Voltage': 4.367208432016097,
    'Battery_Temp': 29.897433074274034,
    'Battery_SoH': 1,
    'Estimated_SoE': 28.288033656191335,
    'Estimated_Soc': 51.58602201870697,
    'Estimated_Battery_Capacity': 94.00000187893664,
    'estimated_range': 0,
    'Vehicle_speed': 75.31394819741372,
    'Distance_Travelled': 1.5940879190100734,
    'LED_OverCurrent': False,
    'LED_UnderTemp': False,
    'LED_OverTemp': False,
    'LED_UnderVoltage': False,
    'LED_OverVoltage': False,
    'Pack_Current': 199.9999958950171,
    'Pack_Voltage': 410.5175926095123,
    'SoP': -15794.726385723576,
    'Charging_Status': True,
    'Charge_Discharge_Cycles': 0.0014192705660867639,
    'Time_To_Charge': 0
}
```

## RUL Definition

**Remaining Useful Life (RUL)** is the estimated time or usage remaining before a battery reaches its end-of-life threshold.

### RUL Expressions

1. **Cycle-based RUL**: Number of charge-discharge cycles until SoH drops to 80%
   - Example: "500 cycles remaining"

2. **Time-based RUL**: Calendar time remaining
   - Example: "2.5 years remaining"

3. **Distance-based RUL**: Kilometers the vehicle can travel before battery replacement
   - Example: "50,000 km remaining"

4. **Energy throughput RUL**: Total energy that can still be cycled through the battery
   - Example: "15,000 kWh remaining"

### End-of-Life Threshold

- **Current Battery_SoH**: 1.0 (100%)
- **End-of-life threshold**: 0.8 (80%)
- **RUL**: cycles/time/distance until SoH reaches 0.8

**Important**: RUL is NOT when the battery is completely dead (0% capacity). It's when the battery degrades to a point where it's no longer useful for its intended purpose. For EVs, this is typically 80% of original capacity.

### RUL Calculation in This Model

```python
# RUL is calculated as cycles remaining until SoH ≤ 0.8
EOL_THRESHOLD = 0.8
RUL = Cycles_until_SoH_reaches_0.8

# The model predicts this by:
# 1. Learning degradation patterns from historical data
# 2. Forecasting future feature values
# 3. Estimating when SoH will cross the threshold
```

## Installation

### Requirements

- Python 3.8+
- PyTorch 2.0+
- PyTorch Geometric 2.3+
- NumPy, Pandas, Scikit-learn
- Matplotlib, Seaborn (for visualization)

### Install Dependencies

```bash
pip install -r requirements.txt
```

### requirements.txt

```
torch>=2.0.0
torch-geometric>=2.3.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
```

## Usage

### 1. Data Preparation

Your CSV file should contain these columns:

```
timestamp, Battery_Current, Battery_Voltage, Battery_Temp, Battery_SoH,
Estimated_SoE, Estimated_Soc, Estimated_Battery_Capacity, estimated_range,
Vehicle_speed, Distance_Travelled, LED_OverCurrent, LED_UnderTemp,
LED_OverTemp, LED_UnderVoltage, LED_OverVoltage, Pack_Current,
Pack_Voltage, SoP, Charging_Status, Charge_Discharge_Cycles, Time_To_Charge
```

### 2. Training

```python
from train_bilstm_gnn import prepare_data, BatteryDataset, RULTrainer
from bilstm_gnn_rul_model import BiLSTM_GNN_RUL, BatteryGraphBuilder
from torch.utils.data import DataLoader

# Prepare data
train_data, val_data, test_data, scaler = prepare_data('battery_data.csv')

# Create datasets
train_dataset = BatteryDataset(train_data, window_size=100, forecast_horizon=50)
val_dataset = BatteryDataset(val_data, window_size=100, forecast_horizon=50)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Build graph structure
graph_builder = BatteryGraphBuilder()
edge_index = graph_builder.build_edge_index()

# Initialize model
model = BiLSTM_GNN_RUL(
    num_features=22,
    gnn_hidden_dim=64,
    lstm_hidden_dim=128,
    num_gnn_layers=2,
    num_lstm_layers=2,
    num_heads=4,
    dropout=0.2,
    forecast_horizon=50
)

# Train
trainer = RULTrainer(model, edge_index, device='cuda')
trainer.train(train_loader, val_loader, num_epochs=100)
```

Or simply run:

```bash
python train_bilstm_gnn.py
```

### 3. Inference

```python
from inference_and_visualization import RULPredictor
import numpy as np

# Load trained model
predictor = RULPredictor('best_rul_model.pth', device='cpu')

# Prepare input (100 timesteps × 22 features)
input_data = your_battery_data  # Shape: [100, 22]

# Predict
rul, forecasted_features, attention_weights = predictor.predict_single(input_data)

print(f"Predicted RUL: {rul:.2f} cycles")
print(f"Forecasted features shape: {forecasted_features.shape}")
print(f"Top 5 important features: {attention_weights.argsort()[-5:]}")
```

### 4. Visualization

```python
from inference_and_visualization import RULVisualizer

visualizer = RULVisualizer(predictor.feature_names)

# Plot feature importance
visualizer.plot_feature_attention(attention_weights, top_k=10)

# Plot feature forecast for Battery_SoH
soh_idx = predictor.feature_names.index('Battery_SoH')
visualizer.plot_feature_forecast(
    historical_data, 
    forecasted_features, 
    soh_idx, 
    'Battery_SoH'
)

# Plot RUL over time
visualizer.plot_rul_over_time(rul_predictions)

# Plot health indicators
visualizer.plot_health_indicators(battery_data)

# Plot correlation heatmap
visualizer.plot_correlation_heatmap(battery_data)
```

## Model Details

### Graph Structure

The model uses a directed graph where edges represent physical/causal relationships between battery features:

**Electrical → Health**
- Battery_Current → Battery_SoH
- Battery_Voltage → Battery_SoH
- Battery_Current → Estimated_Battery_Capacity
- Pack_Current → Battery_SoH

**Thermal → Health**
- Battery_Temp → Battery_SoH
- Battery_Temp → Estimated_Battery_Capacity

**Thermal → Electrical**
- Battery_Temp → Battery_Voltage
- Battery_Temp → SoP

**Operational → Electrical**
- Vehicle_speed → Battery_Current
- Charging_Status → Battery_Current
- Charging_Status → Battery_Voltage

**State → Health**
- Estimated_SoC → Battery_SoH
- Charge_Discharge_Cycles → Battery_SoH
- Charge_Discharge_Cycles → Estimated_Battery_Capacity

**Faults → Health**
- LED_OverCurrent → Battery_SoH
- LED_OverTemp → Battery_SoH
- LED_OverVoltage → Battery_SoH

**Bidirectional Relationships**
- Battery_Current ↔ Battery_Voltage
- Pack_Current ↔ Pack_Voltage
- Estimated_SoC ↔ Estimated_SoE
- Battery_SoH ↔ Estimated_Battery_Capacity

### Hyperparameters

#### GAT Parameters
```python
gnn_hidden_dim = 64      # Hidden dimension per attention head
num_heads = 4            # Number of attention heads
num_gnn_layers = 2       # Depth of GAT
dropout = 0.2            # Dropout rate
```

#### BiLSTM Parameters
```python
lstm_hidden_dim = 128    # Hidden units per direction
num_lstm_layers = 2      # Stacked BiLSTM layers
bidirectional = True     # Forward + backward
dropout = 0.2            # Dropout between layers
```

#### Training Parameters
```python
window_size = 100        # Past timesteps
forecast_horizon = 50    # Future timesteps to predict
batch_size = 32          # Samples per batch
learning_rate = 0.001    # Adam optimizer
epochs = 100             # Maximum training epochs
early_stopping = 10      # Patience for early stopping
```

### Loss Function

Multi-task learning with weighted losses:

```python
Total_Loss = α × Forecast_Loss + β × RUL_Loss

where:
  Forecast_Loss = MSE(predicted_features, actual_features)
  RUL_Loss = MSE(predicted_rul, actual_rul)
  
  α = 0.3  # Forecasting weight
  β = 0.7  # RUL weight (primary objective)
```

**Why Multi-Task Learning?**
1. Forecasting helps RUL: Learning to predict features improves understanding of degradation
2. Regularization: Prevents overfitting to RUL alone
3. Interpretability: Can see which features drive RUL changes
4. Robustness: Model learns general battery dynamics

