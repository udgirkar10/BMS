# EV Battery RUL Prediction using GAT + BiLSTM

## Table of Contents
- [Overview](#overview)
- [Architecture](#architecture)
- [Current Implementation](#current-implementation)
- [Future Scope](#future-scope)
- [RUL Definition](#rul-definition)
- [Installation & Usage](#installation--usage)
- [Model Details](#model-details)
- [Performance](#performance)

## Overview

This project implements a hybrid deep learning model combining **Graph Attention Networks (GAT)** and **Bidirectional LSTM (BiLSTM)** for predicting the Remaining Useful Life (RUL) of Electric Vehicle batteries using multi-variate time-series data.

### Why GAT + BiLSTM?

#### Graph Attention Network (GAT)

**What it does:**
- Learns feature importance automatically via attention mechanism
- Captures non-linear relationships between battery parameters
- Models physical/causal dependencies (e.g., temperature → voltage, current → SoH)
- Provides interpretability through attention weights
- Scales easily - just add new features as graph nodes

**How it works:**
GAT treats each battery feature as a node in a graph. Edges between nodes represent physical or causal relationships (like how temperature affects voltage). The attention mechanism automatically learns which features are most important for predicting RUL. For example, it might learn that Battery_SoH and Battery_Temp have high importance, while Vehicle_speed has lower importance.

**Key advantage:**
Unlike traditional neural networks that treat all features equally, GAT explicitly models the relationships between features. This makes the model more interpretable and allows it to learn from the physics of battery degradation.

#### Bidirectional LSTM (BiLSTM)

**What it does:**
- Captures temporal dependencies in both directions (past and future)
- Learns how battery degradation patterns evolve over time
- Provides better context understanding than unidirectional LSTM
- Robust to noise in time-series data
- Handles long-term dependencies effectively

**How it works:**
BiLSTM processes the time-series data in two directions: forward (past to present) and backward (future to past). The forward pass learns from historical degradation patterns, while the backward pass provides future context. This bidirectional processing gives the model a complete understanding of the battery's state at any point in time.

**Key advantage:**
Standard LSTM only looks at past data. BiLSTM also considers future context, which is crucial for RUL prediction. For example, if the battery will experience high stress in the near future, the backward pass captures this information and improves the current RUL estimate.

#### Combined Power: GAT + BiLSTM

**Why this combination is optimal:**

1. **Spatial-Temporal Learning**
   - GAT captures spatial relationships (how features interact with each other)
   - BiLSTM captures temporal evolution (how these interactions change over time)
   - Together, they model both "what affects what" and "how it changes"

2. **Multi-Scale Patterns**
   - GAT learns instantaneous relationships (e.g., current temperature affects current voltage)
   - BiLSTM learns long-term trends (e.g., capacity fade over hundreds of cycles)
   - Captures both immediate effects and gradual degradation

3. **Adaptive Predictions**
   - Attention weights adjust based on battery state and usage patterns
   - Model adapts to different operating conditions automatically
   - Personalized predictions for different driving styles and environments

4. **Interpretability**
   - Graph structure shows which features influence RUL
   - Attention weights show how important each feature is
   - Temporal patterns show how degradation progresses
   - Users can understand why the model predicts a certain RUL

5. **Scalability**
   - Easy to add new features (environmental, operational, technical)
   - No architecture changes needed - just extend the graph
   - Model automatically learns how new features interact with existing ones

**The workflow:**
```
1. Input: Time-series of 20 battery features
2. GAT: For each timestep, learn feature relationships via attention
3. BiLSTM: Process the sequence bidirectionally to capture temporal patterns
4. Output: RUL prediction + feature forecasts + attention weights
```

This architecture is specifically designed for battery RUL prediction because it addresses the key challenges:
- Batteries have complex feature interactions (GAT handles this)
- Degradation is a temporal process (BiLSTM handles this)
- Need interpretability for trust and debugging (attention provides this)
- Must scale to more features over time (graph structure enables this)

## Architecture

### High-Level Overview

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
│                   Example: [32, 100, 20]                        │
│                                                                 │
│  • batch: Number of samples processed together (32)            │
│  • time_steps: Historical window size (100 timesteps)          │
│  • num_features: Battery parameters (20 features)              │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│              GRAPH ATTENTION NETWORK (GAT)                      │
│                                                                 │
│  Purpose: Learn spatial relationships between features          │
│                                                                 │
│  For each timestep t:                                           │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Step 1: Feature Vector → Graph Nodes                   │  │
│  │  Each of 20 features becomes a node in the graph        │  │
│  │                                                           │  │
│  │  Step 2: GAT Layer 1 (Multi-Head Attention, 4 heads)    │  │
│  │  • Computes attention: α_ij = attention(h_i, h_j)       │  │
│  │    - Learns which features are important for each other  │  │
│  │    - Example: High attention from Temp to Voltage       │  │
│  │                                                           │  │
│  │  • Aggregates neighbors: h_i' = Σ(α_ij × W × h_j)       │  │
│  │    - Combines information from connected features        │  │
│  │    - Weighted by attention scores                        │  │
│  │                                                           │  │
│  │  • Multi-head (4 heads): Learn different relationships  │  │
│  │    - Head 1: Electrical relationships                    │  │
│  │    - Head 2: Thermal relationships                       │  │
│  │    - Head 3: Degradation pathways                        │  │
│  │    - Head 4: Operational patterns                        │  │
│  │                                                           │  │
│  │  • Output: [batch, 64 × 4] = [batch, 256]               │  │
│  │                                                           │  │
│  │  Step 3: GAT Layer 2 (Multi-Head Attention, 4 heads)    │  │
│  │  • Refines representations from Layer 1                  │  │
│  │  • Captures higher-order feature interactions           │  │
│  │  • Output: [batch, 64 × 4] = [batch, 256]               │  │
│  │                                                           │  │
│  │  Result: Node embeddings capturing feature relationships │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
│  Output: [batch, time_steps, 256]                              │
│  • Rich representations for each timestep                       │
│  • Encodes "what affects what" at each moment                  │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│              BIDIRECTIONAL LSTM (BiLSTM)                        │
│                                                                 │
│  Purpose: Learn temporal evolution of feature relationships     │
│                                                                 │
│  Input: [batch, time_steps, 256] from GAT                      │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Forward LSTM: Processes t=0 → t=99                     │  │
│  │  • Learns from past degradation patterns                 │  │
│  │  • Captures how battery health declined historically     │  │
│  │  • Output: h_fwd [batch, time, 128]                      │  │
│  │                                                           │  │
│  │  Backward LSTM: Processes t=99 → t=0                    │  │
│  │  • Learns from future context                            │  │
│  │  • Captures upcoming stress patterns                     │  │
│  │  • Output: h_bwd [batch, time, 128]                      │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
│  Layer 1: BiLSTM(256 → 128)                                    │
│  • Concatenated output: [batch, time, 256]                     │
│  • Combines forward and backward information                   │
│                                                                 │
│  Layer 2: BiLSTM(256 → 128)                                    │
│  • Stacked for deeper temporal understanding                   │
│  • Concatenated output: [batch, time, 256]                     │
│  • Captures long-term dependencies                             │
│                                                                 │
│  Last hidden state: [batch, 256]                               │
│  • Summary of entire sequence                                  │
│  • Encodes complete degradation history + future context       │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    PREDICTION HEADS                             │
│                                                                 │
│  Input: Last hidden state [batch, 256]                         │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  HEAD 1: FORECASTING                                    │   │
│  │  ────────────────────                                   │   │
│  │  Purpose: Predict future values of all 20 features      │   │
│  │                                                          │   │
│  │  Architecture:                                           │   │
│  │  Input [batch, 256]                                      │   │
│  │    ↓                                                     │   │
│  │  Linear(256 → 256) + ReLU + Dropout(0.2)                │   │
│  │    ↓                                                     │   │
│  │  Linear(256 → 50 × 20 = 1000)                           │   │
│  │    ↓                                                     │   │
│  │  Reshape to [batch, 50, 20]                             │   │
│  │                                                          │   │
│  │  Output: Future feature values for next 50 timesteps    │   │
│  │  • Predicts how SoH, Temp, Current, etc. will evolve    │   │
│  │  • Used to estimate when SoH crosses 80% threshold      │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  HEAD 2: RUL PREDICTION                                 │   │
│  │  ────────────────────                                   │   │
│  │  Purpose: Directly predict remaining useful life        │   │
│  │                                                          │   │
│  │  Architecture:                                           │   │
│  │  Input [batch, 256]                                      │   │
│  │    ↓                                                     │   │
│  │  Linear(256 → 128) + ReLU + Dropout(0.2)                │   │
│  │    ↓                                                     │   │
│  │  Linear(128 → 64) + ReLU + Dropout(0.2)                 │   │
│  │    ↓                                                     │   │
│  │  Linear(64 → 1)                                          │   │
│  │                                                          │   │
│  │  Output: RUL value (cycles remaining until 80% SoH)     │   │
│  │  • Single scalar value                                   │   │
│  │  • Primary objective of the model                        │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  HEAD 3: ATTENTION WEIGHTS (Interpretability)          │   │
│  │  ─────────────────────────────────────                  │   │
│  │  Purpose: Show which features are most important        │   │
│  │                                                          │   │
│  │  Architecture:                                           │   │
│  │  Input [batch, 256]                                      │   │
│  │    ↓                                                     │   │
│  │  Linear(256 → 20) + Softmax                             │   │
│  │                                                          │   │
│  │  Output: Feature importance scores [batch, 20]          │   │
│  │  • Sums to 1.0 across all features                      │   │
│  │  • Higher values = more important for RUL               │   │
│  │  • Example: SoH=0.25, Temp=0.18, Current=0.12, ...     │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                         OUTPUTS                                 │
│                                                                 │
│  1. Forecasted Features: [batch, 50, 20]                       │
│     → Predicted values of all 20 features for next 50 steps    │
│     → Used for understanding future degradation trajectory     │
│                                                                 │
│  2. RUL Prediction: [batch, 1]                                 │
│     → Remaining useful life in cycles                          │
│     → Primary output for maintenance planning                  │
│                                                                 │
│  3. Attention Weights: [batch, 20]                             │
│     → Importance score for each feature                        │
│     → Used for model interpretability and debugging            │
└─────────────────────────────────────────────────────────────────┘
```

### Key Architectural Decisions

#### 1. Why Multi-Head Attention (4 heads)?
- Different heads learn different types of relationships
- Head 1 might focus on electrical parameters
- Head 2 might focus on thermal effects
- Head 3 might focus on degradation pathways
- Head 4 might focus on operational patterns
- Provides richer feature representations

#### 2. Why 2 GAT Layers?
- Layer 1: Learns direct relationships (Temp → Voltage)
- Layer 2: Learns indirect relationships (Temp → Voltage → SoH)
- Captures higher-order feature interactions
- Balances model capacity with computational cost

#### 3. Why 2 BiLSTM Layers?
- Layer 1: Captures short-term temporal patterns
- Layer 2: Captures long-term degradation trends
- Stacking improves temporal modeling capability
- Handles complex degradation dynamics

#### 4. Why Bidirectional?
- Forward pass: Learns from historical degradation
- Backward pass: Provides future context
- Combined: Better understanding of current battery state
- Critical for accurate RUL prediction

#### 5. Why Dual Output Heads?
- Forecasting head: Learns general battery dynamics
- RUL head: Focuses specifically on remaining life
- Multi-task learning improves both predictions
- Forecasting acts as regularization for RUL

#### 6. Why Attention Weights Output?
- Provides interpretability
- Shows which features drive predictions
- Helps debug model behavior
- Builds trust with users

### Information Flow Summary

```
Raw Features (20) 
  → GAT learns "what affects what" 
  → BiLSTM learns "how it changes over time"
  → Forecasting predicts "what happens next"
  → RUL predicts "when will it fail"
  → Attention shows "what matters most"
```

### Model Capacity

- **Total Parameters**: ~2-3 million
- **GAT**: ~500K parameters (attention mechanisms)
- **BiLSTM**: ~1.5M parameters (temporal modeling)
- **Prediction Heads**: ~500K parameters (forecasting + RUL)
- **Model Size**: 10-15 MB (saved file)
- **Inference Speed**: <10ms per sample (GPU)

## Current Implementation

### Features (22 Total)

#### 1. Direct Health Indicators (3) - Critical for RUL
- **Battery_SoH**: State of Health (1=100%, 0.8=EOL)
- **Estimated_Battery_Capacity**: Current capacity in Ah
- **Charge_Discharge_Cycles**: Cumulative cycle count

#### 2. Electrical Parameters (5)
- **Battery_Current**: Current flow (Amps)
- **Battery_Voltage**: Voltage level (Volts)
- **Pack_Current**: Total pack current
- **Pack_Voltage**: Total pack voltage
- **SoP**: State of Power (Watts)

#### 3. Thermal Parameters (3)
- **Battery_Temp**: Temperature (°C), optimal 20-25°C
- **LED_UnderTemp**: Low temp warning
- **LED_OverTemp**: High temp warning

#### 4. State Estimation (3)
- **Estimated_SoC**: State of Charge (%)
- **Estimated_SoE**: State of Energy
- **estimated_range**: Predicted range (km)

#### 5. Operational Context (4)
- **Charging_Status**: Boolean
- **Time_To_Charge**: Remaining charge time
- **Vehicle_speed**: Current vehicle speed (km/h)
- **Distance_Travelled**: Distance covered (km)

#### 6. Fault Indicators (3)
- **LED_OverCurrent**: Overcurrent fault
- **LED_UnderVoltage**: Undervoltage fault
- **LED_OverVoltage**: Overvoltage fault

#### 7. Temporal (1)
- **timestamp**: Time-series index

### Graph Structure

Features connected via physical/causal relationships:
- Temperature → Voltage, SoH
- Current → SoH, Capacity
- Cycles → Capacity
- Faults → SoH
- Vehicle_speed → Current (power demand)
- Distance_Travelled → Cycles
- Charging Status → Current, Voltage

### Data Requirements

**CSV Format**: All 22 features with proper column names

**Quality Requirements**:
- Consistent sampling rate
- Minimal missing values
- Synchronized timestamps
- 1000+ samples minimum
- 50-100 charge cycles (minimum), 200+ optimal

### Current Capabilities

**Can Do**:
- Predict RUL (cycles until 80% SoH)
- Forecast future feature values
- Identify important features via attention
- Detect degradation patterns
- Adapt to different usage patterns

**Limitations**:
- Requires sufficient degradation data
- Performance depends on data quality
- Cannot predict sudden failures
- Vehicle speed and distance data must be available

## Future Scope

### Potential Feature Expansion

The current implementation uses 22 base features. The model architecture is designed to scale easily by adding more features as nodes in the graph. Below are examples of feature categories that could be added based on data availability and specific use cases.

#### Example Category 1: Environmental Features
**Example Features**: Ambient_Temperature, Humidity, Altitude, Weather_Conditions, Road_Grade, Air_Pressure

**Why these could matter**:
- Ambient temperature affects battery thermal management efficiency
- Humidity impacts electrical connections and corrosion
- Altitude affects air density and cooling performance
- Weather conditions influence driving patterns and energy consumption
- Road grade impacts power demand and regenerative braking

**Potential Impact**: Climate-adaptive predictions, geographic considerations

#### Example Category 2: Operational & Usage Features
**Example Features**: Driving_Mode, Acceleration_Pattern, Braking_Pattern, Trip_Type, Idle_Time, Average_Speed, Speed_Variance, Daily_Usage_Hours, Charging_Frequency, Fast_Charge_Ratio

**Note**: Vehicle_speed and Distance_Travelled are already included in the base 22 features.

**Why these could matter**:
- Aggressive driving accelerates battery degradation
- Fast charging increases thermal stress
- Usage patterns reveal stress cycles
- Idle time affects calendar aging
- Charging frequency impacts cycle life

**Potential Impact**: Personalized predictions based on driving style, usage pattern analysis

#### Example Category 3: Technical & Internal Battery Features
**Example Features**: Cell_Voltage_Min, Cell_Voltage_Max, Cell_Voltage_Delta, Cell_Temp_Min, Cell_Temp_Max, Cell_Temp_Delta, Internal_Resistance, Self_Discharge_Rate, Balancing_Activity, Coulombic_Efficiency, Energy_Efficiency, Power_Fade, Calendar_Age, Storage_Time

**Why these could matter**:
- Cell-level imbalances indicate degradation
- Internal resistance directly correlates with aging
- Efficiency metrics reveal capacity fade
- Calendar aging affects batteries even when not in use
- Balancing activity shows cell health variations

**Potential Impact**: Cell-level diagnostics, early fault detection, advanced efficiency tracking

#### Example Category 4: Vehicle & Hardware Features
**Example Features**: Motor_Power_Demand, HVAC_Power_Consumption, Auxiliary_Load, Regenerative_Braking_Energy, Battery_Cooling_System_Status, Thermal_Management_Power, Vehicle_Weight, Tire_Pressure, Aerodynamic_Drag

**Why these could matter**:
- Power demand patterns affect discharge rates
- HVAC and auxiliary loads impact battery stress
- Regenerative braking affects charge cycles
- Thermal management efficiency influences temperature control
- Vehicle weight and aerodynamics affect energy consumption

**Potential Impact**: Complete vehicle system integration, holistic health assessment

### Potential Enhancements

1. **Multi-Battery Learning**: Train on fleet data from multiple batteries for better generalization and transfer learning
2. **Uncertainty Quantification**: Provide confidence intervals and prediction uncertainty for risk assessment
3. **Online Learning**: Incremental model updates without full retraining for continuous improvement
4. **Anomaly Detection**: Real-time detection of unusual patterns, sensor faults, or sudden degradation events
5. **Real-Time Deployment**: Edge device optimization (<5ms inference, <5MB model size) for in-vehicle deployment
6. **Explainable AI**: SHAP values and counterfactual explanations for deeper interpretability
7. **Multi-Modal RUL**: Provide RUL in multiple units (cycles, time, distance, energy throughput)
8. **Predictive Maintenance**: Recommend optimal charging strategies and maintenance schedules
9. **Digital Twin**: Create virtual battery model for what-if scenario analysis

### Scalability

The GAT + BiLSTM architecture is designed for easy feature expansion:
- **Add new features**: Simply add them as nodes in the graph
- **Define relationships**: Add edges based on domain knowledge
- **Retrain**: Model automatically learns feature importance through attention
- **No architecture changes needed**: Same model structure works for any number of features

This makes the model highly adaptable to different data availability scenarios and evolving requirements.

## RUL Definition

**RUL**: Estimated usage remaining before battery reaches end-of-life threshold (80% SoH for EVs)

**Expressions**:
1. Cycle-based: Cycles until SoH ≤ 80%
2. Time-based: Calendar time remaining
3. Distance-based: Kilometers remaining
4. Energy throughput: kWh remaining

**Calculation**: Model learns degradation patterns, forecasts features, estimates when SoH crosses 80% threshold

## Installation & Usage

### Install
```bash
pip install -r requirements.txt
```

**Requirements**: Python 3.8+, PyTorch 2.0+, PyTorch Geometric 2.3+, NumPy, Pandas, Scikit-learn, Matplotlib, Seaborn

### Train
```bash
python model_train.py
```

Automatically handles data loading, preprocessing, 80/10/10 split, training, and saves best model.

### Inference & Visualization
See `inference_and_visualization.py` for prediction and plotting examples.

## Model Details

### Hyperparameters

**GAT**: 64 hidden dim, 4 heads, 2 layers, 0.2 dropout
**BiLSTM**: 128 hidden units/direction, 2 layers, bidirectional, 0.2 dropout
**Training**: Window=100, Forecast=50, Batch=32, LR=0.001, Epochs=100, Early stop=10
**Features**: 22 (complete dataset)

### Loss Function
Total Loss = 0.3 × Forecast Loss + 0.7 × RUL Loss (MSE)

### Training
- Data: Normalize, sliding windows, RUL labels, 80/10/10 split
- Loop: GAT → BiLSTM → Heads, gradient clipping, Adam, LR scheduling
- Time: 2-5 min/epoch (GPU, 10k samples), 3-8 hours total
- Size: 2-3M parameters, 10-15MB file, 2-4GB GPU memory

## Performance

### Expected Metrics
- RUL MAE: <50 cycles
- RUL RMSE: <75 cycles
- R² Score: >0.85

### Comparison

| Aspect | GAT+BiLSTM | Pure LSTM | Pure GNN | Transformer |
|--------|------------|-----------|----------|-------------|
| Feature relationships | ✅ Explicit | ❌ Implicit | ✅ Explicit | ⚠️ Learned |
| Temporal modeling | ✅ Native | ✅ Native | ❌ Limited | ✅ Native |
| Interpretability | ✅ High | ❌ Low | ⚠️ Medium | ⚠️ Medium |
| Data needed | ✅ Moderate | ✅ Moderate | ✅ Moderate | ❌ Large |
| Computational cost | ✅ Efficient | ✅ Efficient | ✅ Efficient | ❌ High |
| Scalability | ✅ Easy | ⚠️ Retrain | ✅ Easy | ⚠️ Retrain |

### Advantages
✅ Spatial-temporal learning
✅ Interpretable (graph + attention)
✅ Scalable (easy feature addition)
✅ Multi-task learning
✅ Bidirectional context
✅ Efficient inference

## Project Structure

```
├── README.md
├── requirements.txt
├── model_architecture.py               # Model architecture (GAT + BiLSTM)
├── model_train.py                      # Training script
├── inference_and_visualization.py      # Inference & visualization
├── extract_data_from_postgres.py       # PostgreSQL data extraction
├── battery_data.csv                    # Your data
└── best_rul_model.pth                  # Trained model
```

## Extending the Model

To add features:
1. Update feature list in `BatteryGraphBuilder` (model_architecture.py)
2. Define edges (relationships)
3. Update model initialization with new feature count
4. Retrain

GAT automatically learns new feature interactions.

## Troubleshooting

**CUDA OOM**: Reduce batch size, window size, or forecast horizon
**Poor RUL**: Check data quality, ensure sufficient degradation data, adjust loss weights
**Overfitting**: Increase dropout, add L2 regularization, use more data
**Slow Training**: Use GPU, reduce model size

## License

MIT License

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Train
python model_train.py

# Predict (see inference_and_visualization.py for examples)
```

---

**Summary**: State-of-the-art GAT + BiLSTM model for battery RUL prediction. Currently uses 22 features, easily scalable to more. Interpretable, efficient, production-ready. 🔋⚡
