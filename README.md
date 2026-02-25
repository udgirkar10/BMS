# EV Battery RUL Prediction using GAT + BiLSTM

## Overview

This project implements a hybrid deep learning model combining Graph Attention Networks (GAT) and Bidirectional LSTM for predicting the Remaining Useful Life (RUL) of Electric Vehicle batteries using multi-variate time-series data.

## Architecture

**GAT + BiLSTM Hybrid Model**

```
Input Time-Series → GAT (Feature Relationships) → BiLSTM (Temporal Patterns) → Dual Output
                                                                                  ├─ Feature Forecasting
                                                                                  └─ RUL Prediction
```

### Key Components

1. **Graph Attention Network (GAT)**: Captures spatial relationships between battery features using attention mechanism
2. **Bidirectional LSTM**: Captures temporal patterns from both past and future contexts
3. **Dual Prediction Heads**: Forecasts future features and predicts RUL simultaneously
4. **Attention Mechanism**: Provides interpretability by showing feature importance

## Why GAT + BiLSTM?

### Graph Attention Network Advantages
- Learns feature importance automatically via attention
- Captures non-linear relationships between battery parameters
- Models physical/causal dependencies (temp → voltage, current → SoH)
- Interpretable through attention weights
- Scalable - easy to add new features

### Bidirectional LSTM Advantages
- Captures temporal dependencies in both directions
- Learns degradation patterns over time
- Better context understanding than unidirectional LSTM
- Robust to noise in time-series data
- Handles long-term dependencies

### Combined Power
- Spatial-Temporal Learning: GAT captures feature interactions, BiLSTM captures time evolution
- Multi-Scale Patterns: GAT learns instantaneous relationships, BiLSTM learns long-term trends
- Adaptive: Attention weights adjust based on battery state and usage patterns
- Predictive: Forecasts future degradation by understanding both structure and dynamics

## Current Features (22)

### Direct Health Indicators
- `Battery_SoH`: State of Health (1 = 100%)
- `Estimated_Battery_Capacity`: Current capacity in Ah
- `Charge_Discharge_Cycles`: Cumulative cycle count

### Electrical Parameters
- `Battery_Current`: Current flow in Amps
- `Battery_Voltage`: Voltage level in Volts
- `Pack_Current`: Total pack current
- `Pack_Voltage`: Total pack voltage
- `SoP`: State of Power (available power in Watts)

### Thermal Parameters
- `Battery_Temp`: Temperature in °C
- `LED_UnderTemp`: Low temperature warning flag
- `LED_OverTemp`: High temperature warning flag

### State Estimation
- `Estimated_SoC`: State of Charge (current charge level %)
- `Estimated_SoE`: State of Energy (available energy)
- `estimated_range`: Predicted driving range

### Operational Context
- `Vehicle_speed`: Current speed in km/h
- `Distance_Travelled`: Trip distance in km
- `Charging_Status`: Boolean indicating if charging
- `Time_To_Charge`: Remaining charge time

### Fault Indicators
- `LED_OverCurrent`: Overcurrent fault flag
- `LED_UnderVoltage`: Undervoltage fault flag
- `LED_OverVoltage`: Overvoltage fault flag

### Temporal
- `timestamp`: Time-series index

## Planned Feature Expansion

### Environmental Features
- Ambient_Temperature, Humidity, Altitude, Weather_Conditions, Road_Grade, Air_Pressure

### Operational & Usage Features
- Driving_Mode, Acceleration_Pattern, Braking_Pattern, Trip_Type, Idle_Time, Average_Speed, Speed_Variance, Daily_Usage_Hours, Charging_Frequency, Fast_Charge_Ratio

### Technical & Internal Battery Features
- Cell_Voltage_Min/Max/Delta, Cell_Temp_Min/Max/Delta, Internal_Resistance, Self_Discharge_Rate, Balancing_Activity, Coulombic_Efficiency, Energy_Efficiency, Power_Fade, Calendar_Age, Storage_Time

### Vehicle & Hardware Features
- Motor_Power_Demand, HVAC_Power_Consumption, Auxiliary_Load, Regenerative_Braking_Energy, Battery_Cooling_System_Status, Thermal_Management_Power, Vehicle_Weight, Tire_Pressure, Aerodynamic_Drag

## What is RUL?

**Remaining Useful Life (RUL)** is the estimated time or usage remaining before a battery reaches its end-of-life threshold.

### RUL Expressions

1. **Cycle-based**: Number of charge-discharge cycles until SoH drops to 80% (e.g., "500 cycles remaining")
2. **Time-based**: Calendar time remaining (e.g., "2.5 years remaining")
3. **Distance-based**: Kilometers the vehicle can travel (e.g., "50,000 km remaining")
4. **Energy throughput**: Total energy that can be cycled (e.g., "15,000 kWh remaining")

### End-of-Life Threshold
- Current Battery_SoH = 1.0 (100%)
- End-of-life threshold = 0.8 (80%)
- RUL = cycles/time/distance until SoH reaches 0.8

**Note**: RUL is NOT when battery is completely dead (0%), but when it's no longer useful for its intended purpose (typically 80% of original capacity for EVs).

## Model Architecture Details

### Architecture Flow

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
