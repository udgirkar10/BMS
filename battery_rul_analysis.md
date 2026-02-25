# EV Battery RUL Prediction - Data Analysis

## Datapoint Categories

### 1. Direct Battery Health Indicators
- **Battery_SoH** (State of Health): Current health percentage (1 = 100%)
- **Estimated_Battery_Capacity**: Current capacity in Ah (94 Ah in example)
- **Charge_Discharge_Cycles**: Cumulative cycle count (0.0014 cycles in example)

### 2. Electrical Parameters
- **Battery_Current**: Current flow in Amps (5.71A)
- **Battery_Voltage**: Voltage level in Volts (4.37V)
- **Pack_Current**: Total pack current (200A)
- **Pack_Voltage**: Total pack voltage (410.5V)
- **SoP** (State of Power): Available power in Watts (-15794W)

### 3. Thermal Parameters
- **Battery_Temp**: Temperature in °C (29.9°C)
- **LED_UnderTemp**: Low temperature warning flag
- **LED_OverTemp**: High temperature warning flag

### 4. State Estimation
- **Estimated_SoC** (State of Charge): Current charge level (51.6%)
- **Estimated_SoE** (State of Energy): Available energy (28.3 units)
- **estimated_range**: Predicted driving range (0 km)

### 5. Operational Context
- **Vehicle_speed**: Current speed in km/h (75.3)
- **Distance_Travelled**: Trip distance in km (1.59)
- **Charging_Status**: Boolean indicating if charging (True)
- **Time_To_Charge**: Remaining charge time (0)

### 6. Fault Indicators
- **LED_OverCurrent**: Overcurrent fault flag
- **LED_UnderVoltage**: Undervoltage fault flag
- **LED_OverVoltage**: Overvoltage fault flag

### 7. Temporal
- **timestamp**: Time-series index

## Key Features for RUL Prediction

### Primary Degradation Indicators
1. **Battery_SoH** - Direct health metric
2. **Estimated_Battery_Capacity** - Capacity fade over time
3. **Charge_Discharge_Cycles** - Cycle aging

### Stress Factors (Accelerate Degradation)
1. **Battery_Temp** - Thermal stress
2. **Battery_Current** - C-rate stress
3. **Pack_Current** - Load stress
4. **Fault flags** - Abuse conditions

### Derived Features to Engineer
1. **C-rate**: Battery_Current / Estimated_Battery_Capacity
2. **Temperature deviation**: Difference from optimal (20-25°C)
3. **Voltage stress**: Deviation from nominal voltage
4. **Depth of Discharge (DoD)**: SoC range per cycle
5. **Capacity fade rate**: Change in capacity over time
6. **Cycle depth**: Average DoD per cycle
7. **Time at extreme temps**: Duration outside safe range
8. **Fast charging frequency**: High current charging events

## RUL Prediction Approach

### Target Variable
**RUL** = Cycles until SoH reaches End-of-Life threshold (typically 80% or 70%)

### Recommended Models
1. **LSTM/GRU Networks** - Capture temporal dependencies
2. **CNN-LSTM Hybrid** - Extract spatial-temporal patterns
3. **Transformer Models** - Attention-based sequence modeling
4. **Physics-Informed Neural Networks** - Incorporate degradation physics

### Data Preprocessing Needs
1. Handle missing values and outliers
2. Normalize/standardize features
3. Create sliding windows for sequence input
4. Calculate rolling statistics (mean, std, trend)
5. Extract cycle-level aggregations
6. Label data with RUL values

### Critical Considerations
- **Capacity fade** is non-linear (accelerates over time)
- **Temperature** has exponential impact on degradation
- **Cycle depth** matters more than cycle count
- **Calendar aging** occurs even without use
- **Knee point** detection (rapid degradation onset)
