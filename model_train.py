"""
Training script for BiLSTM-GNN RUL prediction model
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
from model_architecture import BiLSTM_GNN_RUL, BatteryGraphBuilder


class BatteryDataset(Dataset):
    """
    Dataset for battery time-series data
    """
    
    def __init__(self, data, window_size=100, forecast_horizon=50, stride=1):
        """
        Args:
            data: DataFrame with battery features
            window_size: Number of past timesteps
            forecast_horizon: Number of future timesteps to predict
            stride: Step size for sliding window
        """
        self.data = data
        self.window_size = window_size
        self.forecast_horizon = forecast_horizon
        self.stride = stride
        
        # Feature columns (exclude timestamp)
        self.feature_cols = [col for col in data.columns if col != 'timestamp']
        
        # Calculate RUL for each sample
        self.rul_values = self._calculate_rul()
        
        # Create sliding windows
        self.samples = self._create_windows()
        
    def _calculate_rul(self):
        """
        Calculate RUL based on SoH degradation
        Assumes end-of-life at SoH = 0.8 (80%)
        """
        soh_values = self.data['Battery_SoH'].values
        cycles = self.data['Charge_Discharge_Cycles'].values
        
        rul_list = []
        eol_threshold = 0.8
        
        for i in range(len(soh_values)):
            current_soh = soh_values[i]
            current_cycle = cycles[i]
            
            if current_soh <= eol_threshold:
                rul = 0
            else:
                # Find when SoH reaches EOL threshold
                future_idx = np.where(soh_values[i:] <= eol_threshold)[0]
                
                if len(future_idx) > 0:
                    eol_idx = i + future_idx[0]
                    rul = cycles[eol_idx] - current_cycle
                else:
                    # Estimate based on degradation rate
                    if i > 0:
                        degradation_rate = (soh_values[0] - current_soh) / (current_cycle + 1e-6)
                        remaining_soh = current_soh - eol_threshold
                        rul = remaining_soh / (degradation_rate + 1e-6)
                    else:
                        rul = 1000  # Default high value
            
            rul_list.append(max(0, rul))
        
        return np.array(rul_list)
    
    def _create_windows(self):
        """
        Create sliding windows from time-series data
        """
        samples = []
        max_idx = len(self.data) - self.window_size - self.forecast_horizon
        
        for i in range(0, max_idx, self.stride):
            # Input window
            x_window = self.data[self.feature_cols].iloc[i:i+self.window_size].values
            
            # Check for missing or invalid data
            if np.isnan(x_window).any() or np.isinf(x_window).any():
                # Fill NaN and Inf with 0
                x_window = np.nan_to_num(x_window, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Future features for forecasting
            y_features = self.data[self.feature_cols].iloc[
                i+self.window_size:i+self.window_size+self.forecast_horizon
            ].values
            
            # Check for missing or invalid data in future features
            if np.isnan(y_features).any() or np.isinf(y_features).any():
                y_features = np.nan_to_num(y_features, nan=0.0, posinf=0.0, neginf=0.0)
            
            # RUL at the end of input window
            y_rul = self.rul_values[i+self.window_size-1]
            
            samples.append({
                'x': x_window,
                'y_features': y_features,
                'y_rul': y_rul
            })
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        return {
            'x': torch.FloatTensor(sample['x']),
            'y_features': torch.FloatTensor(sample['y_features']),
            'y_rul': torch.FloatTensor([sample['y_rul']])
        }


class RULTrainer:
    """
    Trainer for BiLSTM-GNN RUL model
    """
    
    def __init__(self, model, edge_index, device='cuda'):
        self.model = model.to(device)
        self.edge_index = edge_index.to(device)
        self.device = device
        
        # Loss functions
        self.forecast_criterion = nn.MSELoss()
        self.rul_criterion = nn.MSELoss()
        
        # Optimizer
        self.optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        # Loss weights
        self.alpha = 0.3  # Forecasting loss weight
        self.beta = 0.7   # RUL loss weight
        
    def train_epoch(self, train_loader):
        """
        Train for one epoch
        """
        self.model.train()
        total_loss = 0
        forecast_loss_sum = 0
        rul_loss_sum = 0
        
        for batch in train_loader:
            x = batch['x'].to(self.device)
            y_features = batch['y_features'].to(self.device)
            y_rul = batch['y_rul'].to(self.device)
            
            # Forward pass
            pred_features, pred_rul, attention = self.model(x, self.edge_index)
            
            # Calculate losses
            forecast_loss = self.forecast_criterion(pred_features, y_features)
            rul_loss = self.rul_criterion(pred_rul, y_rul)
            
            # Combined loss
            loss = self.alpha * forecast_loss + self.beta * rul_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            forecast_loss_sum += forecast_loss.item()
            rul_loss_sum += rul_loss.item()
        
        avg_loss = total_loss / len(train_loader)
        avg_forecast_loss = forecast_loss_sum / len(train_loader)
        avg_rul_loss = rul_loss_sum / len(train_loader)
        
        return avg_loss, avg_forecast_loss, avg_rul_loss
    
    def validate(self, val_loader):
        """
        Validate the model
        """
        self.model.eval()
        total_loss = 0
        forecast_loss_sum = 0
        rul_loss_sum = 0
        
        with torch.no_grad():
            for batch in val_loader:
                x = batch['x'].to(self.device)
                y_features = batch['y_features'].to(self.device)
                y_rul = batch['y_rul'].to(self.device)
                
                # Forward pass
                pred_features, pred_rul, attention = self.model(x, self.edge_index)
                
                # Calculate losses
                forecast_loss = self.forecast_criterion(pred_features, y_features)
                rul_loss = self.rul_criterion(pred_rul, y_rul)
                
                loss = self.alpha * forecast_loss + self.beta * rul_loss
                
                total_loss += loss.item()
                forecast_loss_sum += forecast_loss.item()
                rul_loss_sum += rul_loss.item()
        
        avg_loss = total_loss / len(val_loader)
        avg_forecast_loss = forecast_loss_sum / len(val_loader)
        avg_rul_loss = rul_loss_sum / len(val_loader)
        
        return avg_loss, avg_forecast_loss, avg_rul_loss
    
    def train(self, train_loader, val_loader, num_epochs=100, early_stopping_patience=10):
        """
        Full training loop
        """
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(num_epochs):
            # Train
            train_loss, train_forecast, train_rul = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_forecast, val_rul = self.validate(val_loader)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Print progress
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"  Train Loss: {train_loss:.4f} (Forecast: {train_forecast:.4f}, RUL: {train_rul:.4f})")
            print(f"  Val Loss: {val_loss:.4f} (Forecast: {val_forecast:.4f}, RUL: {val_rul:.4f})")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'best_rul_model.pth')
                print("  → Best model saved!")
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                    break
        
        # Load best model
        self.model.load_state_dict(torch.load('best_rul_model.pth'))
        print("\nTraining completed! Best model loaded.")


def prepare_data(data_path, train_size=0.8, val_size=0.1, test_size=0.1):
    """
    Load and prepare data for training
    
    Args:
        data_path: Path to CSV file with battery data
        train_size: Proportion of data for training (default: 0.8)
        val_size: Proportion of data for validation (default: 0.1)
        test_size: Proportion of data for testing (default: 0.1)
    """
    # Load data
    df = pd.read_csv(data_path)
    
    print(f"Loaded data: {len(df)} rows, {len(df.columns)} columns")
    
    # Check for required columns (20 features - removed Vehicle_speed and Distance_Travelled)
    required_cols = [
        'Battery_Current', 'Battery_Voltage', 'Battery_Temp', 'Battery_SoH',
        'Estimated_SoE', 'Estimated_Soc', 'Estimated_Battery_Capacity',
        'estimated_range', 'LED_OverCurrent', 'LED_UnderTemp', 'LED_OverTemp',
        'LED_UnderVoltage', 'LED_OverVoltage', 'Pack_Current',
        'Pack_Voltage', 'SoP', 'Charging_Status',
        'Charge_Discharge_Cycles', 'Time_To_Charge'
    ]
    
    missing_cols = set(required_cols) - set(df.columns)
    if missing_cols:
        print(f"Warning: Missing columns {missing_cols}, filling with default values")
        for col in missing_cols:
            if 'LED_' in col or col == 'Charging_Status':
                df[col] = False
            else:
                df[col] = 0.0
    
    # Normalize features (except timestamp)
    feature_cols = [col for col in df.columns if col != 'timestamp']
    
    # Handle any remaining NaN or Inf values
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    df[feature_cols] = df[feature_cols].fillna(0.0)
    
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    
    # Split data (time-series split - no shuffling)
    # 80% train, 10% validation, 10% test
    train_end = int(len(df) * train_size)
    val_end = int(len(df) * (train_size + val_size))
    
    train_data = df[:train_end]
    val_data = df[train_end:val_end]
    test_data = df[val_end:]
    
    print(f"Train: {len(train_data)} rows")
    print(f"Validation: {len(val_data)} rows")
    print(f"Test: {len(test_data)} rows")
    
    return train_data, val_data, test_data, scaler


# Main training script
if __name__ == "__main__":
    # Configuration
    DATA_PATH = "battery_data.csv"  # Your data file
    WINDOW_SIZE = 100
    FORECAST_HORIZON = 50
    BATCH_SIZE = 32
    NUM_EPOCHS = 100
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Using device: {DEVICE}")
    print(f"Model features: 20 (removed Vehicle_speed and Distance_Travelled)")
    
    # Prepare data
    print("Loading and preparing data...")
    train_data, val_data, test_data, scaler = prepare_data(DATA_PATH)
    
    # Create datasets
    train_dataset = BatteryDataset(train_data, WINDOW_SIZE, FORECAST_HORIZON)
    val_dataset = BatteryDataset(val_data, WINDOW_SIZE, FORECAST_HORIZON)
    test_dataset = BatteryDataset(test_data, WINDOW_SIZE, FORECAST_HORIZON)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Build graph
    graph_builder = BatteryGraphBuilder()
    edge_index = graph_builder.build_edge_index()
    
    # Initialize model
    model = BiLSTM_GNN_RUL(
        num_features=20,  # Reduced from 22
        gnn_hidden_dim=64,
        lstm_hidden_dim=128,
        num_gnn_layers=2,
        num_lstm_layers=2,
        num_heads=4,
        dropout=0.2,
        forecast_horizon=FORECAST_HORIZON
    )
    
    # Initialize trainer
    trainer = RULTrainer(model, edge_index, device=DEVICE)
    
    # Train model
    print("\nStarting training...")
    trainer.train(train_loader, val_loader, num_epochs=NUM_EPOCHS)
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_loss, test_forecast, test_rul = trainer.validate(test_loader)
    print(f"Test Loss: {test_loss:.4f} (Forecast: {test_forecast:.4f}, RUL: {test_rul:.4f})")
