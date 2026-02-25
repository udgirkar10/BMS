"""
Inference and visualization utilities for BiLSTM-GNN RUL model
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from bilstm_gnn_rul_model import BiLSTM_GNN_RUL, BatteryGraphBuilder


class RULPredictor:
    """
    Inference wrapper for trained RUL model
    """
    
    def __init__(self, model_path, device='cuda'):
        self.device = device
        
        # Load model
        self.model = BiLSTM_GNN_RUL(
            num_features=22,
            gnn_hidden_dim=64,
            lstm_hidden_dim=128,
            num_gnn_layers=2,
            num_lstm_layers=2,
            num_heads=4,
            dropout=0.2,
            forecast_horizon=50
        )
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.to(device)
        self.model.eval()
        
        # Build graph
        graph_builder = BatteryGraphBuilder()
        self.edge_index = graph_builder.build_edge_index().to(device)
        self.feature_names = graph_builder.features
    
    def predict(self, x):
        """
        Predict RUL and future features
        
        Args:
            x: Input tensor [batch_size, window_size, num_features] or numpy array
            
        Returns:
            rul: Predicted RUL values
            forecasted_features: Predicted future features
            attention_weights: Feature importance weights
        """
        if isinstance(x, np.ndarray):
            x = torch.FloatTensor(x)
        
        x = x.to(self.device)
        
        with torch.no_grad():
            forecasted_features, rul, attention_weights = self.model(x, self.edge_index)
        
        return (
            rul.cpu().numpy(),
            forecasted_features.cpu().numpy(),
            attention_weights.cpu().numpy()
        )
    
    def predict_single(self, x):
        """
        Predict for a single sample
        
        Args:
            x: Input [window_size, num_features]
            
        Returns:
            rul: Single RUL value
            forecasted_features: [forecast_horizon, num_features]
            attention_weights: [num_features]
        """
        if isinstance(x, np.ndarray):
            x = torch.FloatTensor(x)
        
        x = x.unsqueeze(0)  # Add batch dimension
        
        rul, forecasted_features, attention_weights = self.predict(x)
        
        return rul[0, 0], forecasted_features[0], attention_weights[0]


class RULVisualizer:
    """
    Visualization utilities for RUL predictions
    """
    
    def __init__(self, feature_names):
        self.feature_names = feature_names
        sns.set_style("whitegrid")
    
    def plot_rul_prediction(self, actual_rul, predicted_rul, save_path=None):
        """
        Plot actual vs predicted RUL
        """
        plt.figure(figsize=(10, 6))
        
        plt.scatter(actual_rul, predicted_rul, alpha=0.5)
        plt.plot([actual_rul.min(), actual_rul.max()], 
                [actual_rul.min(), actual_rul.max()], 
                'r--', label='Perfect prediction')
        
        plt.xlabel('Actual RUL (cycles)', fontsize=12)
        plt.ylabel('Predicted RUL (cycles)', fontsize=12)
        plt.title('RUL Prediction: Actual vs Predicted', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_feature_attention(self, attention_weights, top_k=10, save_path=None):
        """
        Plot feature importance based on attention weights
        """
        # Average attention weights if multiple samples
        if len(attention_weights.shape) > 1:
            attention_weights = attention_weights.mean(axis=0)
        
        # Get top-k features
        top_indices = np.argsort(attention_weights)[-top_k:]
        top_features = [self.feature_names[i] for i in top_indices]
        top_weights = attention_weights[top_indices]
        
        plt.figure(figsize=(10, 6))
        plt.barh(range(top_k), top_weights, color='steelblue')
        plt.yticks(range(top_k), top_features)
        plt.xlabel('Attention Weight', fontsize=12)
        plt.title(f'Top {top_k} Important Features for RUL Prediction', fontsize=14)
        plt.grid(True, alpha=0.3, axis='x')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_feature_forecast(self, historical, forecasted, feature_idx, 
                            feature_name=None, save_path=None):
        """
        Plot historical and forecasted values for a specific feature
        """
        if feature_name is None:
            feature_name = self.feature_names[feature_idx]
        
        hist_values = historical[:, feature_idx]
        forecast_values = forecasted[:, feature_idx]
        
        plt.figure(figsize=(12, 6))
        
        # Historical data
        hist_time = np.arange(len(hist_values))
        plt.plot(hist_time, hist_values, 'b-', label='Historical', linewidth=2)
        
        # Forecasted data
        forecast_time = np.arange(len(hist_values), len(hist_values) + len(forecast_values))
        plt.plot(forecast_time, forecast_values, 'r--', label='Forecasted', linewidth=2)
        
        # Vertical line at forecast start
        plt.axvline(x=len(hist_values), color='gray', linestyle=':', alpha=0.7)
        
        plt.xlabel('Time Step', fontsize=12)
        plt.ylabel(feature_name, fontsize=12)
        plt.title(f'Feature Forecast: {feature_name}', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_rul_over_time(self, rul_predictions, timestamps=None, save_path=None):
        """
        Plot RUL degradation over time
        """
        plt.figure(figsize=(12, 6))
        
        if timestamps is None:
            timestamps = np.arange(len(rul_predictions))
        
        plt.plot(timestamps, rul_predictions, 'b-', linewidth=2)
        plt.fill_between(timestamps, 0, rul_predictions, alpha=0.3)
        
        plt.xlabel('Time', fontsize=12)
        plt.ylabel('Remaining Useful Life (cycles)', fontsize=12)
        plt.title('RUL Degradation Over Time', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_health_indicators(self, data, save_path=None):
        """
        Plot key health indicators (SoH, Capacity, Cycles)
        """
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        # SoH
        axes[0].plot(data['Battery_SoH'], 'b-', linewidth=2)
        axes[0].axhline(y=0.8, color='r', linestyle='--', label='EOL Threshold (80%)')
        axes[0].set_ylabel('State of Health', fontsize=12)
        axes[0].set_title('Battery State of Health Over Time', fontsize=14)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Capacity
        axes[1].plot(data['Estimated_Battery_Capacity'], 'g-', linewidth=2)
        axes[1].set_ylabel('Capacity (Ah)', fontsize=12)
        axes[1].set_title('Battery Capacity Over Time', fontsize=14)
        axes[1].grid(True, alpha=0.3)
        
        # Cycles
        axes[2].plot(data['Charge_Discharge_Cycles'], 'orange', linewidth=2)
        axes[2].set_xlabel('Time Step', fontsize=12)
        axes[2].set_ylabel('Cycles', fontsize=12)
        axes[2].set_title('Charge-Discharge Cycles', fontsize=14)
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_correlation_heatmap(self, data, save_path=None):
        """
        Plot correlation heatmap of features
        """
        # Calculate correlation matrix
        corr_matrix = data.corr()
        
        plt.figure(figsize=(14, 12))
        sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
        plt.title('Feature Correlation Heatmap', fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


# Example usage
if __name__ == "__main__":
    import pandas as pd
    
    # Load trained model
    predictor = RULPredictor('best_rul_model.pth', device='cpu')
    
    # Load test data
    test_data = pd.read_csv('battery_data.csv')
    
    # Example: Predict for a single window
    window_size = 100
    sample_data = test_data.iloc[:window_size].drop('timestamp', axis=1).values
    
    rul, forecasted_features, attention_weights = predictor.predict_single(sample_data)
    
    print(f"Predicted RUL: {rul:.2f} cycles")
    print(f"Forecasted features shape: {forecasted_features.shape}")
    
    # Visualizations
    visualizer = RULVisualizer(predictor.feature_names)
    
    # Plot feature attention
    visualizer.plot_feature_attention(attention_weights, top_k=10)
    
    # Plot forecast for Battery_SoH
    soh_idx = predictor.feature_names.index('Battery_SoH')
    visualizer.plot_feature_forecast(sample_data, forecasted_features, 
                                    soh_idx, 'Battery_SoH')
    
    # Plot health indicators
    visualizer.plot_health_indicators(test_data)
