import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from pathlib import Path
import argparse

from src.handler import RobustZScoreNorm
from src.trainner import DoubleAdaptFramework
from src.dataset import RollingTaskSampler

def load_and_split_data(data_path: str, train_ratio: float = 0.6, valid_ratio: float = 0.2):
    """Load and split data into train, validation and test sets"""
    # Load data
    df = pd.read_csv(data_path)
    
    # Sort by datetime
    df = df.sort_values('datetime')
    
    # Split data
    n_samples = len(df)
    train_size = int(n_samples * train_ratio)
    valid_size = int(n_samples * valid_ratio)
    
    train_df = df[:train_size]
    valid_df = df[train_size:train_size + valid_size]
    test_df = df[train_size + valid_size:]
    
    return train_df, valid_df, test_df

def preprocess_data(train_df, valid_df, test_df, scaler_path: str):
    """Preprocess data using RobustZScoreNorm"""
    # Initialize scaler
    scaler = RobustZScoreNorm(clip_outlier=True)
    
    # Fit scaler on training data
    X_train = train_df.drop(['datetime', 'label'], axis=1).values
    scaler.fit(X_train)
    
    # Save scaler
    scaler.save(scaler_path)
    
    # Transform all datasets
    X_train = scaler.transform(X_train)
    X_valid = scaler.transform(valid_df.drop(['datetime', 'label'], axis=1).values)
    X_test = scaler.transform(test_df.drop(['datetime', 'label'], axis=1).values)
    
    # Get labels
    y_train = train_df['label'].values
    y_valid = valid_df['label'].values
    y_test = test_df['label'].values
    
    return X_train, y_train, X_valid, y_valid, X_test, y_test

def create_tasks(X, y, dates, task_sampler, generate_tensor: bool = True):
    """Create tasks using RollingTaskSampler
    
    Args:
        X: Feature matrix
        y: Target values
        dates: Datetime values
        task_sampler: RollingTaskSampler instance
        generate_tensor: Whether to convert arrays to tensors
        
    Returns:
        List of tasks with support and query sets
    """
    # Create tasks using generate_tasks method
    tasks = task_sampler.generate_tasks(
        features=X,
        labels=y,
        to_tensor=generate_tensor
    )
    
    return tasks

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='outputs')
    parser.add_argument('--support_size', type=int, default=60)
    parser.add_argument('--query_size', type=int, default=20)
    parser.add_argument('--max_epochs', type=int, default=10)
    parser.add_argument('--generate_tensor', type=bool, default=True)
    # Add new arguments for framework parameters
    parser.add_argument('--num_head', type=int, default=8)
    parser.add_argument('--temperature', type=float, default=10.0)
    parser.add_argument('--lr_theta', type=float, default=0.001)
    parser.add_argument('--lr_da', type=float, default=0.01)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--sigma', type=float, default=1.0)
    parser.add_argument('--reg', type=float, default=0.5)
    parser.add_argument('--first_order', type=bool, default=True)
    parser.add_argument('--adapt_x', type=bool, default=True)
    parser.add_argument('--adapt_y', type=bool, default=True)
    parser.add_argument('--sequence_length', type=int, default=10, 
                       help='Length of sequence for LSTM')
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load and split data
    train_df, valid_df, test_df = load_and_split_data(args.data_path)
    
    # Initialize scaler
    scaler_path = output_dir / 'scaler.npy'
    scaler = RobustZScoreNorm(clip_outlier=True)
    
    # Fit scaler on training data only
    train_features = train_df.drop(['datetime', 'label'], axis=1).values
    scaler.fit(train_features)
    scaler.save(scaler_path)
    
    # Transform all datasets
    train_scaled = scaler.transform(train_df.drop(['datetime', 'label'], axis=1).values)
    valid_scaled = scaler.transform(valid_df.drop(['datetime', 'label'], axis=1).values)
    test_scaled = scaler.transform(test_df.drop(['datetime', 'label'], axis=1).values)
    
    # Create sequence data after scaling
    def create_sequence_data_from_scaled(scaled_features: np.ndarray, 
                                       labels: np.ndarray, 
                                       sequence_length: int):
        """Create sequence data from scaled features"""
        X, y = [], []
        for i in range(len(scaled_features) - sequence_length):
            X.append(scaled_features[i:(i + sequence_length)])
            y.append(labels[i + sequence_length])
        return np.array(X), np.array(y)
    
    # Create sequences for each dataset
    X_train, y_train = create_sequence_data_from_scaled(
        train_scaled,
        train_df['label'].values,
        args.sequence_length
    )
    X_valid, y_valid = create_sequence_data_from_scaled(
        valid_scaled,
        valid_df['label'].values,
        args.sequence_length
    )
    X_test, y_test = create_sequence_data_from_scaled(
        test_scaled,
        test_df['label'].values,
        args.sequence_length
    )
    
    print(f"Sequence data shapes:")
    print(f"X_train: {X_train.shape} (batch_size, sequence_length, feature_dim)")
    print(f"X_valid: {X_valid.shape}")
    print(f"X_test: {X_test.shape}")
    
    # Initialize task sampler
    task_sampler = RollingTaskSampler(
        interval=5,
        support_length=args.support_size,
        query_length=args.query_size
    )
    
    # Create tasks with sequence data
    train_tasks = create_tasks(
        X_train, y_train, 
        train_df['datetime'].values[args.sequence_length:], 
        task_sampler,
        generate_tensor=args.generate_tensor
    )
    valid_tasks = create_tasks(
        X_valid, y_valid, 
        valid_df['datetime'].values[args.sequence_length:], 
        task_sampler,
        generate_tensor=args.generate_tensor
    )
    test_tasks = create_tasks(
        X_test, y_test, 
        test_df['datetime'].values[args.sequence_length:], 
        task_sampler,
        generate_tensor=args.generate_tensor
    )
    
    # Define a custom RNN model that properly handles LSTM output
    class RNNModel(nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int = 64):
            super().__init__()
            self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
            self.linear = nn.Linear(hidden_dim, 1)
            
        def forward(self, x):
            # Add sequence dimension if input is 2D
            if len(x.shape) == 2:
                x = x.unsqueeze(1)  # [batch_size, 1, feature_dim]
            
            # LSTM returns (output, (h_n, c_n))
            lstm_out, _ = self.lstm(x)
            # Take the last output
            last_out = lstm_out[:, -1, :]
            # Project to prediction
            pred = self.linear(last_out)
            return pred.squeeze(-1)  # Ensure output is [batch_size]
    
    # Initialize model and framework
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    feature_dim = X_train.shape[2]  # Get feature dimension from sequence data
    
    # Use the custom RNN model instead of Sequential
    model = RNNModel(input_dim=feature_dim)
    
    # Initialize framework with all parameters
    framework = DoubleAdaptFramework(
        model=model,
        criterion=nn.MSELoss(),
        x_dim=feature_dim,
        num_head=args.num_head,
        temperature=args.temperature,
        lr_theta=args.lr_theta,
        lr_da=args.lr_da,
        early_stopping_patience=args.patience,
        device=device,
        sigma=args.sigma,
        reg=args.reg,
        first_order=args.first_order,
        adapt_x=args.adapt_x,
        adapt_y=args.adapt_y,
        is_rnn=True
    )
    
    # Offline training
    print("Starting offline training...")
    framework.offline_training(
        train_tasks=train_tasks,
        valid_tasks=valid_tasks,
        max_epochs=args.max_epochs,
        patience=args.patience
    )
    
    # Online training
    print("Starting online training...")
    metric = framework.online_training(
        valid_tasks=valid_tasks,
        test_tasks=test_tasks
    )
    print(f"Final test metric: {metric:.4f}")

if __name__ == '__main__':
    main()
