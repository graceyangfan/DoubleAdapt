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
    tasks = []
    for support_idx, query_idx in task_sampler.sample(dates):
        if generate_tensor:
            task = {
                'support_x': X[support_idx],  # Already tensor
                'support_y': y[support_idx],  # Already tensor
                'query_x': X[query_idx],      # Already tensor
                'query_y': y[query_idx]       # Already tensor
            }
        else:
            task = {
                'support_x': torch.FloatTensor(X[support_idx]),
                'support_y': torch.FloatTensor(y[support_idx]),
                'query_x': torch.FloatTensor(X[query_idx]),
                'query_y': torch.FloatTensor(y[query_idx])
            }
        tasks.append(task)
    return tasks

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='outputs')
    parser.add_argument('--support_size', type=int, default=60)
    parser.add_argument('--query_size', type=int, default=20)
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--generate_tensor', type=bool, default=True)
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load and split data
    train_df, valid_df, test_df = load_and_split_data(args.data_path)
    
    # Preprocess data
    scaler_path = output_dir / 'scaler.npy'
    X_train, y_train, X_valid, y_valid, X_test, y_test = preprocess_data(
        train_df, valid_df, test_df, scaler_path
    )
    
    # Initialize task sampler with generate_tensor=True
    task_sampler = RollingTaskSampler(
        support_size=args.support_size,
        query_size=args.query_size,
        generate_tensor=args.generate_tensor
    )
    
    # Create tasks
    train_tasks = create_tasks(
        X_train, y_train, 
        train_df['datetime'].values, 
        task_sampler,
        generate_tensor=args.generate_tensor
    )
    valid_tasks = create_tasks(
        X_valid, y_valid, 
        valid_df['datetime'].values, 
        task_sampler,
        generate_tensor=args.generate_tensor
    )
    test_tasks = create_tasks(
        X_test, y_test, 
        test_df['datetime'].values, 
        task_sampler,
        generate_tensor=args.generate_tensor
    )
    
    # Initialize model and framework
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    feature_dim = X_train.shape[1]
    
    # Define a simple RNN model
    model = nn.Sequential(
        nn.LSTM(feature_dim, 64, batch_first=True),
        nn.Linear(64, 1)
    )
    
    framework = DoubleAdaptFramework(
        model=model,
        criterion=nn.MSELoss(),
        x_dim=feature_dim,
        device=device,
        is_rnn=True  # Using RNN model
    )
    
    # Offline training
    print("Starting offline training...")
    framework.offline_training(
        train_tasks=train_tasks,
        valid_tasks=valid_tasks,
        max_epochs=args.max_epochs
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
