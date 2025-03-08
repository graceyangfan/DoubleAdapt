import polars as pl
import pandas as pd 
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import argparse
import os
from typing import Tuple, List, Dict, Optional
from sklearn.model_selection import train_test_split

# Import local modules
from double_adapt.trainner import DoubleAdaptFramework
from double_adapt.dataset import RollingTaskSampler
from double_adapt.handler import RobustZScoreNorm
from double_adapt.model import DoubleAdapt
from tlob.tlob import TLOB
from tlob.mlplob import MLPLOB


def calculate_trend_labels(
    df: pl.DataFrame,
    price_column: str,
    k: int=10,
    h: int=5,
):
    """
    calculate trend labels based on smoothed past and future prices
    
    df: pl.DataFrame
        DataFrame containing price data
    price_column: str
        Name of the price column
    k: int
        Smoothing window size       
    h: int
        Prediction horizon
    """
    if not isinstance(df, pl.DataFrame):
        df = pl.from_pandas(df)
    # create dataframe index 
    if "index" in df.columns:
        df = df.drop(["index"])
    df = df.with_row_index("index")
    df = df.with_columns(
        [
            pl.col("index").cast(pl.Int64),
        ]
    )
    
    #compute roling 
    df = df.with_columns(
        [
            pl.col(price_column).shift(-h).rolling_mean(window_size=k).alias("w_plus"),
        ]
    )
    df = df.with_columns(
        [
            ((pl.col("w_plus") - pl.col(price_column)) / pl.col(price_column)).alias("l_value"),
        ]
    )
    # drop intermediate columns
    df = df.drop(["w_plus"])
    return df


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from CSV or Parquet file
    
    Args:
        file_path: Path to the data file
        
    Returns:
        Polars DataFrame
    """
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path, index_col = 0)
    elif file_path.endswith('.parquet'):
        return pd.read_parquet(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")


def prepare_features_and_labels(
    df: pl.DataFrame,
    price_column: str = 'mid_price',
    features_columns: Optional[List[str]] = None,
    label_horizon: int = 5,
    smooth_window: int = 10
) -> Tuple[pl.DataFrame, np.ndarray, np.ndarray]:
    """
    Prepare features and trend labels from the input dataframe
    
    Args:
        df: Input DataFrame
        price_column: Column name for price
        features_columns: List of feature column names to use (if None, use all except index)
        label_horizon: Prediction horizon for trend labels
        smooth_window: Smoothing window size for trend calculation
        
    Returns:
        Tuple of (processed_df, features_array, labels_array)
    """
    # Calculate trend labels using the function
    df = calculate_trend_labels(
        df, 
        price_column=price_column, 
        k=smooth_window,
        h=label_horizon
    )
    
    # Drop rows with NaN values in the l_value column
    df = df.drop_nulls(subset=["l_value"])
    
    # Select features
    if features_columns is None:
        # Use all columns except index and label columns
        features_columns = [col for col in df.columns 
                           if col not in ["index", "l_value", "w_plus"]]
    # Extract features and labels as numpy arrays
    features = df.select(features_columns).to_numpy()
    labels = df.select("l_value").to_numpy().flatten()
    
    return df, features, labels


def split_dataset(
    features: np.ndarray, 
    labels: np.ndarray,
    train_ratio: float = 0.6,
    valid_ratio: float = 0.2,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split dataset into train, validation, and test sets
    
    Args:
        features: Feature matrix
        labels: Label vector
        train_ratio: Proportion of data for training
        valid_ratio: Proportion of data for validation
        random_state: Random state for reproducibility
        
    Returns:
        Tuple of (X_train, y_train, X_valid, y_valid, X_test, y_test)
    """
    # First split: train and temp (valid + test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        features, labels, 
        test_size=(1-train_ratio),
        shuffle=False  # No shuffle for time series data
    )
    
    # Calculate the ratio of validation set in the temp set
    valid_ratio_adjusted = valid_ratio / (1 - train_ratio)
    
    # Second split: valid and test from temp
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_temp, y_temp, 
        test_size=(1-valid_ratio_adjusted),
        shuffle=False  # No shuffle for time series data
    )
    
    return X_train, y_train, X_valid, y_valid, X_test, y_test


def create_sequence_data(
    features: np.ndarray,
    labels: np.ndarray,
    sequence_length: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequence data from features and labels
    
    Args:
        features: Feature matrix [n_samples, n_features]
        labels: Label vector [n_samples]
        sequence_length: Sequence length
        
    Returns:
        Tuple of (X_seq, y_seq) where X_seq has shape [n_samples-sequence_length, sequence_length, n_features]
        and y_seq has shape [n_samples-sequence_length]
    """
    X_seq, y_seq = [], []
    
    for i in range(len(features) - sequence_length):
        X_seq.append(features[i:i+sequence_length])
        y_seq.append(labels[i+sequence_length])
        
    return np.array(X_seq), np.array(y_seq)


def setup_mlplob_model(
    sequence_length: int,
    feature_dim: int,
    hidden_dim: int = 64,
    num_layers: int = 2,
    output_dim: int = 1,
    num_head: int = 8,
    temperature: float = 10.0,
    device: torch.device = None
):
    """
    Setup a DoubleAdapt model with MLPLOB base model
    
    Args:
        sequence_length: Length of input sequence
        feature_dim: Number of features
        hidden_dim: Hidden dimension size
        num_layers: Number of layers
        output_dim: Output dimension (1 for regression, >1 for classification)
        num_head: Number of attention heads for DoubleAdapt
        temperature: Temperature parameter for attention mechanism
        device: Device to use
        
    Returns:
        Initialized DoubleAdapt model with MLPLOB backbone
    """    
    # Use default device if not specified
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    # Create MLPLOB model
    return MLPLOB(
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        seq_size=sequence_length,
        num_features=feature_dim,
        output_dim=output_dim,
        device=device
    )
    


def setup_tlob_model(
    sequence_length: int,
    feature_dim: int,
    hidden_dim: int = 64,
    num_layers: int = 2,
    num_heads: int = 4,
    output_dim: int = 1,
    is_sin_emb: bool = True,
    da_num_head: int = 8,
    temperature: float = 10.0,
    device: torch.device = None
):
    """
    Setup a DoubleAdapt model with TLOB base model
    
    Args:
        sequence_length: Length of input sequence
        feature_dim: Number of features
        hidden_dim: Hidden dimension size
        num_layers: Number of layers
        num_heads: Number of attention heads in TLOB
        output_dim: Output dimension (1 for regression, >1 for classification)
        is_sin_emb: Whether to use sinusoidal embeddings
        da_num_head: Number of attention heads for DoubleAdapt
        temperature: Temperature parameter for attention mechanism
        device: Device to use
        
    Returns:
        Initialized DoubleAdapt model with TLOB backbone
    """    
    # Use default device if not specified
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    # Create TLOB model
    return TLOB(
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        seq_size=sequence_length,
        num_features=feature_dim,
        num_heads=num_heads,
        output_dim=output_dim,
        is_sin_emb=is_sin_emb,
        device=device
    )
    


def train_double_adapt(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model: DoubleAdapt,
    sequence_length: int = 10,
    support_length: int = 60,
    query_length: int = 20,
    interval: int = 5,
    num_head: int = 8,
    max_epochs: int = 10,
    patience: int = 5,
    lr_theta: float = 0.001,
    lr_da: float = 0.01,
    output_dir: str = "outputs"
) -> Dict:
    """
    Train DoubleAdapt framework with MLPLOB/TLOB model
    
    Args:
        X_train, y_train: Training data
        X_valid, y_valid: Validation data
        X_test, y_test: Test data
        model: DoubeAdapt model 
        sequence_length: Length of input sequence
        support_length: Length of support set
        query_length: Length of query set
        interval: Rolling interval for task sampling
        num_head: Number of attention heads for DoubleAdapt
        max_epochs: Maximum number of epochs
        patience: Early stopping patience
        lr_theta: Learning rate for forecast model
        lr_da: Learning rate for data adapter
        output_dir: Directory to save outputs
        
    Returns:
        Dictionary with training results
    """
    # Create sequence data
    X_train_seq, y_train_seq = create_sequence_data(X_train, y_train, sequence_length)
    X_valid_seq, y_valid_seq = create_sequence_data(X_valid, y_valid, sequence_length)
    X_test_seq, y_test_seq = create_sequence_data(X_test, y_test, sequence_length)
    
    print(f"Sequence data shapes:")
    print(f"X_train_seq: {X_train_seq.shape}")
    print(f"X_valid_seq: {X_valid_seq.shape}")
    print(f"X_test_seq: {X_test_seq.shape}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize task sampler
    task_sampler = RollingTaskSampler(
        interval=interval,
        support_length=support_length,
        query_length=query_length
    )
    
    # Create tasks
    train_tasks = task_sampler.generate_tasks(
        features=X_train_seq,
        labels=y_train_seq,
        to_tensor=True
    )
    
    valid_tasks = task_sampler.generate_tasks(
        features=X_valid_seq,
        labels=y_valid_seq,
        to_tensor=True
    )
    
    test_tasks = task_sampler.generate_tasks(
        features=X_test_seq,
        labels=y_test_seq,
        to_tensor=True
    )
    
    print(f"Created {len(train_tasks)} train tasks, {len(valid_tasks)} valid tasks, {len(test_tasks)} test tasks")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    #MLSE loss for regression 
    criterion = nn.MSELoss()
    # Initialize DoubleAdapt framework
    framework = DoubleAdaptFramework(
        model=model,
        criterion=criterion,
        x_dim=X_train_seq.shape[2],  # Feature dimension
        num_head=num_head,
        temperature=10.0,
        lr_theta=lr_theta,
        lr_da=lr_da,
        early_stopping_patience=patience,
        device=device,
        adapt_x=True,
        adapt_y=True,
        is_rnn=False  # MLPLOB is not RNN
    )
    
    # Offline training
    print("Starting offline training...")
    framework.offline_training(
        train_tasks=train_tasks,
        valid_tasks=valid_tasks,
        max_epochs=max_epochs,
        patience=patience
    )
    
    # Online training
    print("Starting online training...")
    metric = framework.online_training(
        valid_tasks=valid_tasks,
        test_tasks=test_tasks
    )
    
    print(f"Final test metric: {metric:.4f}")
    
    # Save model
    torch.save(framework.meta_model.model.state_dict(), f"{output_dir}/model.pt")
    torch.save(framework.meta_model.feature_adapter.state_dict(), f"{output_dir}/feature_adapter.pt")
    torch.save(framework.meta_model.label_adapter.state_dict(), f"{output_dir}/label_adapter.pt")
    
    return {
        "metric": metric,
        "model_path": f"{output_dir}/model.pt",
        "feature_adapter_path": f"{output_dir}/feature_adapter.pt",
        "label_adapter_path": f"{output_dir}/label_adapter.pt"
    }


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Train DoubleAdapt with MLPLOB/TLOB model")
    parser.add_argument('--data_path', type=str, required=True, help="Path to data file (CSV or Parquet)")
    parser.add_argument('--price_column', type=str, default='mid_price', help="Column name for price")
    parser.add_argument('--feature_columns', type=str, nargs='+', default=None, help="Feature column names to use")
    parser.add_argument('--label_horizon', type=int, default=30, help="Prediction horizon for trend labels")
    parser.add_argument('--smooth_window', type=int, default=5, help="Smoothing window size")
    parser.add_argument('--sequence_length', type=int, default=30, help="Sequence length for models")
    parser.add_argument('--model_type', type=str, choices=['mlplob', 'tlob'], default='tlob', help="Model type")
    parser.add_argument('--hidden_dim', type=int, default=64, help="Hidden dimension size")
    parser.add_argument('--num_layers', type=int, default=2, help="Number of layers")
    parser.add_argument('--num_heads', type=int, default=4, help="Number of attention heads (for TLOB)")
    parser.add_argument('--support_length', type=int, default=15*60, help="Length of support set")
    parser.add_argument('--query_length', type=int, default=5*60, help="Length of query set")
    parser.add_argument('--interval', type=int, default=5, help="Rolling interval for task sampling")
    parser.add_argument('--num_head', type=int, default=8, help="Number of attention heads for DoubleAdapt")
    parser.add_argument('--max_epochs', type=int, default=10, help="Maximum number of epochs")
    parser.add_argument('--patience', type=int, default=5, help="Early stopping patience")
    parser.add_argument('--lr_theta', type=float, default=0.001, help="Learning rate for forecast model")
    parser.add_argument('--lr_da', type=float, default=0.01, help="Learning rate for data adapter")
    parser.add_argument('--output_dir', type=str, default="outputs", help="Directory to save outputs")
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from {args.data_path}...")
    df = load_data(args.data_path)
    df = df.dropna() 
    feature_columns = [] 
    with open("features.txt", "r") as f:
        for line in f:
            line = line.strip().strip("'")  
            if line.endswith(','):
                line = line[:-2] 
            feature_columns.append(line)
    print(f"Loaded data with shape: {df.shape}")
    
    # Prepare features and labels
    print("Preparing features and labels...")
    df, features, labels = prepare_features_and_labels(
        df,
        price_column=args.price_column,
        features_columns=feature_columns,
        label_horizon=args.label_horizon,
        smooth_window=args.smooth_window
    )
    print(f"Prepared features with shape: {features.shape}")
    print(f"Prepared labels with shape: {labels.shape}")
    
    # Split dataset
    print("Splitting dataset...")
    X_train, y_train, X_valid, y_valid, X_test, y_test = split_dataset(
        features, labels, train_ratio=0.6, valid_ratio=0.2
    )
    print(f"Train set: {X_train.shape}, Valid set: {X_valid.shape}, Test set: {X_test.shape}")
    
    # Normalize features
    print("Normalizing features...")
    scaler = RobustZScoreNorm()
    X_train = scaler.fit_transform(X_train)
    X_valid = scaler.transform(X_valid)
    X_test = scaler.transform(X_test)
    
    # Save scaler
    os.makedirs(args.output_dir, exist_ok=True)
    scaler.save(f"{args.output_dir}/scaler.npy")
    
    # Setup model
    print(f"Setting up {args.model_type} model...")
    feature_dim = X_train.shape[1]
    
    if args.model_type == 'mlplob':
        model = setup_mlplob_model(
            sequence_length=args.sequence_length,
            feature_dim=feature_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            output_dim=1
        )
    else:  # tlob
        model = setup_tlob_model(
            sequence_length=args.sequence_length,
            feature_dim=feature_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            output_dim=1,
            is_sin_emb=True,
        )
    
    # Train DoubleAdapt
    print("Training DoubleAdapt framework...")
    results = train_double_adapt(
        X_train=X_train,
        y_train=y_train,
        X_valid=X_valid,
        y_valid=y_valid,
        X_test=X_test,
        y_test=y_test,
        model=model,
        sequence_length=args.sequence_length,
        support_length=args.support_length,
        query_length=args.query_length,
        interval=args.interval,
        num_head=args.num_head,
        max_epochs=args.max_epochs,
        patience=args.patience,
        lr_theta=args.lr_theta,
        lr_da=args.lr_da,
        output_dir=args.output_dir
    )
    
    print(f"Training completed. Results saved to {args.output_dir}")
    print(f"Final test metric: {results['metric']:.4f}")


if __name__ == "__main__":
    main()



