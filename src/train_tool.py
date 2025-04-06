import polars as pl
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import argparse
import os
import pickle
from typing import Tuple, List, Dict, Optional, Any, Union # Added Union
from sklearn.model_selection import train_test_split
import shutil # Import shutil for directory operations
import glob # Import glob for finding files
import random # Import random for shuffling
from tqdm import tqdm # Import tqdm for progress bars
import pyarrow.parquet as pq # Import pyarrow
import pyarrow as pa # Import pyarrow

# Import local modules
from double_adapt.trainer import DoubleAdaptFramework
# Use TimeSeriesDataset and create_dataloader_from_task from dataset.py
from double_adapt.dataset import RollingTaskSampler, TimeSeriesDataset, create_dataloader_from_task 
from double_adapt.handler import RobustZScoreNorm
from double_adapt.model import DoubleAdapt
from double_adapt.high_pass_3_filter import apply_high_pass_filter
from models.tlob import TLOB
from models.mlplob import MLPLOB
from models.gru import GRULOB


def calculate_trend_labels(
    df: pl.DataFrame,
    price_column: str,
    filter_length: float = 10.0,
    shift_length: int = 10,
) -> pl.DataFrame: # Return Polars DataFrame
    """
    Calculate trend labels using high-pass filter and future price comparison.

    Args:
        df: pl.DataFrame
            DataFrame containing price data.
        price_column: str
            Name of the price column.
        filter_length: float
            Length parameter for high-pass filter (default: 10.0).
        shift_length: int
            Length for future price shift (default: 10).

    Returns:
        pl.DataFrame with added columns:
        - filtered_price: Price after high-pass filtering.
        - future_filtered_price: Future filtered price.
        - l_value: Label value ((future_price - current_price)/current_price).
    """
    # Ensure input is Polars DataFrame
    if isinstance(df, pd.DataFrame):
        df = pl.from_pandas(df)
    elif not isinstance(df, pl.DataFrame):
        raise TypeError("Input df must be a Polars or Pandas DataFrame.")


    # Create index if needed
    if "index" not in df.columns:
        df = df.with_row_index("index")
    df = df.with_columns([pl.col("index").cast(pl.Int64)])

    # Apply high-pass filter to current prices
    df = apply_high_pass_filter(
        df=df,
        column_name=price_column,
        output_name="filtered_price",
        filter_length=filter_length
    )

    # Calculate future filtered price
    df = df.with_columns([
        pl.col("filtered_price").shift(-shift_length).alias("future_filtered_price")
    ])

    # Calculate trend labels
    df = df.with_columns([
        ((pl.col("future_filtered_price") - pl.col(price_column)) / pl.col(price_column)).alias("l_value")
    ])

    return df


def load_data(file_path: str) -> pl.DataFrame: # Return Polars DataFrame
    """
    Load data from CSV or Parquet file using Polars.

    Args:
        file_path: Path to the data file.

    Returns:
        Polars DataFrame.
    """
    print(f"Attempting to load data from: {file_path}")
    if file_path.endswith('.csv'):
        # Use scan_csv for potentially large files, then collect
        try:
            df = pl.scan_csv(file_path).collect()
            print(f"Successfully loaded CSV: {file_path}")
            return df
        except Exception as e:
            print(f"Error loading CSV {file_path}: {e}")
            raise
    elif file_path.endswith('.parquet'):
        # Use scan_parquet for potentially large files, then collect
        try:
            df = pl.scan_parquet(file_path).collect()
            print(f"Successfully loaded Parquet: {file_path}")
            return df
        except Exception as e:
            print(f"Error loading Parquet {file_path}: {e}")
            raise
    else:
        raise ValueError(f"Unsupported file format: {file_path}")


def normalize_labels(labels: np.ndarray, num_std: float = 5.0) -> np.ndarray:
    """
    Normalize labels to [-1, 1] range while preserving directionality.

    Args:
        labels: Input labels.
        num_std: Number of standard deviations for clipping (default: 5.0).

    Returns:
        Normalized labels.
    """
    # Calculate mean and std
    mean = np.mean(labels)
    std = np.std(labels)

    # Clip values to mean ± num_std * std
    # Handle potential NaN/Inf in std
    if std > 1e-9:
        lower_bound = mean - num_std * std
        upper_bound = mean + num_std * std
        clipped = np.clip(labels, lower_bound, upper_bound)
    else:
        # If std is zero or very small, just use the mean (or handle as constant)
        clipped = np.clip(labels, mean, mean) # Or simply labels if no clipping needed


    # Split positive and negative values relative to the mean
    pos_mask = clipped > mean
    neg_mask = clipped < mean
    zero_mask = clipped == mean # Handle values exactly at the mean

    # Initialize normalized array
    normalized = np.zeros_like(clipped, dtype=np.float32) # Use float32

    # Normalize positive and negative deviations separately
    if np.any(pos_mask):
        pos_dev = clipped[pos_mask] - mean
        pos_max_dev = np.max(pos_dev)
        if pos_max_dev > 1e-9: # Avoid division by zero
             normalized[pos_mask] = pos_dev / pos_max_dev # Scale to [0, 1]

    if np.any(neg_mask):
        neg_dev = clipped[neg_mask] - mean # These are negative
        neg_min_dev = np.min(neg_dev) # Most negative deviation
        if abs(neg_min_dev) > 1e-9: # Avoid division by zero
            normalized[neg_mask] = neg_dev / abs(neg_min_dev) # Scale to [-1, 0]

    # Values at the mean remain 0
    normalized[zero_mask] = 0.0

    return normalized


def prepare_features_and_labels(
    df: pl.DataFrame,
    price_column: str = 'mid_price',
    features_columns: Optional[List[str]] = None,
    filter_length: float = 10.0,
    shift_length: int = 10
) -> Tuple[pl.DataFrame, List[str], str]: # Return DataFrame, feature list, label name
    """
    Prepare features and trend labels from the input dataframe.

    Args:
        df: Input Polars DataFrame.
        price_column: Column name for price.
        features_columns: List of feature column names to use (if None, use all except index and label-related).
        filter_length: Length parameter for high-pass filter.
        shift_length: Length for future price shift.

    Returns:
        Tuple of (processed_df, final_feature_columns, label_column_name).
    """
    print("Calculating trend labels...")
    df = df.with_columns(
        [pl.col(col_name).cast(pl.Float64) for col_name in features_columns]
    )
    df = df.with_columns(
        pl.col(price_column).cast(pl.Float64)
    )
    # Calculate trend labels using the enhanced function
    df = calculate_trend_labels(
        df,
        price_column=price_column,
        filter_length=filter_length,
        shift_length=shift_length,
    )
    print("Trend labels calculated.")

    # Drop rows with NaN values potentially created by filters/shifts
    initial_rows = len(df)
    df = df.drop_nulls() # Drop rows with nulls in any column after calculations
    print(f"Dropped {initial_rows - len(df)} rows with nulls after label calculation.")

    if len(df) == 0:
        raise ValueError("DataFrame is empty after dropping nulls. Check filter/shift parameters or input data.")

    # Select features
    label_column_name = "normalized_l_value"
    # Ensure intermediate columns used in calculation are excluded from features
    intermediate_cols = ["index", "l_value", "filtered_price", "future_filtered_price", price_column]
    # Also exclude the final label column itself if it exists already (shouldn't, but safety check)
    if label_column_name in df.columns:
         intermediate_cols.append(label_column_name)

    if features_columns is None:
        # Use all columns except intermediate ones
        final_feature_columns = [col for col in df.columns if col not in intermediate_cols]
        print(f"Auto-detected {len(final_feature_columns)} feature columns.")
    else:
        # Ensure provided feature columns exist and don't overlap with intermediate ones
        final_feature_columns = [col for col in features_columns if col in df.columns and col not in intermediate_cols]
        if len(final_feature_columns) != len(features_columns):
             print("Warning: Some specified feature columns were not found or overlapped with label/intermediate columns.")
        print(f"Using specified {len(final_feature_columns)} feature columns.")

    if not final_feature_columns:
        raise ValueError("No feature columns selected. Check input features list or data columns.")

    # Extract raw labels and normalize them
    print("Normalizing labels...")
    raw_labels = df.select("l_value").to_numpy().flatten()
    normalized_labels = normalize_labels(raw_labels, num_std=5.0)
    print("Labels normalized.")

    # Add normalized labels to DataFrame and select final columns
    # Ensure the label column name is unique before adding
    if label_column_name in final_feature_columns:
        raise ValueError(f"Label column name '{label_column_name}' conflicts with a feature column.")
        
    df = df.with_columns(
        pl.Series(label_column_name, normalized_labels)
    ).select(final_feature_columns + [label_column_name]) # Keep only final features and normalized label

    print(f"Prepared data with {len(final_feature_columns)} features and label '{label_column_name}'.")
    return df, final_feature_columns, label_column_name


# --- Helper Function: Chunk Parquet Files ---
def chunk_parquet_files(
    input_file: str, # Takes a single input file path
    chunk_size: int,
    output_dir: str,
    prefix: str = "chunk" # Added prefix for output filenames
) -> List[str]:
    """
    Splits a single large Parquet file into smaller chunks.

    Args:
        input_file: Path to the original Parquet file to chunk.
        chunk_size: Desired number of rows per chunk file.
        output_dir: Directory to save the chunked files.
        prefix: Prefix for the output chunk filenames.

    Returns:
        List of paths to the newly created chunked Parquet files, sorted alphabetically.
    """
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file for chunking not found: {input_file}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    chunked_files = []
    chunk_counter = 0
    
    print(f"Chunking file '{os.path.basename(input_file)}' into chunks of size {chunk_size}...")
    try:
        # Use iter_batches for memory efficiency with large files
        reader = pq.ParquetFile(input_file)
        num_rows = reader.metadata.num_rows
        schema = reader.schema.to_arrow_schema() # Get schema once

        current_row = 0
        pbar = tqdm(total=num_rows, desc="Chunking Progress")
        while current_row < num_rows:
            # Read a chunk of rows
            # Note: Reading exact chunk_size might require reading batches and slicing
            # Simpler approach: read slightly more if needed, then slice
            # More robust: iterate through row groups if chunk_size aligns well
            
            # Read the table slice directly (often efficient enough)
            table_slice = pq.read_table(input_file, columns=None).slice(current_row, chunk_size)

            if table_slice.num_rows > 0:
                chunk_filename = f"{prefix}_{chunk_counter:06d}.parquet"
                chunk_output_path = os.path.join(output_dir, chunk_filename)
                pq.write_table(table_slice, chunk_output_path)
                chunked_files.append(chunk_output_path)
                chunk_counter += 1
                pbar.update(table_slice.num_rows)
                current_row += table_slice.num_rows
            else:
                # Should not happen if current_row < num_rows, but safety break
                break
        pbar.close()

    except Exception as e:
        print(f"Error processing file {input_file} during chunking: {e}")
        raise # Re-raise the exception

    print(f"Created {len(chunked_files)} chunked files in {output_dir}")
    # Sort files to ensure consistent order for splitting and task generation
    chunked_files.sort()
    return chunked_files

# --- Helper Function: Split File List ---
def split_files(
    file_paths: List[str],
    train_ratio: float = 0.6,
    valid_ratio: float = 0.2,
    shuffle: bool = True # Add shuffle option
) -> Tuple[List[str], List[str], List[str]]:
    """
    Splits a list of file paths into train, validation, and test sets.
    Assumes files are sorted chronologically if shuffle=False.

    Args:
        file_paths: List of file paths to split.
        train_ratio: Proportion of files for the training set.
        valid_ratio: Proportion of files for the validation set.
        shuffle: Whether to shuffle the files before splitting.

    Returns:
        A tuple containing lists of file paths for train, validation, and test sets.
    """
    if not (0 < train_ratio < 1 and 0 < valid_ratio < 1 and train_ratio + valid_ratio <= 1): # Allow sum to be 1
         raise ValueError("Ratios must be between 0 and 1, and train + valid must be <= 1.")

    n_total = len(file_paths)
    if n_total == 0:
        return [], [], []

    if shuffle:
        print("Shuffling files before splitting...")
        files_to_split = random.sample(file_paths, n_total)
    else:
        print("Splitting files sequentially (assuming chronological order)...")
        files_to_split = file_paths # Use original order

    n_train = int(n_total * train_ratio)
    n_valid = int(n_total * valid_ratio)
    # Ensure test set gets the remainder
    n_test = n_total - n_train - n_valid

    train_files = files_to_split[:n_train]
    valid_files = files_to_split[n_train : n_train + n_valid]
    test_files = files_to_split[n_train + n_valid :] # Take the rest

    print(f"Split files: Train={len(train_files)}, Valid={len(valid_files)}, Test={len(test_files)}")
    return train_files, valid_files, test_files

# --- Helper Function: Get File Metadata ---
def get_file_infos(file_paths: List[str]) -> List[Dict[str, Union[str, int]]]:
    """
    Gets metadata (filename, length) for a list of Parquet files.
    Ensures files are sorted by filename before returning.

    Args:
        file_paths: List of paths to Parquet files.

    Returns:
        List of dictionaries, each containing 'filename' and 'length', sorted by filename.
    """
    if not file_paths:
        return []
        
    file_infos = []
    print(f"Getting metadata for {len(file_paths)} files...")
    # Sort paths first to ensure consistent order in file_infos
    sorted_paths = sorted(file_paths)

    for path in tqdm(sorted_paths, desc="Reading Metadata"):
        try:
            if not os.path.exists(path):
                 print(f"Warning: File not found during metadata read: {path}. Skipping.")
                 continue
            pf = pq.ParquetFile(path)
            file_infos.append({'filename': path, 'length': pf.metadata.num_rows})
        except Exception as e:
            print(f"Warning: Could not read metadata for {path}: {e}")

    # The list is already sorted because we iterated over sorted_paths
    return file_infos


def setup_mlplob_model(
    sequence_length: int,
    feature_dim: int,
    hidden_dim: int = 64,
    num_layers: int = 2,
    output_dim: int = 1,
    device: torch.device = None
):
    """ Setup MLPLOB base model. """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    num_heads: int = 4, # TLOB heads
    output_dim: int = 1,
    is_sin_emb: bool = True,
    # Removed DA specific args
    device: torch.device = None
):
    """ Setup TLOB base model. """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

def setup_gru_model(
    sequence_length: int,
    feature_dim: int,
    hidden_dim: int = 64,
    num_layers: int = 2,
    output_dim: int = 1,
    dropout: float = 0.1,
    device: torch.device = None
):
    """ Setup GRULOB model. """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return GRULOB(
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        seq_size=sequence_length,
        num_features=feature_dim,
        output_dim=output_dim,
        device=device,
        dropout=dropout
    )


# Removed train_double_adapt function as its logic is now integrated into main()


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Train DoubleAdapt with MLPLOB/TLOB/GRU model")
    parser.add_argument('--data_path', type=str, required=True, help="Path to the single input data file (CSV or Parquet)")
    parser.add_argument('--price_column', type=str, default='mid_price', help="Column name for price")
    parser.add_argument('--features_file', type=str, default='features.txt', help="Path to file containing feature column names (one per line)")
    parser.add_argument('--filter_length', type=float, default=30.0, help="Length parameter for high-pass filter")
    parser.add_argument('--shift_length', type=int, default=60, help="Length for future price shift for label calculation")
    
    # Chunking Args
    parser.add_argument('--chunk_size', type=int, default=3600*4, help='Number of rows per chunked file (0 to disable chunking)')
    parser.add_argument('--chunked_dir', type=str, default='./data/processed_chunked', help='Directory to save chunked files')

    # Train/Valid/Test Split Args
    parser.add_argument('--train_ratio', type=float, default=0.6, help='Proportion of *chunked files* for training')
    parser.add_argument('--valid_ratio', type=float, default=0.2, help='Proportion of *chunked files* for validation')

    # Task Sampling Args
    parser.add_argument('--task_interval', type=int, default=3600, help='Interval between rolling tasks (in rows across files)')
    parser.add_argument('--support_length', type=int, default=3600*2, help='Number of *samples* (rows) in the support window')
    parser.add_argument('--query_length', type=int, default=3600, help='Number of *samples* (rows) in the query window')

    # Dataset/DataLoader Args
    parser.add_argument('--seq_length', type=int, default=10, help='Sequence length for model input')
    parser.add_argument('--stride', type=int, default=1, help='Stride for creating sequences in TimeSeriesDataset')
    parser.add_argument('--cache_size', type=int, default=10, help='LRU cache size for TimeSeriesDataset sequence reading')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size for DataLoader')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for DataLoader')
    parser.add_argument('--pin_memory', action='store_true', help='Use pin memory for DataLoader')

    # Model Args
    parser.add_argument('--model_type', type=str, choices=['mlplob', 'tlob', 'gru'], default='gru', help="Base model type")
    parser.add_argument('--hidden_dim', type=int, default=64, help="Hidden dimension size for base model")
    parser.add_argument('--num_layers', type=int, default=2, help="Number of layers for base model")
    parser.add_argument('--num_heads', type=int, default=4, help="Number of attention heads (for TLOB base model)")
    parser.add_argument('--is_sin_emb', action='store_true', help='sinusoidal embeddings for TLOB')
    parser.add_argument('--dropout', type=float, default=0.1, help="Dropout rate (only for GRU model)")

    # DoubleAdapt Framework Args
    parser.add_argument('--da_num_head', type=int, default=8, help="Number of attention heads for DoubleAdapt adapters")
    parser.add_argument('--da_temperature', type=float, default=10.0, help="Temperature for DoubleAdapt attention")
    parser.add_argument('--lr_theta', type=float, default=0.0005, help="Learning rate for forecast model")
    parser.add_argument('--lr_da', type=float, default=0.005, help="Learning rate for data adapters")
    parser.add_argument('--sigma', type=float, default=1.0, help='Sigma for label adapter regularization')
    parser.add_argument('--reg', type=float, default=0.5, help='Coefficient for label adapter regularization')
    parser.add_argument('--first_order', action='store_true', help='Use first-order approximation for MAML')
    parser.add_argument('--adapt_x', action='store_true', help='feature adaptation')
    parser.add_argument('--adapt_y', action='store_true', help='label adaptation')
    parser.add_argument('--is_rnn', action='store_true', help='Flag if the base model is RNN-based (affects higher setup)')

    # Training Args
    parser.add_argument('--task_type', type=str, choices=['regression', 'classification'], default='regression', help="Task type")
    parser.add_argument('--metric_name', type=str, default='concordance', help="Metric name for evaluation (e.g., concordance, pearson, mse for regression; accuracy, f1, matthews_corr for classification)")
    parser.add_argument('--max_epochs', type=int, default=20, help="Maximum number of epochs for offline training")
    parser.add_argument('--offline_patience', type=int, default=5, help="Early stopping patience for offline training")
    parser.add_argument('--output_dir', type=str, default="outputs", help="Directory to save outputs")
    parser.add_argument('--model_prefix', type=str, default="v1", help="Prefix for saved model and artifact files")
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID to use')

    args = parser.parse_args()

    # --- 1. Setup ---
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create output directory and chunked directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.chunked_dir, exist_ok=True) # Ensure chunked dir exists

    # Define paths for artifacts
    scaler_path = os.path.join(args.output_dir, f"{args.model_prefix}_scaler.pkl")
    features_path = os.path.join(args.output_dir, f"{args.model_prefix}_features.pkl")
    intermediate_processed_path = os.path.join(args.output_dir, f"{args.model_prefix}_processed_temp.parquet")


    # --- 2. Load Data ---
    df = load_data(args.data_path)
    df = df.drop_nulls() # Initial drop of rows with nulls
    print(f"Loaded data shape after initial null drop: {df.shape}")
    print(df.columns)
    feature_columns = ['bid_size_1', 'ask_price_1', 'ask_size_1', 'bid_price_2', 'bid_size_2', 'ask_price_2', 'ask_size_2', 'bid_price_3', 'bid_size_3', 'ask_price_3', 'ask_size_3', 'bid_price_4', 'bid_size_4', 'ask_price_4', 'ask_size_4', 'bid_price_5', 'bid_size_5', 'ask_price_5', 'ask_size_5', 'bid_price_6', 'bid_size_6', 'ask_price_6', 'ask_size_6', 'bid_price_7', 'bid_size_7', 'ask_price_7', 'ask_size_7', 'bid_price_8', 'bid_size_8', 'ask_price_8', 'ask_size_8', 'bid_price_9', 'bid_size_9', 'ask_price_9', 'ask_size_9', 'bid_price_10', 'bid_size_10', 'ask_price_10', 'ask_size_10', 'bid_price_11', 'bid_size_11', 'ask_price_11', 'ask_size_11', 'bid_price_12', 'bid_size_12', 'ask_price_12', 'ask_size_12', 'bid_price_13', 'bid_size_13', 'ask_price_13', 'ask_size_13', 'bid_price_14', 'bid_size_14', 'ask_price_14', 'ask_size_14', 'bid_price_15', 'bid_size_15', 'ask_price_15', 'ask_size_15', 'bid_price_16', 'bid_size_16', 'ask_price_16', 'ask_size_16', 'bid_price_17', 'bid_size_17', 'ask_price_17', 'ask_size_17', 'bid_price_18', 'bid_size_18', 'ask_price_18', 'ask_size_18', 'bid_price_19', 'bid_size_19', 'ask_price_19', 'ask_size_19', 'bid_price_20', 'bid_size_20', 'ask_price_20', 'ask_size_20']
    with open(features_path, "wb") as f:
        pickle.dump(feature_columns, f)
    # --- 3. Load Feature Columns ---
    try:
        if os.path.exists(features_path):
             with open(features_path, "rb") as f:
                 feature_columns = pickle.load(f)
             print(f"Loaded feature column list from previous run: {features_path}")
        elif os.path.exists(args.features_file):
             print(f"Loading feature names from: {args.features_file}...")
             with open(args.features_file, "r", encoding='utf-8') as f:
                 # Read lines, strip whitespace, remove potential quotes/commas
                 feature_columns = [line.strip().strip("'\" ,") for line in f if line.strip()]
             # Save as pickle for future use
             with open(features_path, "wb") as f:
                 pickle.dump(feature_columns, f)
             print(f"Saved feature column list to {features_path}")
        else:
             raise FileNotFoundError(f"Features file not found at {args.features_file} or {features_path}. Cannot determine feature columns.")
    except Exception as e:
        raise Exception(f"Error loading feature columns: {str(e)}")

    # Ensure price column is not in features
    if args.price_column in feature_columns:
        print(f"Warning: Price column '{args.price_column}' found in feature list. Removing it.")
        feature_columns.remove(args.price_column)
    print(f"Using {len(feature_columns)} feature columns.")


    # --- 4. Prepare Features and Labels ---
    print("Preparing features and labels (calculating trend, normalizing labels)...")
    df_processed, final_feature_columns, label_column_name = prepare_features_and_labels(
        df,
        price_column=args.price_column,
        features_columns=feature_columns, # Pass the loaded feature columns
        filter_length=args.filter_length,
        shift_length=args.shift_length
    )
    print(f"Processed data shape: {df_processed.shape}")
    print(f"Using label column: '{label_column_name}'")
    print(f"Final feature columns ({len(final_feature_columns)}): {final_feature_columns}")

    # Save the final feature list actually used
    with open(features_path, "wb") as f:
        pickle.dump(final_feature_columns, f)
    print(f"Final feature list saved to {features_path}")


    # --- 5. Normalize Features ---
    print("Normalizing features...")
    scaler = RobustZScoreNorm()
    # Fit on training portion (first train_ratio part of the processed data)
    # Use a fraction based on train_ratio for fitting the scaler
    fit_end_index = int(len(df_processed) * args.train_ratio)
    if fit_end_index > 0:
        train_data_for_fit = df_processed[:fit_end_index][final_feature_columns].to_numpy()
        scaler.fit(train_data_for_fit)
        print(f"Scaler fitted on the first {args.train_ratio*100:.1f}% of the processed data.")
    else:
        print("Warning: Not enough data to fit scaler based on train_ratio. Fitting on all data.")
        scaler.fit(df_processed[final_feature_columns].to_numpy())


    # Transform the entire feature set
    features_normalized = scaler.transform(df_processed[final_feature_columns].to_numpy())

    # Create a new DataFrame with normalized features and the label
    df_normalized = pl.DataFrame(
        features_normalized,
        schema=final_feature_columns # Use final feature columns as schema
    ).with_columns(
        df_processed[label_column_name] # Add the label column back
    )
    print("Features normalized.")

    # Save scaler
    scaler.save(scaler_path) # Save using pickle format
    print(f"Scaler saved to {scaler_path}")

    # --- 6. Save Intermediate Processed Data ---
    print(f"Saving fully processed and normalized data to: {intermediate_processed_path}")
    df_normalized.write_parquet(intermediate_processed_path)
    print("Intermediate data saved.")
    del df, df_processed, df_normalized # Free up memory

    # --- 7. Chunk Intermediate File ---
    if args.chunk_size > 0:
        chunked_files = chunk_parquet_files(
            input_file=intermediate_processed_path,
            chunk_size=args.chunk_size,
            output_dir=args.chunked_dir,
            prefix=args.model_prefix # Use model prefix for chunk names
        )
    else:
        print("Chunking disabled (chunk_size=0). Using the single processed file.")
        # If not chunking, the 'chunked_files' list contains the single intermediate file
        chunked_files = [intermediate_processed_path]

    if not chunked_files:
        raise ValueError("No chunked files were created or found. Check chunking process.")

    # --- 8. Split Chunked Files ---
    print("Splitting list of chunked files...")
    train_files, valid_files, test_files = split_files(
        chunked_files,
        args.train_ratio,
        args.valid_ratio,
        shuffle=False
    )

    # --- 9. Get Metadata and Generate Tasks ---
    # Get metadata required by the sampler for each split
    train_file_infos = get_file_infos(train_files)
    valid_file_infos = get_file_infos(valid_files)
    test_file_infos = get_file_infos(test_files)
    print(train_file_infos)
    print(valid_file_infos)
    print(test_file_infos)

    # Extract sorted filenames (get_file_infos already sorts them)
    sorted_train_filenames = [info['filename'] for info in train_file_infos]
    sorted_valid_filenames = [info['filename'] for info in valid_file_infos]
    sorted_test_filenames = [info['filename'] for info in test_file_infos]

    # Initialize task sampler
    sampler = RollingTaskSampler(
        interval=args.task_interval,
        sequence_length=args.seq_length, # Model's input sequence length
        support_length=args.support_length, # Number of *samples* (rows) in support window
        query_length=args.query_length,     # Number of *samples* (rows) in query window
        task_type=args.task_type
    )

    # Generate tasks for each set
    print("Generating tasks...")
    train_tasks = sampler.generate_tasks(train_file_infos, sorted_train_filenames)
    valid_tasks = sampler.generate_tasks(valid_file_infos, sorted_valid_filenames)
    test_tasks = sampler.generate_tasks(test_file_infos, sorted_test_filenames)
    print(f"Generated tasks: Train={len(train_tasks)}, Valid={len(valid_tasks)}, Test={len(test_tasks)}")

    if not train_tasks or not valid_tasks:
         print("Warning: Train or validation task set is empty. Check sampler parameters, data size, and chunking.")
         # Decide how to handle this - exit or proceed? Exiting is safer.
         raise ValueError("Cannot proceed without training or validation tasks.")
    if not test_tasks:
         print("Warning: Test task set is empty. Online evaluation will be skipped.")


    # --- 10. Setup Model ---
    print(f"Setting up base model: {args.model_type}...")
    feature_dim = len(final_feature_columns) # Dimension of normalized features

    if args.model_type == 'mlplob':
        base_model = setup_mlplob_model(
            sequence_length=args.seq_length,
            feature_dim=feature_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            output_dim=1, # Assuming regression or binary classification output
            device=device
        )
    elif args.model_type == 'tlob':
        base_model = setup_tlob_model(
            sequence_length=args.seq_length,
            feature_dim=feature_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            num_heads=args.num_heads, # Heads for TLOB itself
            output_dim=1, # Assuming regression or binary classification output
            is_sin_emb=args.is_sin_emb,
            device=device
        )
    elif args.model_type == 'gru':
        base_model = setup_gru_model(
            sequence_length=args.seq_length,
            feature_dim=feature_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            output_dim=1,
            dropout=args.dropout,
            device=device
        )
    else:
        raise ValueError(f"Unsupported model_type: {args.model_type}")

    print(f"Base model '{args.model_type}' initialized and moved to {device}.")


    # --- 11. Initialize Framework ---
    # Define loss_type based on task type
    if args.task_type == 'regression':
        loss_type = 'concordance'
    else:
        raise ValueError(f"Unsupported task_type: {args.task_type}")
    # Initialize the DoubleAdapt framework
    framework = DoubleAdaptFramework(
        # Dataset related parameters needed by create_dataloader_from_task
        feature_columns=final_feature_columns, # Pass final feature names
        label_column=label_column_name, # Pass the single label name as a list
        seq_length=args.seq_length,
        cache_size=args.cache_size,
        stride=args.stride,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        # Model and training parameters
        model=base_model, # Pass the initialized base model
        loss_type=loss_type,
        x_dim=feature_dim, # Feature dimension
        task_type=args.task_type,
        metric=args.metric_name, # Use the metric specified in args
        num_head=args.da_num_head, # DA heads
        temperature=args.da_temperature, # DA temp
        lr_theta=args.lr_theta,
        lr_da=args.lr_da,
        device=device,
        early_stopping_patience=args.offline_patience, # Use offline patience here
        sigma=args.sigma,
        reg=args.reg,
        first_order=args.first_order,
        adapt_x=args.adapt_x,
        adapt_y=args.adapt_y,
        is_rnn=args.is_rnn
    )
    print("DoubleAdaptFramework initialized.")


    # --- 12. Training ---
    print("Starting Offline Training...")
    framework.offline_training(
        train_tasks=train_tasks,
        valid_tasks=valid_tasks,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        patience=args.offline_patience # Pass patience for early stopping
    )

    print("Starting Online Training/Evaluation...")
    if not test_tasks:
         print("Skipping online evaluation as no test tasks were generated.")
         final_metric = float('nan')
    else:
        final_metric = framework.online_training(
            valid_tasks=valid_tasks, # Use validation set for online adaptation phase
            test_tasks=test_tasks,   # Evaluate on the test set
            batch_size=args.batch_size
        )

    print(f"--- Training Complete ---")
    print(f"Final Test Metric ({args.metric_name}): {final_metric:.4f}")


    # --- 13. Save Final Models ---
    print("Saving final models...")
    model_path = os.path.join(args.output_dir, f"{args.model_prefix}_model.pth")
    feature_adapter_path = os.path.join(args.output_dir, f"{args.model_prefix}_feature_adapter.pth")
    label_adapter_path = os.path.join(args.output_dir, f"{args.model_prefix}_label_adapter.pth")

    try:
        # Save state dicts (more reliable than scripting complex models)
        torch.save(framework.meta_model.model.state_dict(), model_path)
        if framework.adapt_x:
             torch.save(framework.meta_model.feature_adapter.state_dict(), feature_adapter_path)
        if framework.adapt_y:
             torch.save(framework.meta_model.label_adapter.state_dict(), label_adapter_path)
        print(f"Saved state dicts to {args.output_dir} with prefix {args.model_prefix}")

    except Exception as e:
        print(f"Error saving model state dicts: {e}")

    # --- 14. Clean up intermediate file ---
    try:
        if os.path.exists(intermediate_processed_path):
            os.remove(intermediate_processed_path)
            print(f"Removed intermediate processed file: {intermediate_processed_path}")
    except Exception as e:
        print(f"Warning: Could not remove intermediate file {intermediate_processed_path}: {e}")


    print("-" * 30)
    print("Run Summary:")
    print(f"Output Directory: {args.output_dir}")
    print(f"Model Prefix: {args.model_prefix}")
    print(f"Final Test Metric ({args.metric_name}): {final_metric:.4f}")
    print(f"Model Path: {model_path}")
    if framework.adapt_x: print(f"Feature Adapter Path: {feature_adapter_path}")
    if framework.adapt_y: print(f"Label Adapter Path: {label_adapter_path}")
    print(f"Scaler Path: {scaler_path}")
    print(f"Features Path: {features_path}")
    print(f"Chunked Data Dir: {args.chunked_dir if args.chunk_size > 0 else 'N/A (Chunking Disabled)'}")
    print("-" * 30)


if __name__ == "__main__":
    main()