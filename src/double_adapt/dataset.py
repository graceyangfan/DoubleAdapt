from typing import List, Optional, Dict, Union, Tuple
import numpy as np
import pandas as pd 
import torch
from torch.utils.data import Dataset,DataLoader
import pyarrow.parquet as pq
import os
from functools import lru_cache

class TimeSeriesDataset(Dataset):
    """
    Dataset for efficiently loading time series sequences from Parquet files.
    
    This dataset handles multiple Parquet files, each containing time series data,
    and provides efficient access to sequences of specified length from anywhere
    in these files, along with corresponding labels.
    
    Attributes:
        file_paths (List[str]): List of paths to Parquet files
        seq_length (int): Length of each sequence to extract
        feature_columns (Optional[List[str]]): Specific columns to load as features (None = all columns except label)
        label_column (str): Name of the column containing the labels
        task_type (str): Type of task ('classification' or 'regression'), determines label tensor type
        cache_size (int): Number of sequences to cache in memory
        stride (int): Step size for sequence extraction (1 = consecutive sequences)
        sequence_info (List[Dict]): Metadata for all available sequences
    """
    
    def __init__(
        self, 
        file_paths: List[str], 
        start_row_at_first:int,
        end_row_at_last:int,
        label_column: str, 
        feature_columns: Optional[List[str]] = None,
        task_type: str = "regression",
        seq_length: int = 60,
        cache_size: int = 128,
        stride: int = 1
    ) -> None:
        """
        Initialize the TimeSeriesDataset.
        
        Args:
            file_paths: List of paths to Parquet files containing time series data
            start_row_at_first: The first row to start reading from the first file
            end_row_at_last: The last row to read from the last file
            label_column: Name of the column containing the target labels
            feature_columns: Specific columns to load as features (None = all columns except label)
            task_type: 'classification' or 'regression'. Determines label tensor dtype.
            seq_length: Length of each sequence to extract
            cache_size: Number of sequences to cache in memory
            stride: Step size between consecutive sequences (for efficient sampling)
        
        Raises:
            FileNotFoundError: If any of the provided file paths doesn't exist
            ValueError: If seq_length, stride, or task_type is invalid, or label_column not found
        """
        self.file_paths = file_paths
        self.start_row_at_first = start_row_at_first
        self.end_row_at_last = end_row_at_last
        self.label_column = label_column
        self.seq_length = seq_length
        self.feature_columns = feature_columns
        self.stride = max(1, stride)  # Ensure stride is at least 1
        self.task_type = task_type.lower()
        
        if seq_length <= 0:
            raise ValueError("seq_length must be positive")
        if self.task_type not in ["classification", "regression"]:
            raise ValueError("task_type must be 'classification' or 'regression'")
        
        # Verify all files exist and check columns
        self._validate_files_and_columns()
        
        # Create LRU cache for sequence reading
        # Cache now stores tuples (features, labels)
        self.read_sequence = lru_cache(maxsize=cache_size)(self._read_sequence)
        
        # Initialize sequence info
        self.sequence_info = []
        self._index_files()
    
    def _validate_files_and_columns(self) -> None:
        """Validate file existence and column names."""
        temp_feature_columns = self.feature_columns
        columns_to_check = []
        
        for path in self.file_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"File not found: {path}")
            
            # Check columns in the first file
            pf = pq.ParquetFile(path)
            schema_names = pf.schema.names
            
            if self.label_column not in schema_names:
                raise ValueError(f"Label column '{self.label_column}' not found in file: {path}")

            if temp_feature_columns is None:
                # If feature_columns is None, use all columns except the label column
                temp_feature_columns = [col for col in schema_names if col != self.label_column]
                # Store the resolved feature columns if they were initially None
                if self.feature_columns is None:
                    self.feature_columns = temp_feature_columns 
            else:
                # Check if specified feature columns exist
                for col in temp_feature_columns:
                    if col not in schema_names:
                         raise ValueError(f"Feature column '{col}' not found in file: {path}")
                if self.label_column in temp_feature_columns:
                    raise ValueError(f"Label column '{self.label_column}' cannot be included in feature_columns")

            # Ensure feature_columns is set after checking the first file
            if self.feature_columns is None:
                 raise RuntimeError("Feature columns could not be determined.") # Should not happen

            columns_to_check = self.feature_columns + [self.label_column]
            break # Only need to check schema on the first file assuming consistency

    def _index_files(self) -> None:
        """
        Index all available sequences across all files.
        This creates a mapping from dataset index to file location information.
        """
        for file_idx, file_path in enumerate(self.file_paths):
            # Get file metadata without loading contents
            pf = pq.ParquetFile(file_path)
            file_rows = pf.metadata.num_rows
            
            # Calculate valid starting positions based on sequence length and stride
            if file_rows >= self.seq_length:
                valid_starts = range(0, file_rows - self.seq_length + 1, self.stride)
                
                # Store mapping from dataset index to file and position
                for start_pos in valid_starts:
                    if file_idx == 0 and start_pos < self.start_row_at_first:
                        continue
                    if file_idx == len(self.file_paths) - 1 and start_pos + self.seq_length > self.end_row_at_last:
                        continue
                    self.sequence_info.append({
                        'file_idx': file_idx,
                        'start_row': start_pos
                    })

    def _read_sequence(self, file_path: str, start_row: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Read a sequence (features and label) from a Parquet file.
        This is the internal implementation wrapped with LRU cache.
        
        Args:
            file_path: Path to the Parquet file
            start_row: Starting row index of the sequence
            
        Returns:
            A tuple containing:
                - features (np.ndarray): Sequence feature data
                - labels (np.ndarray): Sequence label data
        """
        pf = pq.ParquetFile(file_path)
        
        # Determine columns to read: features + label
        columns_to_read = self.feature_columns + [self.label_column]
        
        # --- Efficient row group reading logic (simplified for clarity) ---
        # A more robust implementation would identify specific row groups
        # For simplicity here, we read a slice which might be less efficient
        # on very large files but easier to implement correctly.
        # Note: pyarrow's read_table can be faster for slices than read_row_group loops
        
        table = pq.read_table(file_path, columns=columns_to_read)
        # Slice the table directly (more efficient for this case than row group logic)
        sequence_table = table.slice(start_row, self.seq_length)
        df = sequence_table.to_pandas()
        # Separate features and labels
        features = df[self.feature_columns].values
        label = df.iloc[-1][self.label_column]
        
        return features, label
    
    def __len__(self) -> int:
        """Return the total number of available sequences."""
        return len(self.sequence_info)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a specific sequence (features and label) by index.
        
        Args:
            idx: The index of the sequence to retrieve
            
        Returns:
            A tuple containing:
                - features (torch.Tensor): Float tensor of feature data
                - labels (torch.Tensor): Float or Long tensor of label data, based on task_type
            
        Raises:
            IndexError: If the index is out of range
        """
        if idx < 0 or idx >= len(self.sequence_info):
            raise IndexError(f"Index {idx} out of range [0, {len(self.sequence_info) - 1}]")
        
        # Get sequence location
        seq_info = self.sequence_info[idx]
        file_path = self.file_paths[seq_info['file_idx']]
        start_row = seq_info['start_row']
        
        # Read sequence (features, labels) - uses cache
        features_np, label_np = self.read_sequence(file_path, start_row)
        
        # Convert to tensors with appropriate types
        features_tensor = torch.tensor(features_np, dtype=torch.float32)
        
        if self.task_type == "classification":
            label_tensor = torch.tensor(label_np, dtype=torch.long)
        else: # regression
            label_tensor = torch.tensor(label_np, dtype=torch.float32)
        return features_tensor, label_tensor
    
    def get_sequence_info(self, idx: int) -> Dict:
        """
        Get metadata about a specific sequence.
        
        Args:
            idx: The index of the sequence
            
        Returns:
            Dictionary with file_path and start_row
        """
        if idx < 0 or idx >= len(self.sequence_info):
            raise IndexError(f"Index {idx} out of range")
        
        info = self.sequence_info[idx].copy()
        info['file_path'] = self.file_paths[info['file_idx']]
        return info
    
    def get_file_stats(self) -> List[Dict]:
        """
        Get statistics about the files in this dataset.
        
        Returns:
            List of dictionaries with file statistics
        """
        stats = []
        for i, file_path in enumerate(self.file_paths):
            pf = pq.ParquetFile(file_path)
            stats.append({
                'file_path': file_path,
                'num_rows': pf.metadata.num_rows,
                'num_row_groups': pf.metadata.num_row_groups,
                'num_sequences': sum(1 for info in self.sequence_info if info['file_idx'] == i)
            })
        return stats
    

class RollingTaskSampler:
    """
    A class to generate rolling tasks for meta-learning and read task-specific data.

    Attributes:
    -----------
    interval : int
        The interval between tasks.
    sequence_length : int
        The length of the input sequence for each sample.
    support_length : int
        The number of samples in the support set.
    query_length : int
        The number of samples in the query set.
    task_type : str
        The type of task (e.g., "classification" or "regression").
    """
    def __init__(
        self,
        interval: int,
        sequence_length: int,
        support_length: int,
        query_length: int,
        task_type: str = "classification"
    ):
        if interval <= 0 or support_length <= 0 or query_length <= 0:
            raise ValueError("All window lengths must be positive.")

        self.interval = interval
        self.sequence_length = sequence_length 
        self.support_length = support_length
        self.query_length = query_length
        self.total_length = support_length + query_length  # Total length of support and query sets
        self.task_type = task_type

    def generate_tasks(
        self,
        file_infos: List[Dict[str, int]],
        sorted_filenames: List[str],
    ) -> List[Dict]:
        """
        Generate rolling tasks based on file information and filenames.

        Parameters:
        -----------
        file_infos : List[Dict[str, int]]
            A list of dictionaries containing file metadata (e.g., filename and length).
        sorted_filenames : List[str]
            A list of filenames sorted in the desired order.       
        Returns:
        --------
        List[Dict]:
            A list of tasks, each containing indices for support and query sets, task ID, and filenames.
        """
        task_id = 0
        tasks = []
        start_idx = 0
        # Compute cumulative indices for file boundaries
        cumsum_index = np.cumsum([item["length"] for item in file_infos]) - 1 
        print(cumsum_index)
        while start_idx + self.total_length + self.sequence_length - 1 < cumsum_index[-1]:
            support_end_index = start_idx + self.support_length - 1
            total_end_index = start_idx + self.total_length - 1
            # Calculate indices for support and query sets
            support_file_start_index = np.searchsorted(cumsum_index, start_idx)
            support_file_end_index = np.searchsorted(cumsum_index, support_end_index)
            query_file_start_index = np.searchsorted(cumsum_index, support_end_index + 1)
            query_file_end_index = np.searchsorted(cumsum_index, total_end_index)

            support_start_row_at_first = start_idx - cumsum_index[support_file_start_index-1] if support_file_start_index > 0 else start_idx
            support_end_row_at_last = support_end_index - cumsum_index[support_file_end_index-1] if support_file_end_index > 0 else support_end_index
            query_start_row_at_first = support_end_index + 1 - cumsum_index[query_file_start_index-1] if query_file_start_index > 0 else support_end_index + 1 
            query_end_row_at_last = total_end_index - cumsum_index[query_file_end_index-1] if query_file_end_index > 0 else total_end_index 

            task = {
                'support_start_row_at_first': support_start_row_at_first,
                'support_end_row_at_last': support_end_row_at_last,
                'query_start_row_at_first': query_start_row_at_first,
                'query_end_row_at_last': query_end_row_at_last,
                'task_id': task_id,
                'support_filenames': [sorted_filenames[i] for i in range(support_file_start_index, support_file_end_index + 1)],
                'query_filenames': [sorted_filenames[i] for i in range(query_file_start_index, query_file_end_index + 1)],
            }
            tasks.append(task)
            start_idx += self.interval
            task_id += 1
        return tasks

def create_dataloader_from_task(
    task: Dict,
    feature_columns: Optional[List[str]],
    label_column: str,
    task_type: str,
    seq_length: int,
    cache_size: int,
    stride: int = 1,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create DataLoader for support and query datasets based on the provided task.
    """
    support_dataset = TimeSeriesDataset(
        file_paths=task['support_filenames'],
        start_row_at_first=task['support_start_row_at_first'],
        end_row_at_last=task['support_end_row_at_last'],
        label_column=label_column,
        feature_columns=feature_columns,
        task_type=task_type,
        seq_length=seq_length,
        cache_size=cache_size,
        stride=stride
    )
    query_dataset = TimeSeriesDataset(
        file_paths=task['query_filenames'],
        start_row_at_first=task['query_start_row_at_first'],
        end_row_at_last=task['query_end_row_at_last'],
        label_column=label_column,
        feature_columns=feature_columns,
        task_type=task_type,
        seq_length=seq_length,
        cache_size=cache_size,
        stride=stride
    )
    support_dataloader = DataLoader(
        support_dataset, 
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    query_dataloader = DataLoader(
        query_dataset, 
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    return  support_dataloader, query_dataloader
