import numpy as np
from typing import List, Dict, Optional
import torch

class RollingTaskSampler:
    """
    Rolling Task Sampler for meta-learning
    
    Parameters:
    -----------
    interval : int
        Rolling interval r between tasks
    support_length : int
        Length of support set window (previously train_length)
    query_length : int
        Length of query set window (previously test_length)
        
    Example:
    --------
    >>> sampler = RollingTaskSampler(interval=5, support_length=20, query_length=5)
    >>> tasks = sampler.generate_tasks(features, labels)
    """
    def __init__(
        self,
        interval: int,
        support_length: int,
        query_length: int
    ):
        if interval <= 0 or support_length <= 0 or query_length <= 0:
            raise ValueError("All window lengths must be positive")
            
        self.interval = interval
        self.support_length = support_length
        self.query_length = query_length

    def _validate_inputs(self, features: np.ndarray, labels: np.ndarray) -> None:
        """验证输入数据的有效性"""
        if not isinstance(features, np.ndarray) or not isinstance(labels, np.ndarray):
            raise TypeError("Features and labels must be numpy arrays")
            
        if features.shape[0] != labels.shape[0]:
            raise ValueError("Features and labels must have same number of samples")
            
        min_length = self.support_length + self.query_length
        if features.shape[0] < min_length:
            raise ValueError(
                f"Data length must be at least {min_length} "
                f"(support_length + query_length)"
            )

    def generate_tasks(
        self, 
        features: np.ndarray,
        labels: np.ndarray,
        to_tensor: bool = False 
    ) -> List[Dict]:
        """
        Generate rolling tasks from features and labels
        
        Parameters:
        -----------
        features : np.ndarray, shape (n_samples, n_features) or (n_samples, sequence_length, n_features)
            Feature matrix
        labels : np.ndarray, shape (n_samples,) 
            Label vector
            
        Returns:
        --------
        List[Dict]: List of tasks, each containing:
            - support_x: Support set features
            - support_y: Support set labels
            - query_x: Query set features
            - query_y: Query set labels
            - task_id: Task identifier
            - indices: Index information for support and query sets
        """
        self._validate_inputs(features, labels)
        
        tasks = []
        n_samples = len(features)
        task_id = 0
        
        start_idx = 0
        while start_idx + self.support_length + self.query_length <= n_samples:
            # 计算support set索引
            support_start = start_idx
            support_end = support_start + self.support_length
            
            # 计算query set索引
            query_start = support_end
            query_end = query_start + self.query_length
            
            # 创建任务
            task = {
                'support_x': features[support_start:support_end],
                'support_y': labels[support_start:support_end],
                'query_x': features[query_start:query_end],
                'query_y': labels[query_start:query_end],
                'task_id': task_id,
                'indices': {
                    'support': (support_start, support_end),
                    'query': (query_start, query_end)
                }
            }
            
            if to_tensor:
                task = self._convert_to_tensor(task)
            
            tasks.append(task)
            
            start_idx += self.interval
            task_id += 1
            
        if not tasks:
            raise ValueError("No tasks could be generated with current parameters")
            
        return tasks
    
    @staticmethod
    def _convert_to_tensor(task: Dict) -> Dict:
        """Convert numpy arrays in a task to torch tensors"""
        return {
            'support_x': torch.FloatTensor(task['support_x']),
            'support_y': torch.FloatTensor(task['support_y']),
            'query_x': torch.FloatTensor(task['query_x']),
            'query_y': torch.FloatTensor(task['query_y']),
            'task_id': task['task_id'],
            'indices': task['indices']
        }
    
    @staticmethod
    def to_torch_tensor(task_list: List[Dict]) -> List[Dict]:
        """Convert a list of tasks from numpy arrays to torch tensors"""
        return [RollingTaskSampler._convert_to_tensor(task) for task in task_list]
