import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Union, Optional, Any
import numpy as np
from scipy import stats
import random
import copy
from collections import OrderedDict
import higher
from tqdm import tqdm
from .higher_optim import *  
from .model import DoubleAdapt 
from .dataset import create_dataloader_from_task
from .loss_accumulator import LossAccumulator
from torchmetrics.functional.classification import (
    accuracy, f1_score, precision, recall,
    auroc, average_precision, cohen_kappa,
    matthews_corrcoef, specificity
)
from torchmetrics.functional.regression import (
    concordance_corrcoef, mean_squared_error,
    mean_absolute_error, r2_score, explained_variance,
    pearson_corrcoef, spearman_corrcoef,
    mean_absolute_percentage_error, symmetric_mean_absolute_percentage_error
)
            

class DoubleAdaptFramework:
    """
    DoubleAdapt Meta-Learning Framework with Two-phase Training
    
    Components:
    - Offline Training (Algorithm 1)
    - Online Training (Algorithm 1) 
    - DoubleAdapt Process (Algorithm 2)
    """
    def __init__(
        self,
        # Dataset related parameters
        feature_columns: List[str],
        label_column: str, 
        seq_length: int,
        cache_size: int,
        stride: int,
        num_workers: int,
        pin_memory: bool,
        # Model and training related parameters
        model: nn.Module,
        loss_type: str,
        x_dim: int,
        task_type: str = 'classification',
        metric: str = 'matthews_corr',
        num_head: int = 8,
        temperature: float = 10.0,
        lr_theta: float = 0.001,
        lr_da: float = 0.01,
        device: torch.device = None,
        early_stopping_patience: int = 5,
        sigma: float = 1.0,
        reg: float = 0.5,
        first_order: bool = True,
        adapt_x: bool = True,
        adapt_y: bool = True, 
        is_rnn: bool = False
    ):
        """Initialize DoubleAdapt framework
        
        Args:
            feature_columns: List of feature column names
            label_column: label column name
            seq_length: Sequence length for time series data
            cache_size: Size of cache for data loading
            stride: Stride length for sliding window
            num_workers: Number of workers for data loading
            pin_memory: Whether to pin memory in data loading
            model: Base model for adaptation
            loss_type: Loss function type (e.g., 'concordance', 'mse')
            x_dim: Feature dimension
            task_type: Type of task ('classification' or 'regression')
            metric: Evaluation metric to use
            num_head: Number of attention heads
            temperature: Temperature for attention
            lr_theta: Learning rate for the model
            lr_da: Learning rate for the data adapter 
            device: Computing device
            early_stopping_patience: Number of epochs for early stopping
            sigma: Sigma for reg_loss calculation
            reg: Regularization coefficient
            first_order: Whether to use first-order approximation
            adapt_x: Whether to use feature adaptation
            adapt_y: Whether to use label adaptation
            is_rnn: Whether the model is RNN
        """
        self.feature_columns = feature_columns
        self.label_column = label_column
        self.seq_length = seq_length
        self.cache_size = cache_size
        self.stride = stride
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.meta_model = DoubleAdapt(
            model=model,
            feature_dim=x_dim,
            num_head=num_head,
            temperature=temperature,
            device=device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        )
        
        self.loss_type = loss_type
        self.lr_theta = lr_theta
        self.sigma = sigma
        self.reg = reg
        self.first_order = first_order
        self.early_stopping_patience = early_stopping_patience
        self.device = self.meta_model.device
        
        self.adapt_x = adapt_x
        self.adapt_y = adapt_y
        self.is_rnn = is_rnn
        self.task_type = task_type
        self.metric = metric
        
        
        self.data_adapter_opt = torch.optim.Adam(
            self.meta_model.meta_parameters,
            lr=lr_da
        )
        self.forecast_opt = torch.optim.Adam(
            self.meta_model.model.parameters(),
            lr=lr_theta
        )

    def offline_training(
        self,
        train_tasks: List[Dict],  # T_train
        valid_tasks: List[Dict],  # T_valid
        max_epochs: int,
        batch_size: int = 32,
        patience: int = 5  
    ) -> None:
        """Offline training phase with memory optimization.
        
        Args:
            train_tasks: Training tasks for meta-training (T_train)
            valid_tasks: Validation tasks for meta-validation (T_valid) 
            max_epochs: Maximum training epochs
            batch_size: Size of batches
            patience: Early stopping patience 
        """
        best_metric = float('-inf')
        patience_counter = patience
        best_phi = None
        best_psi = None
        
        # Initialize progress bar
        pbar = tqdm(range(max_epochs), desc='Offline Training')
        
        # Initialize parameters efficiently
        phi = {k: v.clone() for k, v in self.meta_model.model.state_dict().items()}
        psi = {
            'feature_adapter': {k: v.clone() for k, v in self.meta_model.feature_adapter.state_dict().items()},
            'label_adapter': {k: v.clone() for k, v in self.meta_model.label_adapter.state_dict().items()}
        }
        
        for epoch in pbar:
            # Shuffle training tasks
            shuffled_train = self._shuffle_tasks(train_tasks)
            
            # Train on training set (returns CPU predictions)
            phi, psi, train_metric_mean, train_metric_std = self.double_adapt(
                tasks=shuffled_train,
                batch_size=batch_size,
                phi=phi,
                psi=psi,
                meta_model=self.meta_model
            )
        
            # Evaluate on validation set 
            _, _, valid_metric_mean, valid_metric_std = self.double_adapt(
                tasks=valid_tasks,
                batch_size=batch_size,
                phi=phi,
                psi=psi,
                meta_model=self.meta_model
            )

            # Update progress bar
            pbar.set_postfix({
                'train_metric_mean': f'{train_metric_mean:.4f}',
                'valid_metric_mean': f'{valid_metric_mean:.4f}',
                'best_metric': 'N/A' if best_metric == float('-inf') else f'{best_metric:.4f}',
                'patience': patience_counter
            })
            
            # Early stopping check
            if valid_metric_mean > best_metric:
                best_metric = valid_metric_mean
                
                # Release previous best state dicts before creating new ones
                if best_phi is not None:
                    del best_phi, best_psi
                    torch.cuda.empty_cache()
                
                # Store new best parameters
                best_phi = copy.deepcopy(phi)
                best_psi = copy.deepcopy(psi)
                patience_counter = patience
            else:
                patience_counter -= 1
                if patience_counter <= 0:
                    print(f"\nEarly stopping triggered. Best validation metric: {best_metric:.4f}")
                    break
        # Load best parameters
        if best_phi is not None and best_psi is not None:
            self.meta_model.model.load_state_dict(best_phi)
            self.meta_model.feature_adapter.load_state_dict(best_psi['feature_adapter'])
            self.meta_model.label_adapter.load_state_dict(best_psi['label_adapter'])
            print(f"Loaded best parameters with validation metric: {best_metric:.4f}")
        
        # Final cleanup
        del phi, psi
        if best_phi is not None:
            del best_phi, best_psi
        torch.cuda.empty_cache()

    def online_training(
        self,
        valid_tasks: List[Dict],  # T_valid
        test_tasks: List[Dict],   # T_test
        batch_size: int = 32,
    ) -> float:
        """Online training phase with memory optimization.
        
        Args:
            valid_tasks: Validation tasks (T_valid)
            test_tasks: Test tasks (T_test)
            batch_size: Batch size for data loading
            
        Returns:
            float: Evaluation metric on test set
        """
        print("\nStarting online training...")
        
        # Line 9: Execute DoubleAdapt on validation and test sets
        online_tasks = valid_tasks + test_tasks
        
        # Get current parameters efficiently
        phi = {k: v.clone() for k, v in self.meta_model.model.state_dict().items()}
        psi = {
            'feature_adapter': {k: v.clone() for k, v in self.meta_model.feature_adapter.state_dict().items()},
            'label_adapter': {k: v.clone() for k, v in self.meta_model.label_adapter.state_dict().items()}
        }
        
        # Execute DoubleAdapt with progress bar
        with tqdm(total=1, desc='Online Adaptation') as pbar:
            phi, psi, test_metric_mean, test_metric_std = self.double_adapt(
                tasks=online_tasks,
                batch_size=batch_size,
                phi=phi,
                psi=psi,
                meta_model=self.meta_model
            )
            pbar.update(1)
       
        # Clean up resources
        del phi, psi, online_tasks
        torch.cuda.empty_cache()
        
        print(f"Final test metric: {test_metric_mean:.4f}")
        return test_metric_mean

    def _create_loss_accumulator(self, loss_type):
        """Create a loss accumulator appropriate for the task type
        
        Args:
            loss_type: Type of loss function to accumulate
            
        Returns:
            LossAccumulator: Instance of loss accumulator
        """
        return LossAccumulator(loss_type)

    def double_adapt(
        self,
        tasks: List[Dict],
        batch_size: int,
        phi: OrderedDict,
        psi: OrderedDict,
        meta_model: DoubleAdapt
    ) -> tuple[OrderedDict, OrderedDict, float, float]:
        """Implementation of DoubleAdapt algorithm with memory optimization.
        
        Args:
            tasks: List of tasks containing support and query sets
            batch_size: Batch size for data loading
            phi: Initial model parameters
            psi: Initial adapter parameters
            meta_model: DoubleAdapt model instance
            
        Returns:
            tuple: Updated model parameters, adapter parameters, mean metric, std metric
        """
        metric_values = [] 
        
        # Load state dicts once before the loop
        meta_model.model.load_state_dict(phi)
        meta_model.feature_adapter.load_state_dict(psi['feature_adapter'])
        meta_model.label_adapter.load_state_dict(psi['label_adapter'])
        
        # Create progress bar for tasks
        task_pbar = tqdm(tasks, desc='Processing Tasks', leave=False)
        
        for task_idx, task in enumerate(task_pbar):
            # Update progress bar description
            task_pbar.set_description(f'Task {task_idx + 1}/{len(tasks)}')
            
            # Reset gradients for each task
            self.data_adapter_opt.zero_grad()
            self.forecast_opt.zero_grad()
            
            # Create dataloaders with memory-efficient settings
            support_dataloader, query_dataloader = create_dataloader_from_task(
                task=task,
                feature_columns=self.feature_columns,
                label_column=self.label_column,
                task_type=self.task_type,
                seq_length=self.seq_length,
                cache_size=self.cache_size,
                stride=self.stride,
                batch_size=batch_size,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory
            )

            # Use higher with memory optimization
            with higher.innerloop_ctx(
                meta_model.model,
                self.forecast_opt,
                copy_initial_weights=False,
                track_higher_grads=not self.first_order,
                override={'lr': [self.lr_theta]}
            ) as (fmodel, diffopt):
                with torch.backends.cudnn.flags(enabled=not (self.is_rnn or not self.first_order)):
                    # Process support set
                    l_reg = self._process_support_set(
                        support_dataloader, meta_model, fmodel, diffopt
                    )
                    
                    # Process query set (returns CPU tensor)
                    query_predictions_cpu, query_target_cpu = self._process_query_set(
                        query_dataloader, meta_model, fmodel, l_reg
                    )
                    query_predictions_cpu = query_predictions_cpu.view_as(query_target_cpu) if self.task_type == 'regression' else  query_predictions_cpu
                    # Calculate metric for the current task
                    metric_value = self._calculate_metric(
                        predictions=query_predictions_cpu,
                        targets=query_target_cpu,
                        metric=self.metric,
                        task_type=self.task_type
                    )
                    metric_values.append(metric_value)
            
            # Clean up task-specific resources
            del support_dataloader, query_dataloader, l_reg, query_predictions_cpu, query_target_cpu
            torch.cuda.empty_cache()
            
            # Update progress bar with current metric
            if len(metric_values) > 0:
                task_pbar.set_postfix({
                    'Current Metric': f'{metric_values[-1]:.4f}',
                    'Avg Metric': f'{np.mean(metric_values):.4f}'
                })

        # Close progress bar
        task_pbar.close()

        # Save final parameters efficiently
        phi_prev = {k: v.clone() for k, v in meta_model.model.state_dict().items()}
        psi_prev = {
            'feature_adapter': {k: v.clone() for k, v in meta_model.feature_adapter.state_dict().items()},
            'label_adapter': {k: v.clone() for k, v in meta_model.label_adapter.state_dict().items()}
        }

        return phi_prev, psi_prev, np.mean(metric_values), np.std(metric_values)

    def _process_support_set(
        self,
        support_dataloader,
        meta_model,
        fmodel,
        diffopt
    ) -> torch.Tensor:
        """Process support set with memory optimization.

        Args:
            support_dataloader: Data loader for the support set
            meta_model: The main DoubleAdapt model instance
            fmodel: The functionalized base model
            diffopt: The higher-order optimizer

        Returns:
            torch.Tensor: Normalized regularization loss
        """
        # Initialize values with proper device and dtype
        l_reg = torch.tensor(0.0, device=self.device, dtype=torch.float32)
        total_support_samples = 0
        loss_accumulator = self._create_loss_accumulator(self.loss_type)

        # Create progress bar for support batches
        support_pbar = tqdm(support_dataloader, desc='Support Batches', leave=False)

        for support_batch in support_pbar:
            support_x, support_y = support_batch
            # Use non_blocking for async transfer and proper dtype based on task type
            support_x = support_x.to(self.device, dtype=torch.float32, non_blocking=True)
            support_y = support_y.to(
                self.device, 
                dtype=torch.long if self.task_type == 'classification' else torch.float32, 
                non_blocking=True
            )

            # Update progress bar with current batch size
            support_pbar.set_postfix({
                'Batch Size': support_x.size(0),
                'Total Samples': total_support_samples
            })

            # Feature adaptation and prediction with memory optimization
            with torch.cuda.amp.autocast(enabled=True):
                y_hat, adapted_x = meta_model(
                    support_x,
                    model=fmodel,
                    transform=self.adapt_x
                )

                if self.adapt_y:
                    # Store reference to original labels
                    raw_y = support_y
                    # Adapt labels
                    y = meta_model.label_adapter(support_x, raw_y, inverse=False)

                    if self.task_type == 'regression':
                        # Calculate regularization term efficiently
                        reg_term = (y.view_as(raw_y) - raw_y).pow(2)
                        l_reg += reg_term.sum()
                        del reg_term  # Clean up immediately

                    # Clean up original labels reference
                    del raw_y
                else:
                    y = support_y

                # Update loss accumulator
                loss_accumulator.update(
                    y_hat.view_as(y) if self.task_type == 'regression' else y_hat,
                    y
                )

                # Update sample count
                total_support_samples += support_x.size(0)

            # Clean up tensors immediately
            del support_x, support_y, y_hat, adapted_x, y

        # Close progress bar
        support_pbar.close()

        # Calculate final loss and update model
        train_loss = loss_accumulator.get_average()
        diffopt.step(train_loss)

        # Clean up and normalize regularization loss
        del train_loss
        if total_support_samples > 0:
            l_reg = l_reg / total_support_samples
        else:
            l_reg = torch.tensor(0.0, device=self.device, dtype=torch.float32)

        # Final cleanup
        del loss_accumulator
        torch.cuda.empty_cache()

        return l_reg
    
    def _process_query_set(
        self, 
        query_dataloader, 
        meta_model, 
        fmodel, 
        l_reg
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process query set with memory optimization.
        
        Args:
            query_dataloader: Query set data loader
            meta_model: DoubleAdapt model instance
            fmodel: Functionalized model
            l_reg: Regularization loss from support set
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple of predictions and targets
        """
        # Initialize loss accumulators with memory optimization
        loss_accumulator = self._create_loss_accumulator(self.loss_type)
        old_loss_accumulator = None
        if self.first_order and self.adapt_y:
            old_loss_accumulator = self._create_loss_accumulator(self.loss_type)

        # Store predictions in GPU memory temporarily
        total_query_predictions_gpu = []
        total_query_target_gpu = [] 
        total_query_samples = 0

        # Create progress bar for query batches
        query_pbar = tqdm(query_dataloader, desc='Query Batches', leave=False)

        for query_batch in query_pbar:
            query_x, query_y = query_batch
            # Use non_blocking for async transfer 
            query_x = query_x.to(self.device, non_blocking=True)
            query_y = query_y.to(self.device, non_blocking=True)

            total_query_samples += query_x.size(0)
            # Update progress bar with current batch info
            query_pbar.set_postfix({
                'Batch Size': query_x.size(0),
                'Total Samples': total_query_samples
            })

            # Feature adaptation and prediction with memory optimization
            with torch.cuda.amp.autocast(enabled=True):
                query_predictions, adapted_query_x = meta_model(
                    query_x,
                    model=fmodel,
                    transform=self.adapt_x
                )

                if self.adapt_y:
                    # Label adaptation with memory optimization
                    query_predictions = meta_model.label_adapter(
                        query_x, 
                        query_predictions, 
                        inverse=True
                    )
                    loss_accumulator.update(
                        query_predictions.view_as(query_y) if self.task_type == 'regression' else query_predictions,
                        query_y
                    )

                    # Calculate first-order approximation if needed
                    if self.first_order and old_loss_accumulator is not None:
                        with torch.no_grad():
                            pred2, _ = meta_model(adapted_query_x, model=None, transform=False)
                            pred2 = meta_model.label_adapter(
                                query_x, 
                                pred2, 
                                inverse=True
                            ).detach()
                            old_loss_accumulator.update(pred2.view_as(query_y), query_y)
                            del pred2  # Clean up immediately
                else:
                    loss_accumulator.update(
                        query_predictions.view_as(query_y) if self.task_type == 'regression' else query_predictions,
                        query_y
                    )

                # Store predictions on GPU to avoid CPU-GPU transfers
                total_query_predictions_gpu.append(query_predictions.detach())
                total_query_target_gpu.append(query_y.detach())

            # Clean up tensors immediately
            del query_x, query_y, adapted_query_x

        # Close progress bar
        query_pbar.close()

        # Calculate query loss with memory optimization
        query_loss = loss_accumulator.get_average()
        loss_diff = torch.tensor(0.0, device=self.device, dtype=torch.float32)

        if self.adapt_y:
            if self.first_order and old_loss_accumulator is not None:
                old_loss = old_loss_accumulator.get_average()
                loss_diff = query_loss - old_loss
                l_reg = l_reg * self.reg + l_reg * loss_diff.item() / self.sigma
                del old_loss  # Clean up old loss tensor
            else:
                l_reg = l_reg * self.reg
        else:
            l_reg = torch.tensor(0.0, device=self.device, dtype=torch.float32)

        # Calculate final loss and backpropagate
        final_loss = query_loss + l_reg
        final_loss.backward()

        # Update parameters
        if self.adapt_x or self.adapt_y:
            self.data_adapter_opt.step()
        self.forecast_opt.step()

        # Concatenate and move predictions to CPU efficiently
        with torch.no_grad():
            if total_query_predictions_gpu:
                all_predictions = torch.cat(total_query_predictions_gpu, dim=0)
                all_query_target = torch.cat(total_query_target_gpu, dim=0)
                # Convert predictions to appropriate dtype based on task type
                predictions_cpu = all_predictions.cpu().to(
                    dtype=torch.long if self.task_type == 'classification' else torch.float32
                )
                # Keep targets in original dtype for metric calculation
                target_cpu = all_query_target.cpu()
            else:
                # Empty tensor with appropriate dtype
                predictions_cpu = torch.tensor(
                    [], 
                    device='cpu',
                    dtype=torch.long if self.task_type == 'classification' else torch.float32
                )
                target_cpu = torch.tensor([], device='cpu')

        # Clean up GPU tensors and accumulators
        del total_query_predictions_gpu, total_query_target_gpu, query_loss, final_loss, loss_diff, l_reg
        del loss_accumulator
        if old_loss_accumulator is not None:
            del old_loss_accumulator
        if 'all_predictions' in locals():
            del all_predictions
        if 'all_query_target' in locals():
            del all_query_target

        # Clear CUDA cache
        torch.cuda.empty_cache()

        return predictions_cpu, target_cpu

    @staticmethod
    def _shuffle_tasks(tasks: List[Dict]) -> List[Dict]:
        """Shuffle task order randomly"""
        return random.sample(tasks, len(tasks))
    @staticmethod
    def _calculate_metric(
        predictions: torch.Tensor,
        targets: torch.Tensor,
        metric: str = None,
        quantile: Optional[float] = 0.5,
        task_type: str = 'regression',
        average: str = 'macro'
    ) -> float:
        """Calculate evaluation metric with memory optimization.
        
        Args:
            predictions: predctions from the model
            targets: True labels
            metric: Metric name
            quantile: Quantile for quantile metrics
            task_type: Type of task
            average: Averaging method for classification
            
        Returns:
            float: Calculated metric value
        """
        # Use no_grad context to prevent gradient tracking
        with torch.no_grad():                        
            result = 0.0  # Default value
            try:
                if task_type == 'regression':
                    if metric == 'concordance':
                        result = concordance_corrcoef(100*predictions, 100*targets).item()
                    elif metric == 'mse':
                        result = -mean_squared_error(predictions, targets).item()
                    elif metric == 'mae':
                        result = -mean_absolute_error(predictions, targets).item()
                    elif metric == 'r2':
                        result = r2_score(predictions, targets).item()
                    elif metric == 'rmse':
                        result = -torch.sqrt(mean_squared_error(predictions, targets)).item()
                    elif metric == 'mape':
                        result = -mean_absolute_percentage_error(predictions, targets).item()
                    elif metric == 'smape':
                        result = -symmetric_mean_absolute_percentage_error(predictions, targets).item()
                    elif metric == 'explained_variance':
                        result = explained_variance(predictions, targets).item()
                    elif metric == 'pearson':
                        result = pearson_corrcoef(predictions, targets).item()
                    elif metric == 'spearman':
                        result = spearman_corrcoef(predictions, targets).item()
                
                elif task_type == 'classification':
                    # Check binary classification case
                    is_binary = targets.unique().numel() <= 2
                    targets = targets.long()
                    
                    # Handle prediction format
                    if len(predictions.shape) > 1 and predictions.shape[1] > 1:
                        pred_classes = torch.argmax(predictions, dim=1)
                        pred_probs = torch.softmax(predictions, dim=1) if metric in ['auroc', 'average_precision'] else None
                    else:
                        if is_binary:
                            pred_classes = (predictions > 0.5).long()
                            pred_probs = predictions
                        else:
                            pred_classes = torch.round(predictions).long()
                            pred_probs = None
                    
                    # Calculate classification metrics
                    if metric == 'accuracy':
                        result = accuracy(pred_classes, targets, 
                                       task='binary' if is_binary else 'multiclass', 
                                       average=average).item()
                    elif metric == 'f1':
                        result = f1_score(pred_classes, targets,
                                        task='binary' if is_binary else 'multiclass',
                                        average=average).item()
                    elif metric == 'precision':
                        result = precision(pred_classes, targets,
                                        task='binary' if is_binary else 'multiclass',
                                        average=average).item()
                    elif metric == 'recall':
                        result = recall(pred_classes, targets,
                                     task='binary' if is_binary else 'multiclass',
                                     average=average).item()
                    elif metric == 'auroc':
                        if is_binary:
                            result = auroc(pred_probs, targets, task="binary").item()
                        else:
                            num_classes = (predictions.shape[1] if len(predictions.shape) > 1 
                                         else targets.max().item() + 1)
                            result = auroc(pred_probs, targets,
                                         task="multiclass",
                                         num_classes=num_classes,
                                         average=average).item()
                    elif metric == 'average_precision':
                        if is_binary:
                            result = average_precision(pred_probs, targets, task="binary").item()
                        else:
                            num_classes = (predictions.shape[1] if len(predictions.shape) > 1 
                                         else targets.max().item() + 1)
                            result = average_precision(pred_probs, targets,
                                                    task="multiclass",
                                                    num_classes=num_classes,
                                                    average=average).item()
                    elif metric == 'cohen_kappa':
                        result = cohen_kappa(pred_classes, targets,
                                          task='binary' if is_binary else 'multiclass').item()
                    elif metric == 'matthews_corr':
                        result = matthews_corrcoef(pred_classes, targets,
                                                 task='binary' if is_binary else 'multiclass').item()
                    elif metric == 'specificity':
                        result = specificity(pred_classes, targets,
                                          task='binary' if is_binary else 'multiclass',
                                          average=average).item()
                    
                    # Clean up classification-specific tensors
                    del pred_classes
                    if pred_probs is not None:
                        del pred_probs
            
            except Exception as e:
                result = 0.0
                    
            return result

