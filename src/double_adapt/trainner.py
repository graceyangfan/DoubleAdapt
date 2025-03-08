import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Union, Optional, Any
import numpy as np
import random
import copy
from collections import OrderedDict
import higher
from tqdm import tqdm
from .higher_optim import *  
from .model import DoubleAdapt  

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
        model: nn.Module,
        criterion: Any,
        x_dim: int,
        num_head: int = 8,
        temperature: float = 10.0,
        lr_theta: float = 0.001,  # Learning rate for forecast model η_θ
        lr_da: float = 0.01,      # Learning rate for data adapter
        early_stopping_patience: int = 5,
        device: torch.device = None,
        sigma: float = 1.0,       # For reg_loss calculation
        reg: float = 0.5,         # Regularization coefficient
        first_order: bool = True,  # Whether to use first-order approximation
        adapt_x: bool = True,     # Whether to use feature adaptation
        adapt_y: bool = True,     # Whether to use label adaptation
        is_rnn: bool = False      # Whether the model is RNN
    ):
        """Initialize DoubleAdapt framework
        
        Args:
            model: model adapter model
            criterion: Loss function
            x_dim: Feature dimension
            num_head: Number of attention heads
            temperature: Temperature for attention
            lr_theta: Learning rate for the model
            lr_da: Learning rate for the data adapter
            early_stopping_patience: Number of epochs for early stopping
            device: Computing device
            sigma: Sigma for reg_loss calculation
            reg: Regularization coefficient
            first_order: Whether to use first-order approximation
            adapt_x: Whether to use feature adaptation
            adapt_y: Whether to use label adaptation
            is_rnn: Whether the model is RNN
        """
        self.meta_model = DoubleAdapt(
            model=model,
            feature_dim=x_dim,
            num_head=num_head,
            temperature=temperature,
            device=device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        )
        
        self.criterion = criterion
        self.lr_theta = lr_theta
        self.sigma = sigma
        self.reg = reg
        self.first_order = first_order
        self.early_stopping_patience = early_stopping_patience
        self.device = self.meta_model.device
        
        self.adapt_x = adapt_x
        self.adapt_y = adapt_y
        self.is_rnn = is_rnn
        
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
        patience: int = 5  # ζ epochs
    ) -> None:
        """Offline training phase (Algorithm 1, Line 2-8)
        
        Args:
            train_tasks: Training tasks for meta-training (T_train)
            valid_tasks: Validation tasks for meta-validation (T_valid) 
            max_epochs: Maximum training epochs
            patience: Early stopping patience (ζ)
        """
        best_metric = float('-inf')  # Changed to -inf since we want to maximize IC
        patience_counter = patience
        best_phi = None
        best_psi = None
        
        # Initialize progress bar
        pbar = tqdm(range(max_epochs), desc='Offline Training')
        
        # Initialize parameters (Line 1)
        phi = self.meta_model.model.state_dict()
        psi = {
            'feature_adapter': self.meta_model.feature_adapter.state_dict(),
            'label_adapter': self.meta_model.label_adapter.state_dict()
        }
        
        # Offline training phase (Line 2-8)
        for epoch in pbar:
            # Shuffle training tasks (Line 3)
            shuffled_train = self._shuffle_tasks(train_tasks)
            
            # Train on training set (Line 4)
            phi, psi, train_predictions = self.double_adapt(
                tasks=shuffled_train,
                phi=phi,
                psi=psi,
                meta_model=self.meta_model
            )
            
            # Calculate training metric
            train_metric = self._calculate_metric(
                predictions=train_predictions,
                query_y=[task['query_y'].to(self.device) for task in train_tasks]
            )
            
            # Save current parameters
            curr_phi = copy.deepcopy(phi)
            curr_psi = copy.deepcopy(psi)
            
            # Evaluate on validation set (Line 6)
            _, _, valid_predictions = self.double_adapt(
                tasks=valid_tasks,
                phi=phi,
                psi=psi,
                meta_model=self.meta_model
            )
            
            # Restore parameters
            self.meta_model.model.load_state_dict(curr_phi)
            self.meta_model.feature_adapter.load_state_dict(curr_psi['feature_adapter'])
            self.meta_model.label_adapter.load_state_dict(curr_psi['label_adapter'])
            
            # Calculate validation metric (Line 7)
            valid_metric = self._calculate_metric(
                predictions=valid_predictions,
                query_y=[task['query_y'].to(self.device) for task in valid_tasks]
            )
            
            # Update progress bar with metrics
            pbar.set_postfix({
                'train_ic': f'{train_metric:.4f}',
                'valid_ic': f'{valid_metric:.4f}',
                'best_ic': f'{best_metric:.4f}',
                'patience': patience_counter
            })
            
            # Early stopping check (Line 8)
            if valid_metric > best_metric:
                best_metric = valid_metric
                best_phi = copy.deepcopy(curr_phi)
                best_psi = copy.deepcopy(curr_psi)
                patience_counter = patience
            else:
                patience_counter -= 1
                if patience_counter <= 0:
                    print(f"\nEarly stopping triggered. Best validation IC: {best_metric:.4f}")
                    break
        
        # Load best parameters
        if best_phi is not None and best_psi is not None:
            self.meta_model.model.load_state_dict(best_phi)
            self.meta_model.feature_adapter.load_state_dict(best_psi['feature_adapter'])
            self.meta_model.label_adapter.load_state_dict(best_psi['label_adapter'])

    def online_training(
        self,
        valid_tasks: List[Dict],  # T_valid
        test_tasks: List[Dict],   # T_test
    ) -> float:
        """Online training phase (Algorithm 1, Line 9-11)
        
        Args:
            valid_tasks: Validation tasks (T_valid)
            test_tasks: Test tasks (T_test)
            
        Returns:
            metric: Evaluation metric on test set
        """
        print("\nStarting online training...")
        
        # Line 9: Execute DoubleAdapt on validation and test sets
        online_tasks = valid_tasks + test_tasks
        
        # Get current parameters as initial values
        phi = self.meta_model.model.state_dict()
        psi = {
            'feature_adapter': self.meta_model.feature_adapter.state_dict(),
            'label_adapter': self.meta_model.label_adapter.state_dict()
        }
        
        # Execute DoubleAdapt with progress bar
        with tqdm(total=1, desc='Online Adaptation') as pbar:
            phi, psi, predictions = self.double_adapt(
                tasks=online_tasks,
                phi=phi,
                psi=psi,
                meta_model=self.meta_model
            )
            pbar.update(1)
        
        # Line 10: Calculate metric on test set
        test_metric = self._calculate_metric(
            predictions=predictions[-len(test_tasks):],
            query_y=[task['query_y'].to(self.device) for task in test_tasks]
        )
        
        print(f"Final test IC: {test_metric:.4f}")
        return test_metric

    def double_adapt(
        self,
        tasks: List[Dict],
        phi: OrderedDict,
        psi: OrderedDict,
        meta_model: DoubleAdapt
    ) -> tuple[OrderedDict, OrderedDict, List[torch.Tensor]]:
        """Implementation of DoubleAdapt algorithm (Algorithm 2)
        
        Args:
            tasks: List of tasks containing support and query sets
            phi: Initial model parameters φ^(m-1)
            psi: Initial adapter parameters ψ^(m-1)
            meta_model: DoubleAdapt model instance
            
        Returns:
            tuple: Updated model parameters, adapter parameters and predictions
        """
        all_predictions = []

        # Load initial parameters
        meta_model.model.load_state_dict(phi)
        meta_model.feature_adapter.load_state_dict(psi['feature_adapter'])
        meta_model.label_adapter.load_state_dict(psi['label_adapter'])
        
        for task in tasks:
            self.data_adapter_opt.zero_grad()
            self.forecast_opt.zero_grad()
            
            support_x = task['support_x'].to(self.device)
            support_y = task['support_y'].to(self.device)
            
            # Meta-learning with higher
            with higher.innerloop_ctx(
                meta_model.model,
                self.forecast_opt,
                copy_initial_weights=False,
                track_higher_grads=not self.first_order,
                override={'lr': [self.lr_theta]}
            ) as (fmodel, diffopt):
                with torch.backends.cudnn.flags(enabled=self.first_order or not self.is_rnn):
                    y_hat, adapted_support_x = meta_model(
                        support_x,
                        model=fmodel,
                        transform=self.adapt_x
                    )
                
                # Apply label adaptation if enabled
                if self.adapt_y:
                    raw_y = support_y
                    y = meta_model.label_adapter(
                        support_x,
                        raw_y,
                        inverse=False 
                    )
                # Calculate support set loss (Eq. 15) on agent space 
                train_loss = self.criterion(y_hat, y)
                diffopt.step(train_loss)
                
                # Forward pass on query set
                query_x = task['query_x'].to(self.device)
                query_y = task['query_y'].to(self.device)
                query_predictions, adapted_query_x = meta_model(
                    query_x,
                    model=fmodel,
                    transform=self.adapt_x
                )
                
                if self.adapt_y:
                    query_predictions = meta_model.label_adapter(query_x, query_predictions, inverse=True)
                
                # Calculate query set loss and regularization 
                query_loss = self.criterion(query_predictions, query_y) # in real target space 
                # Calculate regularization loss if label adaptation is enabled
                if self.adapt_y:
                    # Use support_y directly instead of raw_y
                    if not self.first_order:
                        y = meta_model.label_adapter(support_x, raw_y, inverse=False)
                    loss_y = F.mse_loss(y, raw_y)
                    if self.first_order:
                        with torch.no_grad():
                            pred2, _ = meta_model(adapted_query_x, model=None, transform=False)
                            pred2 = meta_model.label_adapter(query_x, pred2, inverse=True).detach()
                            loss_old = self.criterion(pred2.view_as(query_y), query_y)
                        loss_y = (loss_old.item() - query_loss.item()) / self.sigma * loss_y + loss_y * self.reg
                    else:
                        loss_y = loss_y * self.reg
                    loss_y.backward()
                # Update adapters
                query_loss.backward()
                if self.adapt_x or self.adapt_y:
                    self.data_adapter_opt.step()
                self.forecast_opt.step() 
                # Collect predictions
                all_predictions.append(query_predictions.detach().cpu())  # Keep as tensor

        # Save final parameters
        phi_prev = copy.deepcopy(meta_model.model.state_dict())
        psi_prev = dict(
            feature_adapter=copy.deepcopy(meta_model.feature_adapter.state_dict()),
            label_adapter=copy.deepcopy(meta_model.label_adapter.state_dict())
        )
        
        return phi_prev, psi_prev, all_predictions

    @staticmethod
    def _shuffle_tasks(tasks: List[Dict]) -> List[Dict]:
        """Shuffle task order randomly"""
        return random.sample(tasks, len(tasks))

    def _calculate_metric(self, predictions: List[Union[torch.Tensor, np.ndarray]], query_y: List[torch.Tensor]) -> float:
        """Calculate evaluation metric (e.g. IC) for different tasks 
        
        Args:
            predictions: List of prediction arrays (either torch tensors or numpy arrays)
            query_y: List of ground truth tensors
            
        Returns:
            float: Calculated metric value
        """
        # Convert predictions to numpy arrays if they're still tensors
        pred_arrays = [p.numpy() if torch.is_tensor(p) else p for p in predictions]
        
        # Calculate IC (Information Coefficient)
        ic_values = []
        for pred, y in zip(pred_arrays, query_y):
            # Convert y to numpy if it's a tensor
            y_np = y.cpu().numpy() if torch.is_tensor(y) else y
            # Calculate correlation between predictions and ground truth
            corr = np.corrcoef(pred.flatten(), y_np.flatten())[0,1]
            ic_values.append(corr)
        
        # Return mean IC
        return np.mean(ic_values)