import torch
import torch.nn as nn

class LossAccumulator:
    """Accumulates statistics batch-wise to calculate Concordance Correlation Coefficient."""

    def __init__(self, loss_type='concordance'):
        """Initialize accumulator with optimized memory settings."""
        if loss_type != 'concordance':
            raise NotImplementedError(f"LossAccumulator currently only supports 'concordance', not '{loss_type}'")

        self.scaler = 100.0 
        self.loss_type = loss_type
        self.total_samples = 0
        self.device = None  # Will be set on first update

        # Initialize sums as None, will become tensors on first update
        self.sum_xy = None
        self.sum_x = None
        self.sum_y = None
        self.sum_x2 = None
        self.sum_y2 = None

    def _init_device_and_sums(self, tensor_input):
        """Initialize device and concordance sums with memory optimization."""
        # This check ensures initialization happens only once and only for concordance
        if self.device is None and self.loss_type == 'concordance':
            self.device = tensor_input.device
            # Use float32 for consistent precision and memory efficiency
            zeros = torch.zeros(1, device=self.device, dtype=torch.float32)
            self.sum_xy = zeros.clone()
            self.sum_x = zeros.clone()
            self.sum_y = zeros.clone()
            self.sum_x2 = zeros.clone()
            self.sum_y2 = zeros.clone()

    def update(self, batch_predict: torch.Tensor, batch_target: torch.Tensor):
        """Update accumulated statistics with memory-optimized operations."""
        # Initialize device and sums on first call
        if self.device is None:
            self._init_device_and_sums(batch_predict)

        # Ensure tensors are on the correct device and type with non-blocking transfer
        batch_predict = batch_predict.to(self.device, dtype=torch.float32, non_blocking=True) *self.scaler 
        batch_target = batch_target.to(self.device, dtype=torch.float32, non_blocking=True) *self.scaler 

        # Flatten predictions and targets
        y_pred = batch_predict.view(-1)
        y_true = batch_target.view(-1)
        batch_size = y_pred.shape[0]

        if batch_size == 0: 
            return  # Skip empty batches

        # Calculate and accumulate statistics directly on device
        with torch.cuda.amp.autocast(enabled=True):
            self.sum_xy += torch.sum(y_pred * y_true)
            self.sum_x += torch.sum(y_pred)
            self.sum_y += torch.sum(y_true)
            self.sum_x2 += torch.sum(y_pred**2)
            self.sum_y2 += torch.sum(y_true**2)
            self.total_samples += batch_size

        # Clean up intermediate tensors
        del y_pred, y_true

    def get_average(self):
        """Calculate final Concordance loss with memory optimization."""
        if self.total_samples == 0 or self.sum_x is None:
            return torch.tensor(0.0, device=self.device)

        # Use float32 for calculations
        n = torch.tensor(self.total_samples, device=self.device, dtype=torch.float32)

        # Calculate means with memory optimization
        mean_x = self.sum_x / n
        mean_y = self.sum_y / n

        # Calculate variances with numerical stability
        epsilon_var = torch.tensor(1e-8, device=self.device, dtype=torch.float32)
        var_x = (self.sum_x2 / n - mean_x**2).clamp(min=0)
        var_y = (self.sum_y2 / n - mean_y**2).clamp(min=0)

        # Calculate covariance
        cov_xy = self.sum_xy / n - mean_x * mean_y

        # Calculate denominator with numerical stability
        denominator = torch.sqrt((var_x + epsilon_var) * (var_y + epsilon_var))
        epsilon_denom = torch.tensor(1e-6, device=self.device, dtype=torch.float32)

        # Check denominator validity with memory optimization
        if denominator < epsilon_denom:
            concordance = torch.tensor(0.0, device=self.device, dtype=torch.float32)
        else:
            concordance = cov_xy / denominator
            concordance = torch.clamp(concordance, -1.0, 1.0)

        # Clean up intermediate tensors
        del mean_x, mean_y, var_x, var_y, cov_xy, denominator
        torch.cuda.empty_cache()

        return 1.0 - concordance

    def __del__(self):
        """Release tensor resources on deletion."""
        try:
            tensor_attrs = ['sum_xy', 'sum_x', 'sum_y', 'sum_x2', 'sum_y2']
            for attr_name in tensor_attrs:
                if hasattr(self, attr_name) and isinstance(getattr(self, attr_name), torch.Tensor):
                    delattr(self, attr_name)
            torch.cuda.empty_cache()
        except Exception:
            pass

