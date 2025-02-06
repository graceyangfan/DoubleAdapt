# DoubleAdapt

A refined implementation of the double adaptation framework for time series prediction, based on the paper ["Double Adaptation for Time Series Forecasting"](https://arxiv.org/abs/2306.09862) (ICML 2023). This repository is a restructured version of the [original implementation](https://github.com/SJTU-DMTai/DoubleAdapt) with improved code organization and usability.

## Overview

DoubleAdapt addresses the challenge of distribution shifts in time series forecasting through a novel double adaptation framework that combines:

1. **Offline Meta-Learning**: Learns generalizable patterns across historical tasks
2. **Online Adaptation**: Rapidly adapts to new data distributions using meta-learned initialization

Key advantages:
- Handles both gradual and abrupt distribution shifts
- Requires minimal adaptation data
- Maintains model stability while enabling quick adaptation
- Achieves superior performance compared to traditional online learning methods

## Features

- **Enhanced Implementation**:
  - Clean, modular code structure
  - Type hints and comprehensive documentation
  - Improved data preprocessing with robust Z-score normalization
  - Flexible task sampling for time series data
  
- **Core Components**:
  - Meta-learning based offline training
  - Efficient online adaptation mechanism
  - Support/query set based task creation
  - Customizable model architectures

## Installation

### Using UV (Recommended)

```bash
uv venv
uv pip install .
```

### Using pip

```bash
pip install .
```

## Project Structure

```
doubleadpt/
├── src/
│   ├── dataset.py      # Rolling task sampler for meta-learning
│   ├── handler.py      # Data preprocessing and normalization
│   ├── model.py        # Neural network architectures
│   └── trainner.py     # Double adaptation implementation
├── main.py             # Training script with configuration
├── pyproject.toml      # Project dependencies
└── README.md
```

## Quick Start

```python
from doubleadpt.src.handler import RobustZScoreNorm
from doubleadpt.src.trainner import DoubleAdaptFramework
from doubleadpt.src.dataset import RollingTaskSampler

# Initialize data preprocessing
normalizer = RobustZScoreNorm()
X_norm = normalizer.fit_transform(X)

# Create task sampler
sampler = RollingTaskSampler(
    X_norm, 
    support_size=60,  # Historical context window
    query_size=20     # Prediction window
)

# Initialize and train framework
framework = DoubleAdaptFramework(
    model=your_model,
    criterion=your_loss_function,
    x_dim=feature_dim,
    device=device
)

# Train offline (meta-learning phase)
framework.offline_training(train_tasks, valid_tasks, max_epochs=100)

# Adapt online (deployment phase)
metrics = framework.online_training(valid_tasks, test_tasks)
```

## Command Line Usage

```bash
python main.py --data_path /path/to/your/data.csv --output_dir outputs
```

### Key Arguments

- `--data_path`: Path to input CSV data file (required)
- `--output_dir`: Directory for saving outputs (default: 'outputs')
- `--support_size`: Historical context window size (default: 60)
- `--query_size`: Prediction window size (default: 20)
- `--max_epochs`: Maximum training epochs (default: 100)

## Input Data Format

Required CSV structure:
- `datetime`: Temporal index column
- `label`: Target values column
- Feature columns used for prediction

## Citation

If you use this implementation in your research, please cite:

```bibtex
@inproceedings{zhang2023double,
  title={Double Adaptation for Time Series Forecasting},
  author={Zhang, Xiyuan and Wen, Rui and Wang, Zhe and Li, Bo},
  booktitle={International Conference on Machine Learning},
  year={2023}
}
```

## License

MIT License 