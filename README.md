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
│   ├── double_adapt/
│   │   ├── dataset.py      # Rolling task sampler for meta-learning
│   │   ├── handler.py      # Data preprocessing and normalization
│   │   ├── model.py        # Neural network architectures
│   │   └── trainner.py     # Double adaptation implementation
│   ├── tlob/               # TLOB model implementation
│   │   ├── tlob.py        # TLOB architecture
│   │   └── mlplob.py      # MLP-based LOB model
│   └── train_tool.py       # Main training script with CLI interface
├── pyproject.toml          # Project dependencies
└── README.md
```

## Quick Start with Command Line Interface

The main training script (`src/train_tool.py`) provides a comprehensive CLI for training the DoubleAdapt framework with either TLOB (Transformer) or MLPLOB (MLP-based) models.

```bash
python src/train_tool.py \
  --data_path /path/to/your/data.csv \
  --model_type tlob \
  --price_column mid_price \
  --label_horizon 30 \
  --sequence_length 30 \
  --hidden_dim 64 \
  --num_layers 2 \
  --support_length 900 \  # 15 minutes (15 * 60)
  --query_length 300      # 5 minutes (5 * 60)
```

### Key Arguments

- Data Configuration:
  - `--data_path`: Path to input data file (CSV/Parquet)
  - `--price_column`: Name of the price column (default: 'mid_price')
  - `--label_horizon`: Prediction horizon for trend labels (default: 30)
  - `--smooth_window`: Smoothing window for label calculation (default: 5)

- Model Configuration:
  - `--model_type`: Choose between 'tlob' or 'mlplob' (default: 'tlob')
  - `--hidden_dim`: Hidden dimension size (default: 64)
  - `--num_layers`: Number of model layers (default: 2)
  - `--num_heads`: Number of attention heads for TLOB (default: 4)
  - `--sequence_length`: Input sequence length (default: 30)

- Training Configuration:
  - `--support_length`: Historical context window (default: 900)
  - `--query_length`: Prediction window (default: 300)
  - `--interval`: Rolling interval for sampling (default: 5)
  - `--max_epochs`: Maximum training epochs (default: 10)
  - `--patience`: Early stopping patience (default: 5)
  - `--lr_theta`: Learning rate for forecast model (default: 0.001)
  - `--lr_da`: Learning rate for data adapter (default: 0.01)

## Input Data Format

Supported file formats:
- CSV files (*.csv)
- Parquet files (*.parquet)

Required structure:
- Temporal features (e.g., price data, technical indicators)
- Target column for prediction

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
