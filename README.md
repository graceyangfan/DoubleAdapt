# DoubleAdapt

A refined implementation of the double adaptation framework for time series prediction, based on the paper ["Double Adaptation for Time Series Forecasting"](https://arxiv.org/abs/2306.09862) (ICML 2023).

## Overview

DoubleAdapt is a sophisticated framework for time series forecasting that addresses distribution shifts through:

1. **Offline Meta-Learning**: Pre-trains on historical tasks to learn transferable patterns
2. **Online Adaptation**: Rapidly adapts to new data distributions using meta-learned initialization

## Core Features

### Advanced Data Processing

#### Large-Scale Data Management
- **Efficient File Formats**
  - Native support for CSV and Parquet files
  - Smart chunking for large datasets
  - Progressive data loading with memory optimization

- **Intelligent Data Streaming**
  - LRU caching mechanism for frequent sequences
  - Configurable cache sizes and sequence lengths
  - Memory-efficient batch processing

#### Smart Data Processing
- **Trend Label Generation**
  - High-pass filtering for price movements
  - Automated trend detection and labeling
  - Configurable prediction horizons

- **Feature Engineering**
  - Robust Z-score normalization
  - Efficient feature scaling for large datasets
  - Automatic feature selection

### Model Architecture

#### Multiple Model Support
- **TLOB (Transformer-based)**
  - Attention-based architecture for complex patterns
  - Configurable number of heads and layers
  - Optional sinusoidal embeddings

- **MLPLOB (MLP-based)**
  - Lightweight architecture for fast inference
  - Adjustable hidden dimensions
  - Efficient parameter management

- **GRULOB (GRU-based)**
  - Sequential pattern learning
  - Configurable dropout and layer count
  - Memory-efficient implementation

### Training Framework

#### Double Adaptation Process
- **Offline Training Phase**
  - Meta-learning across historical tasks
  - Efficient parameter optimization
  - Early stopping with validation monitoring
  
- **Online Adaptation Phase**
  - Rapid adaptation to new distributions
  - Support/query set based adaptation
  - Progressive parameter updates

#### Performance Optimization
- **Memory Management**
  - Efficient gradient accumulation
  - Smart tensor cleanup
  - Automatic resource management

- **Training Efficiency**
  - Parallel data loading
  - Batch size optimization
  - GPU memory optimization

## Technical Details

### Core Components

```
src/
├── double_adapt/
│   ├── dataset.py        # Efficient data loading and processing
│   ├── handler.py        # Data preprocessing and normalization
│   ├── loss_accumulator.py # Memory-efficient loss computation
│   ├── model.py          # Neural network architectures
│   └── trainer.py        # Double adaptation implementation
├── models/
│   ├── tlob.py          # Transformer-based model
│   ├── mlplob.py        # MLP-based model
│   └── gru.py           # GRU-based model
└── train_tool.py         # Main training script
```

### Key Parameters

#### Data Processing
- `chunk_size`: Size of data chunks for processing (default: 3600*4)
- `sequence_length`: Length of input sequences (default: 10)
- `support_length`: Support set window size (default: 3600*2)
- `query_length`: Query set window size (default: 3600)
- `cache_size`: LRU cache size for sequences (default: 10)
- `stride`: Stride for sequence creation (default: 1)

#### Training Configuration
- `lr_theta`: Learning rate for forecast model (default: 0.0005)
- `lr_da`: Learning rate for data adapters (default: 0.005)
- `first_order`: Enable first-order approximation
- `adapt_x`: Enable feature adaptation
- `adapt_y`: Enable label adaptation
- `sigma`: Label adapter regularization parameter (default: 1.0)
- `reg`: Regularization coefficient (default: 0.5)

### Memory Optimization

- **TimeSeriesDataset**
  - Progressive sequence loading
  - LRU caching mechanism
  - Efficient row group reading
  - Smart batch processing

- **LossAccumulator**
  - Batch-wise loss computation
  - Memory-efficient gradient accumulation
  - Automatic tensor cleanup
  - Device-aware computation

- **DoubleAdaptFramework**
  - Efficient parameter management
  - Progressive loading of support/query sets
  - Smart resource cleanup
  - Optimized parameter updates

## Quick Start

```bash
python src/train_tool.py \
  --data_path /path/to/data.csv \
  --model_type tlob \
  --sequence_length 30 \
  --support_length 900 \
  --query_length 300 \
  --chunk_size 3600 \
  --cache_size 128
```

## Performance Considerations

1. **Memory Usage**
   - Configure `cache_size` based on available RAM
   - Adjust `chunk_size` for optimal I/O
   - Set appropriate `batch_size` for GPU memory

2. **Processing Speed**
   - Enable parallel processing with `num_workers`
   - Optimize sequence stride for data loading
   - Configure Parquet row group sizes

3. **Training Efficiency**
   - Use first-order approximation for large datasets
   - Enable early stopping for convergence
   - Optimize device placement

## Citation

```bibtex
@inproceedings{zhang2023double,
  title={Double Adaptation for Time Series Forecasting},
  author={Zhang, Xiyuan and Wen, Rui and Wang, Zhe and Li, Bo},
  booktitle={International Conference on Machine Learning},
  year={2023}
}
```
