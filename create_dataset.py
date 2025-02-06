import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def create_synthetic_dataset(n_samples: int = 1000, n_features: int = 5, seed: int = 42) -> pd.DataFrame:
    """
    Create a synthetic time series dataset.
    
    Args:
        n_samples: Number of samples to generate
        n_features: Number of features to generate
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with datetime index, features, and target variable
    """
    np.random.seed(seed)
    
    # Create datetime index
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(hours=i) for i in range(n_samples)]
    
    # Create base signals
    t = np.linspace(0, 10, n_samples)
    
    # Generate features with different patterns
    features = {}
    
    # Feature 1: Sine wave
    features['f1'] = np.sin(t) + np.random.normal(0, 0.1, n_samples)
    
    # Feature 2: Cosine wave
    features['f2'] = np.cos(t) + np.random.normal(0, 0.1, n_samples)
    
    # Feature 3: Linear trend with noise
    features['f3'] = 0.5 * t + np.random.normal(0, 0.1, n_samples)
    
    # Feature 4: Exponential pattern
    features['f4'] = np.exp(t/10) / np.exp(1) + np.random.normal(0, 0.1, n_samples)
    
    # Feature 5: Random walk
    random_walk = np.random.normal(0, 0.1, n_samples)
    features['f5'] = np.cumsum(random_walk)
    
    # Create target variable (combination of features with noise)
    y = (0.3 * features['f1'] + 
         0.2 * features['f2'] + 
         0.15 * features['f3'] + 
         0.25 * features['f4'] + 
         0.1 * features['f5'] + 
         np.random.normal(0, 0.1, n_samples))
    
    # Create DataFrame
    df = pd.DataFrame(features)
    df['datetime'] = dates
    df['label'] = y
    
    # Normalize features and target
    for col in df.columns:
        if col != 'datetime':
            df[col] = (df[col] - df[col].mean()) / df[col].std()
    
    return df

if __name__ == "__main__":
    # Create dataset
    df = create_synthetic_dataset()
    
    # Save to CSV
    output_path = "synthetic_data.csv"
    df.to_csv(output_path, index=False)
    print(f"Dataset saved to {output_path}")
    print(f"Dataset shape: {df.shape}")
    print("\nFirst few rows:")
    print(df.head())