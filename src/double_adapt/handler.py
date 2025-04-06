import numpy as np
import pickle

class RobustZScoreNorm:
    """
    Robust Z-Score Normalization using median and median absolute deviation (MAD)
    
    Attributes:
        clip_outlier (bool): whether to clip outliers to [-3, 3]
        median_ (np.ndarray): learned median values
        mad_ (np.ndarray): learned median absolute deviation values
        fitted (bool): whether the scaler is fitted
    """
    def __init__(self, clip_outlier=True):
        self.clip_outlier = clip_outlier
        self.median_ = None
        self.mad_ = None
        self.fitted = False
        
    def fit(self, X):
        """Compute median and MAD"""
        self.median_ = np.nanmedian(X, axis=0)
        self.mad_ = np.nanmedian(np.abs(X - self.median_), axis=0) * 1.4826  # 1.4826 = 1/norm.ppf(3/4)
        self.fitted = True 
        
    def transform(self, X):
        """Apply normalization"""
        if not self.fitted:
            raise ValueError("RobustZScoreNorm is not fitted yet.")
            
        X_norm = (X - self.median_) / (self.mad_ + 1e-8)  # avoid division by zero
        
        if self.clip_outlier:
            X_norm = np.clip(X_norm, -3, 3)
            
        return X_norm
        
    def fit_transform(self, X):
        """Fit and transform in one step"""
        self.fit(X)
        return self.transform(X)
        
    def inverse_transform(self, X):
        """Inverse transform normalized data back to original scale"""
        if not self.fitted:
            raise ValueError("RobustZScoreNorm is not fitted yet.")
            
        return X * self.mad_ + self.median_
        
    def is_fitted(self):
        """Check if scaler is fitted"""
        return self.fitted

    def save(self, path):
        """Save scaler parameters"""
        params = {
            'median': self.median_,
            'mad': self.mad_,
            'fitted': self.fitted
        }
        with open(path, 'wb') as f:
            pickle.dump(params, f)
        
    def load(self, path):
        """Load scaler parameters"""
        with open(path, 'rb') as f:
            params = pickle.load(f)
        self.median_ = params['median']
        self.mad_ = params['mad']
        self.fitted = params['fitted']
