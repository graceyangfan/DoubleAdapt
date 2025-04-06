import numba as nb
import numpy as np 
import polars as pl 
from typing import Union

@nb.experimental.jitclass([
    ('length', nb.float64),
    ('f', nb.float64),
    ('a1', nb.float64),
    ('c1', nb.float64),
    ('c2', nb.float64),
    ('c3', nb.float64),
    ('x_prev2', nb.float64),
    ('x_prev1', nb.float64),
    ('hp_prev2', nb.float64),
    ('hp_prev1', nb.float64)
])
class NumbaHighPassFilter:
    def __init__(self, length):
        """
        Initialize high pass filter parameters
        
        Args:
            length (float): Filter length parameter
        """
        # Calculate filter coefficients
        self.length = length
        self.f = 1.414 * np.pi / length
        self.a1 = np.exp(-self.f)
        self.c2 = 2 * self.a1 * np.cos(self.f)
        self.c3 = -self.a1 * self.a1
        self.c1 = (1 + self.c2 - self.c3) / 4

        # Initialize state variables
        self.x_prev2 = 0
        self.x_prev1 = 0
        self.hp_prev2 = 0
        self.hp_prev1 = 0


    def process(self, x_current):
        """
        Process a single data point
        
        Args:
            x_current (float): Current input value
        
        Returns:
            float: High pass filter residual
        """
        # Calculate high pass component
        hp_current = (
            self.c1 * (x_current - 2 * self.x_prev1 + self.x_prev2) + 
            self.c2 * self.hp_prev1 + 
            self.c3 * self.hp_prev2
        )

        # Update states
        self.x_prev2 = self.x_prev1
        self.x_prev1 = x_current
        self.hp_prev2 = self.hp_prev1
        self.hp_prev1 = hp_current

        return hp_current

 
    def filter(self, x_current):
        """
        Get the filtered price
        
        Args:
            x_current (float): Current input value
        
        Returns:
            float: Filtered price
        """
        return x_current - self.process(x_current)


# Wrapper function to apply the filter to a column
def apply_high_pass_filter(
    df: pl.DataFrame, 
    column_name: str, 
    output_name: str,
    filter_length: float
) -> pl.DataFrame:
    """
    Apply the high pass filter to a specified column in the DataFrame.

    Args:
        df (pl.DataFrame): Input DataFrame.
        column_name (str): Name of the column to filter.
        output_name (str): Name of the output column with filtered data.
        filter_length (float): Length parameter for the high pass filter.

    Returns:
        pl.DataFrame: DataFrame with the filtered column added.
    """
    # Convert the column to a numpy array
    column_data = df[column_name].to_numpy()

    # Initialize the filter
    hp_filter = NumbaHighPassFilter(filter_length)

    # Apply the filter to each element using a list comprehension for better performance
    filtered_data = np.array([hp_filter.filter(x) for x in column_data])

    # Add the filtered data as a new column to the DataFrame
    df = df.with_columns(pl.Series(output_name, filtered_data))
    
    return df
