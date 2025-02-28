import numpy as np
import pandas as pd


def is_1d_array(x: np.ndarray) -> bool:
    """
    Checks if the given NumPy array is 1-dimensional.

    Args:
        x (np.array): A NumPy array of any dimension.

    Returns:
        bool: True if `x` is a 1D array, otherwise False.
    """

    return True if x.ndim == 1 else False


def is_2d_array(x: np.ndarray) -> bool:
    """
    Checks if the given NumPy array is 2-dimensional.

    Args:
        x (np.array): A NumPy array of any dimension.

    Returns:
        bool: True if `x` is a 2D array, otherwise False.
    """

    return True if x.ndim == 2 else False


def pd_to_np(x: pd.DataFrame) -> np.ndarray:
    """
    Converts a pandas DataFrame (or Series) to a NumPy array.

    Args:
        x (pd.DataFrame): The pandas DataFrame or Series to be converted.

    Returns:
        np.ndarray: A NumPy array representation of the input DataFrame or Series.
    """

    return np.array(x)
