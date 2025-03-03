from typing import Union
import numpy as np
import pandas as pd
from .handler import handle_extraction


def extract_statistics(
    x: Union[np.ndarray, pd.Series, pd.DataFrame], feature_names: list = None
) -> pd.DataFrame:
    """
    Extracts statistical information from the input data. This function processes the input data
    using `handle_extraction` to compute relevant statistics and returns them as a Pandas DataFrame.

    Args:
        x (Union[np.ndarray, pd.Series, pd.DataFrame]): The input data, which can be a NumPy array, Pandas Series, or Pandas DataFrame.
        feature_names (list, optional): A list of feature names to use as column names.
                                        If not provided, column names are generated dynamically.

    Returns:
        pd.DataFrame: A pandas DataFrame containing the extracted statistics.
    """

    # Call the handler
    extracted_statistics = handle_extraction(x=x, cols=feature_names)

    return extracted_statistics
