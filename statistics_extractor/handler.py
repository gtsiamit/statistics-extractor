from statistics_functions import (
    calculate_mean,
    calculate_median,
    calculate_mode,
    calculate_variance,
    calculate_std,
    calculate_25th_perc,
    calculate_75th_perc,
    calculate_skewness,
    calculate_kurtosis,
)
from utils import pd_to_np
import sys
import numpy as np
import pandas as pd
from typing import Union


STATISTICS_NAMES = [
    "mean",
    "median" "mode",
    "variance",
    "standard_deviation",
    "percentile",
    "skewness",
    "kurtosis",
]


def get_statistics(x: np.ndarray) -> Union[np.float64, np.int64, np.ndarray]:
    """
    Computes multiple statistical measures for a given NumPy array.

    Args:
        x (np.ndarray): A NumPy array, which can be either 1D or 2D.

    Returns:
        list[Union[np.float64, np.int64, np.ndarray]]:
            A list containing the following computed statistics:
            - Mean
            - Median
            - Mode
            - Variance
            - Standard Deviation
            - 25th Percentile
            - 75th Percentile
            - Skewness
            - Kurtosis

            If `x` is 1D, each statistic is returned as a single `float` or `int`.
            If `x` is 2D, each statistic is returned as a 1D NumPy array, computed along columns.
    """

    stats_list = [
        calculate_mean(x),
        calculate_median(x),
        calculate_mode(x),
        calculate_variance(x),
        calculate_std(x),
        calculate_25th_perc(x),
        calculate_75th_perc(x),
        calculate_skewness(x),
        calculate_kurtosis(x),
    ]

    return stats_list


def generate_column_names(num_cols: int) -> list:
    """
    Generates a list of column names in the format ["col1", "col2", ..., "colN"].

    Args:
        num_cols (int): The number of column names to generate.

    Returns:
        list: A list of column names in the specified format.
    """

    return list(map("col{}".format, range(1, num_cols + 1)))


def find_number_of_columns(data: np.ndarray) -> int:
    """
    Determines the number of columns in a given NumPy array.
    If the input array is 2D, it returns the number of columns.
    If the input array is 1D, it returns 1.

    Args:
        data (np.ndarray): A NumPy array that can be either 1D or 2D.

    Returns:
        int: The number of columns if the array is 2D. Otherwise, returns 1.
    """

    return data[0].shape[1] if isinstance(data[0], np.ndarray) else 1


def convert_output_to_df(list_output: list, cols: list = None) -> pd.DataFrame:
    """
    Converts a list of statistical outputs into a Pandas DataFrame.
    If column names are not provided, they are generated dynamically based on the number of columns.

    Args:
        list_output (list): A list containing the data to be converted into a DataFrame.
        cols (list, optional): A list of column names. If not provided, column names are generated dynamically.

    Returns:
        pd.DataFrame: A pandas DataFrame with the provided or generated column names and indexed by STATISTICS_NAMES.
    """

    if cols:
        df = pd.DataFrame(list_output, columns=cols, index=STATISTICS_NAMES)
    else:
        num_cols = find_number_of_columns(data=list_output)
        cols = generate_column_names(num_cols=num_cols)
        df = pd.DataFrame(list_output, columns=cols, index=STATISTICS_NAMES)

    return df


def handle_extraction(
    x: Union[np.ndarray, pd.Series, pd.DataFrame], cols: list = None
) -> pd.DataFrame:
    """
    Handles the extraction of statistical information from the given data.
    This function takes a NumPy array, Pandas Series, or Pandas DataFrame,
    converts it into a NumPy array if necessary, computes relevant statistics,
    and returns the results as a Pandas DataFrame.

    Args:
        x (Union[np.ndarray, pd.Series, pd.DataFrame]): The input data, which can be a NumPy array, Pandas Series, or Pandas DataFrame.
        cols (list, optional): A list of column names. If not provided, column names are generated dynamically.

    Returns:
        pd.DataFrame: A pandas DataFrame containing the extracted statistics.

    Raises:
        ValueError: If the input is not a NumPy array, Pandas Series, or Pandas DataFrame.
    """

    if not (
        isinstance(x, np.ndarray)
        or isinstance(x, pd.Series)
        or isinstance(x, pd.DataFrame)
    ):
        raise ValueError(
            f"Expected a NumPy array, a Pandas Series or a Pandas DataFrame but got {type(x).__name__}"
        )
        sys.exit(1)

    if isinstance(x, pd.Series) or isinstance(x, pd.DataFrame):
        x = pd_to_np(x)

    stats = get_statistics(x)

    stats = convert_output_to_df(list_output=stats, cols=cols)

    return stats
