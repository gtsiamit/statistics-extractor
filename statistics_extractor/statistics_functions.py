import numpy as np
from scipy.stats import mode, skew, kurtosis
from typing import Union, Callable
from utils import is_1d_array, is_2d_array


STAT_FUNC = {
    "mean": np.mean,
    "median": np.median,
    "mode": mode,
    "variance": np.var,
    "standard_deviation": np.std,
    "percentile": np.percentile,
    "skewness": skew,
    "kurtosis": kurtosis,
}


def calculate_statistic(
    x: np.ndarray,
    stat_func: Callable[[np.ndarray], Union[np.float64, np.int64, np.ndarray]],
    **kwargs  # for any extra keyword arguments
) -> Union[np.float64, np.int64, np.array]:
    """
    Applies a statistical function (e.g., mean, median, mode) to a NumPy array.

    Args:
        x (np.ndarray): A NumPy array, which can be either 1D or 2D.
        stat_func (Callable[[np.ndarray], Union[np.float64, np.int64, np.ndarray]]):
            A function that computes a statistic, such as `np.mean`, `np.median`, or `mode`.
        **kwargs: Additional keyword arguments, such as `q` for percentiles.

    Returns:
        Union[np.float64, np.int64, np.ndarray]:
            - If `x` is 1D, returns the computed statistic as a single value (`float` or `int`).
            - If `x` is 2D, returns a 1D NumPy array containing the statistic for each column.
    """

    if is_1d_array(x):
        return stat_func(x, **kwargs)
    elif is_2d_array(x):
        return stat_func(x, axis=0, **kwargs)


def calculate_mean(x: np.ndarray) -> Union[np.float64, np.int64, np.array]:
    """
    Computes the mean of a NumPy array.

    Args:
        x (np.ndarray): A NumPy array, which can be either 1D or 2D.

    Returns:
        Union[np.float64, np.ndarray]:
            - If `x` is 1D, returns the mean as a single `np.float64` or `np.int64`.
            - If `x` is 2D, returns a 1D NumPy array (`np.ndarray`) containing the mean of each column.
    """

    return calculate_statistic(x=x, stat_func=STAT_FUNC.get("mean"))


def calculate_median(x: np.array) -> Union[np.float64, np.int64, np.array]:
    """
    Computes the median of a NumPy array.

    Args:
        x (np.ndarray): A NumPy array, which can be either 1D or 2D.

    Returns:
        Union[np.float64, np.ndarray]:
            - If `x` is 1D, returns the median as a single `np.float64` or `np.int64`.
            - If `x` is 2D, returns a 1D NumPy array (`np.ndarray`) containing the median of each column.
    """

    return calculate_statistic(x=x, stat_func=STAT_FUNC.get("median"))


def calculate_mode(x: np.array) -> Union[np.float64, np.int64, np.array]:
    """
    Computes the mode of a NumPy array.

    Args:
        x (np.ndarray): A NumPy array, which can be either 1D or 2D.

    Returns:
        Union[np.float64, np.ndarray]:
            - If `x` is 1D, returns the mode as a single `np.float64` or `np.int64`.
            - If `x` is 2D, returns a 1D NumPy array (`np.ndarray`) containing the mode of each column.
    """

    return calculate_statistic(x=x, stat_func=STAT_FUNC.get("mode"))


def calculate_variance(x: np.array) -> Union[np.float64, np.int64, np.array]:
    """
    Computes the variance of a NumPy array.

    Args:
        x (np.ndarray): A NumPy array, which can be either 1D or 2D.

    Returns:
        Union[np.float64, np.ndarray]:
            - If `x` is 1D, returns the variance as a single `np.float64` or `np.int64`.
            - If `x` is 2D, returns a 1D NumPy array (`np.ndarray`) containing the variance of each column.
    """

    return calculate_statistic(x=x, stat_func=STAT_FUNC.get("variance"))


def calculate_std(x: np.array) -> Union[np.float64, np.int64, np.array]:
    """
    Computes the standard deviation of a NumPy array.

    Args:
        x (np.ndarray): A NumPy array, which can be either 1D or 2D.

    Returns:
        Union[np.float64, np.ndarray]:
            - If `x` is 1D, returns the standard deviation as a single `np.float64` or `np.int64`.
            - If `x` is 2D, returns a 1D NumPy array (`np.ndarray`) containing the standard deviation of each column.
    """

    return calculate_statistic(x=x, stat_func=STAT_FUNC.get("standard_deviation"))


def calculate_25th_perc(x: np.array) -> Union[np.float64, np.int64, np.array]:
    """
    Computes the 25th percentile of a NumPy array.

    Args:
        x (np.ndarray): A NumPy array, which can be either 1D or 2D.

    Returns:
        Union[np.float64, np.ndarray]:
            - If `x` is 1D, returns the 25th percentile as a single `np.float64` or `np.int64`.
            - If `x` is 2D, returns a 1D NumPy array (`np.ndarray`) containing the 25th percentile of each column.
    """

    return calculate_statistic(x=x, stat_func=STAT_FUNC.get("percentile"), q=25)


def calculate_75th_perc(x: np.array) -> Union[np.float64, np.int64, np.array]:
    """
    Computes the 75th percentile of a NumPy array.

    Args:
        x (np.ndarray): A NumPy array, which can be either 1D or 2D.

    Returns:
        Union[np.float64, np.ndarray]:
            - If `x` is 1D, returns the 75th percentile as a single `np.float64` or `np.int64`.
            - If `x` is 2D, returns a 1D NumPy array (`np.ndarray`) containing the 75th percentile of each column.
    """

    return calculate_statistic(x=x, stat_func=STAT_FUNC.get("percentile"), q=75)


def calculate_skewness(x: np.array) -> Union[np.float64, np.int64, np.array]:
    """
    Computes the skewness of a NumPy array.

    Args:
        x (np.ndarray): A NumPy array, which can be either 1D or 2D.

    Returns:
        Union[np.float64, np.ndarray]:
            - If `x` is 1D, returns the skewness as a single `np.float64` or `np.int64`.
            - If `x` is 2D, returns a 1D NumPy array (`np.ndarray`) containing the skewness of each column.
    """
    return calculate_statistic(x=x, stat_func=STAT_FUNC.get("skewness"))


def calculate_kurtosis(x: np.array) -> Union[np.float64, np.int64, np.array]:
    """
    Computes the kurtosis of a NumPy array.

    Args:
        x (np.ndarray): A NumPy array, which can be either 1D or 2D.

    Returns:
        Union[np.float64, np.ndarray]:
            - If `x` is 1D, returns the kurtosis as a single `np.float64` or `np.int64`.
            - If `x` is 2D, returns a 1D NumPy array (`np.ndarray`) containing the kurtosis of each column.
    """
    return calculate_statistic(x=x, stat_func=STAT_FUNC.get("kurtosis"))
