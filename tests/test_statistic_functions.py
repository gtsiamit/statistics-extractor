import sys
from pathlib import Path
import numpy as np
from scipy.stats import mode, skew, kurtosis
from scipy.stats._stats_py import ModeResult

sys.path.append(str(Path(__file__).parent.parent / "statistics_extractor"))

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
    calculate_statistic,
)


# ----- CALCULATE STATISTICS Test -----
def test_calculate_statistic():
    # mean 1d
    x = np.array([10, 2, 16, 4])
    assert calculate_statistic(x, stat_func=np.mean) == 8

    # mean 2d
    x = np.array(
        [
            [1, 2, 3, 4],
            [10, 12, 14, 16],
            [12, 10, 16, 14],
            [10, 2, 16, 4],
            [2, 4, 8, 12],
        ]
    )
    np.testing.assert_array_equal(
        calculate_statistic(x, stat_func=np.mean), np.array([7, 6, 11.4, 10])
    )

    # median 1d
    x = np.array([10, 2, 16, 4])
    assert calculate_statistic(x, stat_func=np.median) == 7

    # median 2d
    x = np.array(
        [
            [1, 2, 3, 4],
            [10, 12, 14, 16],
            [12, 10, 16, 14],
            [10, 2, 16, 4],
            [2, 4, 8, 12],
        ]
    )
    np.testing.assert_array_equal(
        calculate_statistic(x, stat_func=np.median), np.array([10, 4, 14, 12])
    )

    # mode 1d
    x = np.array([10, 2, 16, 4, 10])
    assert calculate_statistic(x, stat_func=mode) == ModeResult(
        mode=np.int64(10), count=np.int64(2)
    )

    # mode 2d
    x = np.array(
        [
            [1, 2, 3, 4],
            [10, 12, 14, 16],
            [12, 10, 16, 14],
            [10, 2, 16, 4],
            [2, 4, 8, 12],
        ]
    )
    result = calculate_statistic(x, stat_func=mode)
    np.testing.assert_array_equal(result.mode, np.array([10, 2, 16, 4]))
    np.testing.assert_array_equal(result.count, np.array([2, 2, 2, 2]))

    # variance 1d
    x = np.array([10, 2, 16, 4, 10])
    assert calculate_statistic(x, stat_func=np.var) == 24.64

    # variance 2d
    x = np.array(
        [
            [1, 2, 3, 4],
            [10, 12, 14, 16],
            [12, 10, 16, 14],
            [10, 2, 16, 4],
            [2, 4, 8, 12],
        ]
    )
    np.testing.assert_array_equal(
        calculate_statistic(x, stat_func=np.var), np.array([20.8, 17.6, 26.24, 25.6])
    )

    # standard deviation 1d
    x = np.array([10, 2, 16, 4, 10])
    assert round(calculate_statistic(x, stat_func=np.std), 2) == 4.96

    # standard deviation 2d
    x = np.array(
        [
            [1, 2, 3, 4],
            [10, 12, 14, 16],
            [12, 10, 16, 14],
            [10, 2, 16, 4],
            [2, 4, 8, 12],
        ]
    )
    np.testing.assert_array_equal(
        np.round(calculate_statistic(x, stat_func=np.std), 2),
        np.array([4.56, 4.20, 5.12, 5.06]),
    )

    # 25th percentile 1d
    x = np.array([10, 2, 16, 4, 10])
    assert calculate_statistic(x, stat_func=np.percentile, q=25) == 4

    # 25th percentile 2d
    x = np.array(
        [
            [1, 2, 3, 4],
            [10, 12, 14, 16],
            [12, 10, 16, 14],
            [10, 2, 16, 4],
            [2, 4, 8, 12],
        ]
    )
    np.testing.assert_array_equal(
        calculate_statistic(x, stat_func=np.percentile, q=25),
        np.array([2.0, 2.0, 8.0, 4.0]),
    )

    # 75th percentile 1d
    x = np.array([10, 2, 16, 4, 10])
    assert calculate_statistic(x, stat_func=np.percentile, q=75) == 10

    # 75th percentile 2d
    x = np.array(
        [
            [1, 2, 3, 4],
            [10, 12, 14, 16],
            [12, 10, 16, 14],
            [10, 2, 16, 4],
            [2, 4, 8, 12],
        ]
    )
    np.testing.assert_array_equal(
        calculate_statistic(x, stat_func=np.percentile, q=75),
        np.array([10.0, 10.0, 16.0, 14.0]),
    )

    # skewness 1d
    x = np.array([10, 2, 16, 4, 10])
    assert round(calculate_statistic(x, stat_func=skew), 2) == 0.16

    # skewness 2d
    x = np.array(
        [
            [1, 2, 3, 4],
            [10, 12, 14, 16],
            [12, 10, 16, 14],
            [10, 2, 16, 4],
            [2, 4, 8, 12],
        ]
    )
    np.testing.assert_array_equal(
        np.round(calculate_statistic(x, stat_func=skew), 2),
        np.array([-0.34, 0.39, -0.62, -0.22]),
    )

    # kurtosis 1d
    x = np.array([10, 2, 16, 4, 10])
    assert round(calculate_statistic(x, stat_func=kurtosis), 2) == -1.22

    # kurtosis 2d
    x = np.array(
        [
            [1, 2, 3, 4],
            [10, 12, 14, 16],
            [12, 10, 16, 14],
            [10, 2, 16, 4],
            [2, 4, 8, 12],
        ]
    )
    np.testing.assert_array_equal(
        np.round(calculate_statistic(x, stat_func=kurtosis), 2),
        np.array([-1.75, -1.66, -1.24, -1.73]),
    )


# ----- MEAN Test-----
def test_calculate_mean():

    # 1d array
    x = np.array([10, 2, 16, 4])
    assert calculate_mean(x) == 8

    # 1d array, 1 value
    x = np.array([10])
    assert calculate_mean(x) == 10

    # 2d array
    x = np.array(
        [
            [1, 2, 3, 4],
            [10, 12, 14, 16],
            [12, 10, 16, 14],
            [10, 2, 16, 4],
            [2, 4, 8, 12],
        ]
    )
    np.testing.assert_array_equal(calculate_mean(x), np.array([7, 6, 11.4, 10]))

    # 2d array, 1 value per row, vector
    x = np.array([[1], [10], [12], [10], [2]])
    assert calculate_mean(x) == 7


# ----- MEDIAN Test-----
def test_calculate_median():

    # 1d array
    x = np.array([10, 2, 16, 4])
    assert calculate_median(x) == 7

    # 1d array, 1 value
    x = np.array([10])
    assert calculate_median(x) == 10

    # 2d array
    x = np.array(
        [
            [1, 2, 3, 4],
            [10, 12, 14, 16],
            [12, 10, 16, 14],
            [10, 2, 16, 4],
            [2, 4, 8, 12],
        ]
    )
    np.testing.assert_array_equal(calculate_median(x), np.array([10, 4, 14, 12]))

    # 2d array, 1 value per row, vector
    x = np.array([[1], [10], [12], [10], [2]])
    assert calculate_median(x) == 10


# ----- MODE Test-----
def test_calculate_mode():

    # 1d array
    x = np.array([10, 2, 16, 4, 10])
    assert calculate_mode(x) == 10

    # 1d array, 1 value
    x = np.array([10])
    assert calculate_mode(x) == 10

    # 2d array
    x = np.array(
        [
            [1, 2, 3, 4],
            [10, 12, 14, 16],
            [12, 10, 16, 14],
            [10, 2, 16, 4],
            [2, 4, 8, 12],
        ]
    )
    np.testing.assert_array_equal(calculate_mode(x), np.array([10, 2, 16, 4]))

    # 2d array, 1 value per row, vector
    x = np.array([[1], [10], [12], [10], [2]])
    assert calculate_mode(x) == 10


# ----- VARIANCE Test-----
def test_calculate_variance():

    # 1d array
    x = np.array([10, 2, 16, 4, 10])
    assert calculate_variance(x) == 24.64

    # 1d array, 1 value
    x = np.array([10])
    assert calculate_variance(x) == 0

    # 2d array
    x = np.array(
        [
            [1, 2, 3, 4],
            [10, 12, 14, 16],
            [12, 10, 16, 14],
            [10, 2, 16, 4],
            [2, 4, 8, 12],
        ]
    )
    np.testing.assert_array_equal(
        calculate_variance(x), np.array([20.8, 17.6, 26.24, 25.6])
    )

    # 2d array, 1 value per row, vector
    x = np.array([[1], [10], [12], [10], [2]])
    assert calculate_variance(x) == 20.8


# ----- STANDARD DEVIATION Test-----
def test_calculate_std():

    # 1d array
    x = np.array([10, 2, 16, 4, 10])
    assert round(calculate_std(x), 2) == 4.96

    # 1d array, 1 value
    x = np.array([10])
    assert calculate_std(x) == 0

    # 2d array
    x = np.array(
        [
            [1, 2, 3, 4],
            [10, 12, 14, 16],
            [12, 10, 16, 14],
            [10, 2, 16, 4],
            [2, 4, 8, 12],
        ]
    )
    np.testing.assert_array_equal(
        np.round(calculate_std(x), 2), np.array([4.56, 4.20, 5.12, 5.06])
    )

    # 2d array, 1 value per row, vector
    x = np.array([[1], [10], [12], [10], [2]])
    assert np.round(calculate_std(x), 2) == 4.56


# ----- 25th PERCENTILE Test-----
def test_calculate_25th_perc():

    # 1d array
    x = np.array([10, 2, 16, 4, 10])
    assert calculate_25th_perc(x) == 4.0

    # 1d array, 1 value
    x = np.array([10])
    assert calculate_25th_perc(x) == 10.0

    # 2d array
    x = np.array(
        [
            [1, 2, 3, 4],
            [10, 12, 14, 16],
            [12, 10, 16, 14],
            [10, 2, 16, 4],
            [2, 4, 8, 12],
        ]
    )
    np.testing.assert_array_equal(
        calculate_25th_perc(x), np.array([2.0, 2.0, 8.0, 4.0])
    )

    # 2d array, 1 value per row, vector
    x = np.array([[1], [10], [12], [10], [2]])
    assert calculate_25th_perc(x) == 2.0


# ----- 75th PERCENTILE Test-----
def test_calculate_75th_perc():

    # 1d array
    x = np.array([10, 2, 16, 4, 10])
    assert calculate_75th_perc(x) == 10.0

    # 1d array, 1 value
    x = np.array([10])
    assert calculate_75th_perc(x) == 10.0

    # 2d array
    x = np.array(
        [
            [1, 2, 3, 4],
            [10, 12, 14, 16],
            [12, 10, 16, 14],
            [10, 2, 16, 4],
            [2, 4, 8, 12],
        ]
    )
    np.testing.assert_array_equal(
        calculate_75th_perc(x), np.array([10.0, 10.0, 16.0, 14.0])
    )

    # 2d array, 1 value per row, vector
    x = np.array([[1], [10], [12], [10], [2]])
    assert calculate_75th_perc(x) == 10.0


# ----- SKEWNESS Test-----
def test_calculate_skewness():

    # 1d array
    x = np.array([10, 2, 16, 4, 10])
    assert round(calculate_skewness(x), 2) == 0.16

    # 1d array, 1 value
    x = np.array([10])
    assert np.isnan(calculate_skewness(x))

    # 2d array
    x = np.array(
        [
            [1, 2, 3, 4],
            [10, 12, 14, 16],
            [12, 10, 16, 14],
            [10, 2, 16, 4],
            [2, 4, 8, 12],
        ]
    )
    np.testing.assert_array_equal(
        np.round(calculate_skewness(x), 2), np.array([-0.34, 0.39, -0.62, -0.22])
    )

    # 2d array, 1 value per row, vector
    x = np.array([[1], [10], [12], [10], [2]])
    assert np.round(calculate_skewness(x), 2) == -0.34


# ----- KURTOSIS Test-----
def test_calculate_kurtosis():

    # 1d array
    x = np.array([10, 2, 16, 4, 10])
    assert round(calculate_kurtosis(x), 2) == -1.22

    # 1d array, 1 value
    x = np.array([10])
    assert np.isnan(calculate_kurtosis(x))

    # 2d array
    x = np.array(
        [
            [1, 2, 3, 4],
            [10, 12, 14, 16],
            [12, 10, 16, 14],
            [10, 2, 16, 4],
            [2, 4, 8, 12],
        ]
    )
    np.testing.assert_array_equal(
        np.round(calculate_kurtosis(x), 2), np.array([-1.75, -1.66, -1.24, -1.73])
    )

    # 2d array, 1 value per row, vector
    x = np.array([[1], [10], [12], [10], [2]])
    assert np.round(calculate_kurtosis(x), 2) == -1.75
