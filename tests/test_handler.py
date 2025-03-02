import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.append(str(Path(__file__).parent.parent / "statistics_extractor"))

from handler import (
    generate_column_names,
    find_number_of_columns,
    get_statistics,
    convert_output_to_df,
    handle_extraction,
)


# test generate_column_names
def test_generate_column_names():

    # 1 column
    assert generate_column_names(1) == ["col1"]

    # 5 columns
    assert generate_column_names(5) == ["col1", "col2", "col3", "col4", "col5"]


# test find_number_of_columns
def test_find_number_of_columns():

    # 1 column
    x = np.array([12, 24, 32, 46, 54])
    assert find_number_of_columns(x) == 1

    # 4 columns
    x = np.array([[12, 24, 32, 26], [8, 4, 2, 6], [534, 524, 332, 426]])
    assert find_number_of_columns(x) == 4


# test get_statistics
def test_get_statistics():

    # get statistics for 1d array
    x = np.array([10, 2, 16, 4, 10])
    result = get_statistics(x)
    assert isinstance(result, list)
    assert len(result) == 9
    assert result[0] == 8.4  # mean
    assert result[1] == 10  # median
    assert result[2] == 10  # mode
    assert result[3] == 24.64  # variance
    assert round(result[4], 2) == 4.96  # standard deviation
    assert result[5] == 4  # 25th percentile
    assert result[6] == 10  # 75th percentile
    assert round(result[7], 2) == 0.16  # skewness
    assert round(result[8], 2) == -1.22  # kurtosis

    # get statistics for 2d array
    x = np.array(
        [
            [1, 2, 3, 4],
            [10, 12, 14, 16],
            [12, 10, 16, 14],
            [10, 2, 16, 4],
            [2, 4, 8, 12],
        ]
    )
    result = get_statistics(x)
    assert isinstance(result, list)
    assert len(result) == 9
    np.testing.assert_array_equal(result[0], np.array([7, 6, 11.4, 10]))  # mean
    np.testing.assert_array_equal(result[1], np.array([10, 4, 14, 12]))  # median
    np.testing.assert_array_equal(result[2], np.array([10, 2, 16, 4]))  # mode
    np.testing.assert_array_equal(
        result[3], np.array([20.8, 17.6, 26.24, 25.6])
    )  # variance
    np.testing.assert_array_equal(
        np.round(result[4], 2), np.array([4.56, 4.20, 5.12, 5.06])
    )  # standard deviation
    np.testing.assert_array_equal(
        result[5], np.array([2.0, 2.0, 8.0, 4.0])
    )  # 25th percentile
    np.testing.assert_array_equal(
        result[6], np.array([10.0, 10.0, 16.0, 14.0])
    )  # 75th percentile
    np.testing.assert_array_equal(
        np.round(result[7], 2), np.array([-0.34, 0.39, -0.62, -0.22])
    )  # skewness
    np.testing.assert_array_equal(
        np.round(result[8], 2), np.array([-1.75, -1.66, -1.24, -1.73])
    )  # kurtosis


# test convert_output_to_df
def test_convert_output_to_df():

    indices = [
        "mean",
        "median",
        "mode",
        "variance",
        "standard_deviation",
        "25th_percentile",
        "75th_percentile",
        "skewness",
        "kurtosis",
    ]

    # 1d with column names
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    pd.testing.assert_frame_equal(
        convert_output_to_df(x, cols=["feat1"]),
        pd.DataFrame(x, columns=["feat1"], index=indices),
    )

    # 1d without column names
    pd.testing.assert_frame_equal(
        convert_output_to_df(x), pd.DataFrame(x, columns=["col1"], index=indices)
    )

    # 2d with column names
    x = [
        np.array([1, 2, 3]),
        np.array([21, 22, 23]),
        np.array([15, 25, 35]),
        np.array([16, 26, 36]),
        np.array([71, 72, 73]),
        np.array([18, 28, 38]),
        np.array([10, 20, 30]),
        np.array([14, 24, 34]),
        np.array([51, 52, 53]),
    ]
    expected_result = pd.DataFrame(
        [
            [1, 2, 3],
            [21, 22, 23],
            [15, 25, 35],
            [16, 26, 36],
            [71, 72, 73],
            [18, 28, 38],
            [10, 20, 30],
            [14, 24, 34],
            [51, 52, 53],
        ],
        columns=["feat1", "feat2", "feat3"],
        index=indices,
    )
    pd.testing.assert_frame_equal(
        convert_output_to_df(x, cols=["feat1", "feat2", "feat3"]), expected_result
    )

    # 2d without column names
    expected_result = pd.DataFrame(
        [
            [1, 2, 3],
            [21, 22, 23],
            [15, 25, 35],
            [16, 26, 36],
            [71, 72, 73],
            [18, 28, 38],
            [10, 20, 30],
            [14, 24, 34],
            [51, 52, 53],
        ],
        columns=["col1", "col2", "col3"],
        index=indices,
    )
    pd.testing.assert_frame_equal(
        convert_output_to_df(x, cols=["col1", "col2", "col3"]), expected_result
    )


# test handle_extraction
def test_handle_extraction():

    indices = [
        "mean",
        "median",
        "mode",
        "variance",
        "standard_deviation",
        "25th_percentile",
        "75th_percentile",
        "skewness",
        "kurtosis",
    ]

    # input pd Series
    x = pd.Series([2, 12, 10, 2, 4])
    expected_result = pd.DataFrame(
        [6, 4, 2, 17.6, 4.2, 2, 10, 0.39, -1.66], columns=["feat1"], index=indices
    )
    result = handle_extraction(x, cols=["feat1"])
    pd.testing.assert_frame_equal(result.round(2), expected_result)

    # input 2d pd dataframe
    x = pd.DataFrame(
        [
            [1, 2, 3, 4],
            [10, 12, 14, 16],
            [12, 10, 16, 14],
            [10, 2, 16, 4],
            [2, 4, 8, 12],
        ]
    )
    expected_result = pd.DataFrame(
        [
            [7, 6, 11.4, 10],
            [10, 4, 14, 12],
            [10, 2, 16, 4],
            [20.8, 17.6, 26.24, 25.6],
            [4.56, 4.20, 5.12, 5.06],
            [2.0, 2.0, 8.0, 4.0],
            [10.0, 10.0, 16.0, 14.0],
            [-0.34, 0.39, -0.62, -0.22],
            [-1.75, -1.66, -1.24, -1.73],
        ],
        columns=["feat1", "feat2", "feat3", "feat4"],
        index=indices,
    )
    result = handle_extraction(x, cols=["feat1", "feat2", "feat3", "feat4"])
    pd.testing.assert_frame_equal(result.round(2), expected_result)

    # input 1d numpy array
    x = np.array([2, 12, 10, 2, 4])
    expected_result = pd.DataFrame(
        [6, 4, 2, 17.6, 4.2, 2, 10, 0.39, -1.66], columns=["col1"], index=indices
    )
    result = handle_extraction(x)
    pd.testing.assert_frame_equal(result.round(2), expected_result)

    # input 2d numpy array
    x = np.array(
        [
            [1, 2, 3, 4],
            [10, 12, 14, 16],
            [12, 10, 16, 14],
            [10, 2, 16, 4],
            [2, 4, 8, 12],
        ]
    )
    expected_result = pd.DataFrame(
        [
            [7, 6, 11.4, 10],
            [10, 4, 14, 12],
            [10, 2, 16, 4],
            [20.8, 17.6, 26.24, 25.6],
            [4.56, 4.20, 5.12, 5.06],
            [2.0, 2.0, 8.0, 4.0],
            [10.0, 10.0, 16.0, 14.0],
            [-0.34, 0.39, -0.62, -0.22],
            [-1.75, -1.66, -1.24, -1.73],
        ],
        columns=["col1", "col2", "col3", "col4"],
        index=indices,
    )
    result = handle_extraction(x, cols=["col1", "col2", "col3", "col4"])
    pd.testing.assert_frame_equal(result.round(2), expected_result)

    # input list, ValueError should be raised
    with pytest.raises(
        ValueError,
        match="Expected a NumPy array, a Pandas Series or a Pandas DataFrame but got list.",
    ):
        handle_extraction([2, 4, 6, 8, 10], cols=["feat1"])
