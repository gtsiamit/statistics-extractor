import numpy as np
import pandas as pd
import pytest
from statistics_extractor.extractor import extract_statistics


# test extract_statistics
def test_extract_statistics():

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
    result = extract_statistics(x=x, feature_names=["feat1", "feat2", "feat3", "feat4"])
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
    result = extract_statistics(x=x, feature_names=["col1", "col2", "col3", "col4"])
    pd.testing.assert_frame_equal(result.round(2), expected_result)

    # input list, ValueError should be raised
    with pytest.raises(
        ValueError,
        match="Expected a NumPy array, a Pandas Series or a Pandas DataFrame but got list.",
    ):
        extract_statistics([2, 4, 6, 8, 10])
