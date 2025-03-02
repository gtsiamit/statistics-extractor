import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent / "statistics_extractor"))

from utils import is_1d_array, is_2d_array, pd_to_np


# Test is_1d_array
def test_is_1d_array():

    # is 1d array
    x = np.array([2, 4, 6, 8, 10, 12])
    assert is_1d_array(x) is True

    # is not 1d array
    x = np.array([[2, 4, 6], [8, 10, 12]])
    assert is_1d_array(x) is False


# Test is_2d_array
def test_is_2d_array():

    # is 2d array
    x = np.array([2, 4, 6, 8, 10, 12])
    assert is_2d_array(x) is False

    # is not 2d array
    x = np.array([[2, 4, 6], [8, 10, 12]])
    assert is_2d_array(x) is True


def test_pd_to_np():

    # series to array
    x = pd.Series([2, 4, 6, 8, 10, 12])
    np.testing.assert_array_equal(pd_to_np(x), np.array([2, 4, 6, 8, 10, 12]))

    # 1 column dataframe to array
    x = pd.DataFrame([2, 4, 6, 8, 10, 12])
    np.testing.assert_array_equal(
        pd_to_np(x), np.array([[2], [4], [6], [8], [10], [12]])
    )

    # 2d dataframe to array
    x = pd.DataFrame([[2, 4, 6], [8, 10, 12]])
    np.testing.assert_array_equal(pd_to_np(x), np.array([[2, 4, 6], [8, 10, 12]]))
