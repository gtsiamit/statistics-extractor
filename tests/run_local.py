import numpy as np
import sys
import os

sys.path.append(
    os.path.join(
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
        "statistics_extractor",
    )
)
from statistics_functions import calculate_mean


def main():

    array = np.array([[1, 2, 3], [4, 5, 6], [4, 5, 6], [10, 12, 15]])

    return


if __name__ == "__main__":
    main()
