import numpy as np
from statistics_extractor.extractor import extract_statistics


def main():

    # create a numpy array
    array = np.array([[1, 2, 3], [4, 5, 6], [4, 5, 6], [10, 12, 15]])
    print(f"Original data:\n", array, "\n")

    # call the extract_statistics function
    result = extract_statistics(x=array, feature_names=["f1", "f2", "f3"])

    # print the output
    print("Statistics extractor output:\n", result)


if __name__ == "__main__":
    main()
