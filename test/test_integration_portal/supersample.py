import pandas as pd
from pandas import DataFrame


def sample_to_size(dataframe: DataFrame, size: int) -> DataFrame:
    """
    Supersamples a dataframe until it exceeds the specified size. Will be at most double the stated size.
    :param dataframe: The original dataframe
    :param size: The order of magnitude for the resulting dataframe in bytes.
    """
    if dataframe.memory_usage().sum() > size:
        return dataframe
    temp = dataframe.copy()
    while temp.memory_usage().sum() < size:
        temp = pd.concat([temp, temp], axis=0)
    return temp


if __name__ == "__main__":
    source = pd.read_csv("insurance_claims.csv")
    super_sized = sample_to_size(source, size=10 ** 10)
    print(super_sized.memory_usage().sum())
    super_sized.to_csv("supersized.csv")
    print("saved")
