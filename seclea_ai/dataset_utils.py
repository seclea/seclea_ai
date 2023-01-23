from typing import Dict, List

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from pandas.errors import ParserError


"""
Code ref for coercable check

# TODO make sure that this matches the pre-processing.
#  if we auto-convert for analyses then this need to be able to coerce
#  if we don't then we need to only consider the original dtype
try:
    uniques = np.asfarray(uniques)
except ValueError:
    # not coercable to numeric (float) so must be categorical
    return len(uniques)
else:
    # it is numeric so we need to check the distribution to see if regression or not.
    print("numeric coercable")
    pass
"""


def assemble_dataset(data: Dict[str, DataFrame]) -> DataFrame:
    """
    Assembles a single DataFrame from component parts eg. X, y or samples, labels that are both pd compatible
    :param data:
    :return:
    """
    if len(data) == 1:
        return next(iter(data.values()))
    elif len(data) == 2:
        # create dataframe from X and y and upload - will have one item in metadata, the output_col
        for key, val in data.items():
            if not (isinstance(val, DataFrame) or isinstance(val, Series)):
                data[key] = DataFrame(val)
        return pd.concat([x for x in data.values()], axis=1)
    else:
        raise AssertionError(
            "Output doesn't match the requirements. Please review the documentation."
        )


def add_required_metadata(metadata: Dict, required_spec: Dict) -> Dict:
    """
    Adds required - non user specified fields to the metadata
    @param metadata: The metadata dict
    @param required_spec:
    @return: metadata
    """
    for required_key, default in required_spec.items():
        metadata[required_key] = default
    return metadata


def ensure_required_metadata(metadata: Dict, defaults_spec: Dict) -> Dict:
    """
    Ensures that required metadata that can be specified by the user are filled.
    @param metadata: The metadata dict
    @param defaults_spec:
    @return: metadata
    """
    for required_key, default in defaults_spec.items():
        try:
            if metadata[required_key] is None:
                metadata[required_key] = default
        except KeyError:
            metadata[required_key] = default
    return metadata


def aggregate_dataset(datasets: List[str], index) -> DataFrame:
    """
    Aggregates a list of dataset paths into a single file for upload.
    NOTE the files must be split by row and have the same format otherwise this will fail or cause unexpected format
    issues later.
    :param datasets:
    :return:
    """
    loaded_datasets = [pd.read_csv(dset, index_col=index) for dset in datasets]
    aggregated = pd.concat(loaded_datasets, axis=0)
    return aggregated


def get_dataset_type(dataset: DataFrame) -> str:
    if not np.issubdtype(dataset.index.dtype, np.integer):
        try:
            pd.to_datetime(dataset.index.values)
        except (ParserError, ValueError):  # Can't cnvrt some
            return "tabular"
        return "time_series"
    return "tabular"
