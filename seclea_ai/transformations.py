from typing import Callable, Dict, List, Tuple


class DatasetTransformation:
    """
    A Dataset Transformation object.

    :param func: Callable the function that modifies the Dataset.

    :param data_kwargs: Dict the data inputs to the function as keyword arguments. If they are outputs of a previous
        DatasetTransformation in the same series you should use the value "inherit". See the example for more details.

    :param kwargs: Dict any other keyword arguments to the function. Must be specified even if empty.

    :param outputs: List the name of the outputs. These should be the same as the input names of any
        DatasetTransformation that follows. See the example for more details.

    :param split: Optional[str] the split of the dataset. If the function does not split the dataset this may be ignored

    Example::
        >>> import pandas as pd
        >>>
        >>> def drop_correlated(data, thresh):
        >>>     import numpy as np
        >>>
        >>>     # calculate correlations
        >>>     corr_matrix = data.corr().abs()
        >>>     # get the upper part of correlation matrix
        >>>     upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
        >>>
        >>>     # columns with correlation above threshold
        >>>     redundant = [column for column in upper.columns if any(upper[column] >= thresh)]
        >>>     print(f"Columns to drop with correlation > {thresh}: {redundant}")
        >>>     data.drop(columns=redundant, inplace=True)
        >>>     return data
        >>>
        >>> def drop_nulls(df, threshold):
        >>>     cols = [x for x in df.columns if df[x].isnull().sum() / df.shape[0] > threshold]
        >>>     return df.drop(columns=cols)
        >>>
        >>> sample_df = pd.read_csv("some_data.csv")
        >>> corr_threshold = 0.97
        >>> sample_df1 = drop_correlated(data=sample_df, thresh=corr_threshold)
        >>>
        >>> null_threshold = 0.9
        >>> sample_df1 = drop_nulls(df=sample_df1, threshold=null_threshold)
        >>>
        >>> transformations = [
        >>>     DatasetTransformation(
        >>>         drop_correlated, {"data": sample_df}, {"thresh": corr_threshold}, ["df"]
        >>>     ),
        >>>     DatasetTransformation(
        >>>         drop_nulls, {"df": "inherit"}, {"threshold": null_threshold}, ["df"]
        >>>     ),


    """

    def __init__(
        self, func: Callable, data_kwargs: Dict, kwargs: Dict, outputs: List, split: str = None
    ):
        self.func = func
        self._data_kwargs = data_kwargs
        self.kwargs = kwargs
        self.outputs = outputs
        self.split = split

    def __call__(self, previous_output: Dict) -> Dict:
        # preprocess and data substitution.
        self._substitute_data_kwargs(previous_output)
        # runs the transformation function and returns the results (maybe only the named outputs (in self.outputs))
        outputs = self.func(**self._data_kwargs, **self.kwargs)
        outputs = self._process_outputs(outputs)
        return outputs

    def _process_outputs(self, outputs) -> Dict:
        # bind outputs
        results = dict()
        for key, val in zip(
            self.outputs,
            [outputs] if not (isinstance(outputs, Tuple) or isinstance(outputs, List)) else outputs,
        ):
            # strip out any outputs that have None for the key.
            if key is not None:
                results[key] = val
        return results

    def _substitute_data_kwargs(self, previous_output) -> None:
        for key, val in self._data_kwargs.items():
            if isinstance(val, str) and val == "inherit":
                self._data_kwargs[key] = previous_output[key]

    @property
    def data_kwargs(self) -> Dict:
        """Getter for data_kwargs that empties the data - to prevent uploading the data multiple times"""
        emptied_data_kwargs = dict()
        for key, val in self._data_kwargs.items():
            emptied_data_kwargs[key] = key
        return emptied_data_kwargs

    @property
    def raw_data_kwargs(self) -> Dict:
        """Returns the _data_kwargs without emptying them for the cases where that is needed."""
        return self._data_kwargs
