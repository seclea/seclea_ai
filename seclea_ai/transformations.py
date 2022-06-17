from typing import Callable, Dict, List, Tuple
import inspect


class DatasetTransformation:
    """ Represents a dataset transformation
    df' = f_transform(df, *args **kwargs)

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


    """

    def __init__(self, f: Callable):
        """
            :param f: Callable a transformation function.
        """
        self._transform: Callable = f
        self._signature: inspect.Signature = inspect.signature(f)
        self._result: Dict = {}
        ...
    @property
    def transform(self) -> Callable:
        return self._transform

    @transform.setter
    def transform(self, f: Callable):
        # sanitize the function to ensure all args are kwargs and return type is provided.
        self._transform = f

    @property
    def result(self) -> Dict:
        return self._result

    def __call__(self, **kwargs) -> Dict:
        outputs = self.transform(**kwargs)
        return outputs
