# -*- coding: utf-8 -*-

__name__ = 'Transformations'


from src.core.settings import __PATH__
__doc__ = f""" introduction

    Data Transformation is the process of converting data from one format to
    another format which is useful to business users for decision making. In
    Data management data transformation is a key component of data integration
    and pre-processing. Data transformation involves activities such as data
    discovery, data mapping, filtering, combining data, applying different
    aggregation functions and business logic among other specific tasks. Data
    transformation can be simple or complex depending on the business
    requirement, type of data and the transformation rules to be applied. There
    are different tools that automates this process such as ETL/ELT based
    tools, Programming languages (SQL, Python etc.). In this post we will look
    at data transformation as a pre-processing process, its benefits, types of
    data transformation and the process of data transformation.


    {open(__PATH__('DOCS') / 'DB/Dataset_Transformation.txt').read()}

"""


from typing import Callable, Dict, List, Tuple


class DatasetTransformation:

    __doc__ = f""" Description:





    :param function: Callable the function that modifies the Dataset.

    :param data_kwargs: Dict the data inputs to the function as keyword
                        arguments. If they are outputs of a previous
                        DatasetTransformation in the same series you should use
                        the value "inherit". See the example for more details.

    :param kwargs:  Dict any other keyword arguments to the function. Must be
                    specified even if empty.

    :param outputs: List the name of the outputs. These should be the same as
                    the input names of any DatasetTransformation that follows.
                    See the example for more details.

    :param split:   Optional[str] the split of the dataset. If the function
                    does not split the dataset this may be ignored

    Example::
        {open(__PATH__('EXAMPLES') / 'DatasetTransformation.cls').read()}
    """

    def __init__(
        self, func: Callable, data_kwargs: Dict,
        kwargs: Dict, outputs: List, split: str = None
    ):
        self.func = func
        self._data_kwargs = data_kwargs
        self.kwargs = kwargs
        self.outputs = outputs
        self.split = split

    def __call__(self, previous_output: Dict) -> Dict:
        # preprocess and data substitution.
        self._substitute_data_kwargs(previous_output)
        # runs the transformation function and returns the results (maybe only
        # the named outputs (in self.outputs))
        outputs = self.func(**self._data_kwargs, **self.kwargs)
        outputs = self._process_outputs(outputs)
        return outputs

    def _process_outputs(self, outputs) -> Dict:
        # bind outputs
        results = dict()
        for key, val in zip(
            self.outputs,
            [outputs] if not (isinstance(outputs, Tuple)
                              or isinstance(outputs, List)) else outputs,
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
        """Getter for data_kwargs that empties the data - to prevent uploading
        the data multiple times"""
        emptied_data_kwargs = dict()
        for key, val in self._data_kwargs.items():
            emptied_data_kwargs[key] = key
        return emptied_data_kwargs

    @property
    def raw_data_kwargs(self) -> Dict:
        """Returns the _data_kwargs without emptying them for the cases where
        that is needed."""
        return self._data_kwargs
