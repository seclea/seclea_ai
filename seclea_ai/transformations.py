from typing import Callable, Dict, List


class DatasetTransformation:
    def __init__(self, func: Callable, data_kwargs: Dict, kwargs: Dict, outputs: List):
        self.func = func
        self._data_kwargs = data_kwargs
        self.kwargs = kwargs
        self.outputs = outputs

    def __call__(self) -> Dict:
        # runs the transformation function and returns the results (maybe only the named outputs (in self.outputs))
        outputs = self.func(**self._data_kwargs, **self.kwargs)
        # bind outputs
        results = dict()
        # strip out any outputs that have None for the key.
        for key, val in zip(self.outputs, outputs):
            if key is not None:
                results[key] = val
        return results

    @property
    def data_kwargs(self) -> Dict:
        """Getter for data_kwargs that empties the data - to prevent uploading the data multiple times"""
        emptied_data_kwargs = dict()
        for key, val in self._data_kwargs.items():
            emptied_data_kwargs[key] = key
        return emptied_data_kwargs
