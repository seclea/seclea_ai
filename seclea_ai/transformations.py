from typing import Callable, Dict, List, Tuple


class DatasetTransformation:
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
        return self._data_kwargs
