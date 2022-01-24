from typing import Callable, Dict, List


class DatasetTransformation:
    def __init__(self, func: Callable, data_kwargs: Dict, kwargs: Dict, outputs: List):
        self.func = func
        self.data_kwargs = data_kwargs
        self.kwargs = kwargs
        self.outputs = outputs

    def __call__(self) -> Dict:
        # runs the transformation function and returns the results (maybe only the named outputs (in self.outputs))
        outputs = self.func(**self.data_kwargs, **self.kwargs)
        # bind outputs
        results = dict()
        # strip out any outputs that have None for the key.
        for key, val in zip(self.outputs, outputs):
            if key is not None:
                results[key] = val
        return results
