from typing import Callable, Dict, List


class DatasetTransformation:
    def __init__(self, func: Callable, kwargs: Dict, outputs: List):
        self.func = func
        self.kwargs = kwargs
        self.outputs = outputs

    def __call__(self) -> Dict:
        # runs the transformation function and returns the results (maybe only the named outputs (in self.outputs))
        outputs = self.func(**self.kwargs)
        # bind outputs
        results = dict()
        for key, val in zip(self.outputs, outputs):
            if key is not None:
                results[key] = val
        return results
