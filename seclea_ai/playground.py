from __future__ import annotations
from seclea_ai.transformations import DatasetTransformation
from typing import Callable, Dict, Any
import functools


def transform_manager(function=None, project: int = None, **kwargs) -> Any:
    """
    """

    def _decorate(f):
        dt = DatasetTransformation(f=f)

        @functools.wraps(f)
        def transformation(**t_kwargs: Any) -> Dict:
            # enforce kwargs only for transformation (could allow args as well)
            return dt(**t_kwargs)

        return transformation

    if function:
        return _decorate(function)
    return _decorate


project = 5


@transform_manager(project=project)
def f1(a=1, b=1):
    return a + b


def f2(a=1, b=2):
    return a * b


f2 = transform_manager(f2)
c = f1(a=5)
c = f2(a=c)

print(c)
