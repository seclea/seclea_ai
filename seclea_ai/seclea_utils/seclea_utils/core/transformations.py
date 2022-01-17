import base64
import marshal
import types
from typing import Dict, List


def encode_func(func, args: List = None, kwargs: Dict = None):
    if args is None:
        args = list()
    if kwargs is None:
        kwargs = dict()
    return base64.b64encode(
        marshal.dumps({"function": func.__code__, "args": args, "kwargs": kwargs})
    ).decode("ascii")


def decode_func(func):
    args = None
    kwargs = None
    try:
        dic = marshal.loads(base64.b64decode(func))  # nosec
        function = dic["function"]
        f = types.FunctionType(function, globals(), "transformation1")
    except Exception as e:
        f = e
    else:
        try:
            args = dic["args"]
            kwargs = dic["kwargs"]
        except KeyError:
            raise ValueError("Function must contain either args or kwargs")
    return f, args, kwargs
