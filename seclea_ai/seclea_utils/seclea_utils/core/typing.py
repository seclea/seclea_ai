import logging
from io import BytesIO
from typing import Union
from typing.io import IO, BinaryIO

logger = logging.getLogger(__name__)

# Type Definitions #
BytesStream = Union[IO, IO[bytes], BinaryIO, BytesIO]

# Typing utility functions #
# TODO find a better way of recording imports that cant be imported.
_not_importable = set()


def get_type(obj) -> str:
    """
    Get type name from object to avoid dependencies. Credit wandb/client/get_model_manager.py for original code which has been modified.
    :param obj: The object needing to be typed
    :return: (str) String representation of the type.
    """
    name = f"{obj.__class__.__module__}.{obj.__class__.__name__}"
    if name in ["builtins.module", "__builtin__.module"]:
        return obj.__name__
    else:
        return name


def import_module(target, relative_to=None):
    """
    Imports a module.
    Credit wandb/client/get_model_manager.py
    :param target:
    :param relative_to:
    :return:
    """
    target_parts = target.split(".")
    target_depth = target_parts.count("")
    target_path = target_parts[target_depth:]
    target = target[target_depth:]
    fromlist = [target]
    if target_depth and relative_to:
        relative_parts = relative_to.split(".")
        relative_to = ".".join(relative_parts[: -(target_depth - 1) or None])
    if len(target_path) > 1:
        relative_to = ".".join(filter(None, [relative_to]) + target_path[:-1])
        fromlist = target_path[-1:]
        target = fromlist[0]
    elif not relative_to:
        fromlist = []
    mod = __import__(relative_to or target, globals(), locals(), fromlist)
    return getattr(mod, target, mod)


def get_module(name, required=None):
    """
    Return module or None. Absolute import is required.
    Credit wandb/client/get_model_manager.py with some minor modification.
    :param (str) name: Dot-separated module path. E.g., 'scipy.stats'.
    :param (str) required: A string to raise a ValueError if missing
    :return: (module|None) If import succeeds, the module will be returned.
    """
    if name not in _not_importable:
        try:
            return import_module(name)
        except Exception:
            _not_importable.add(name)
            msg = "Error importing optional module {}".format(name)
            if required:
                logger.exception(msg)
    if required is not None and name in _not_importable:
        raise ValueError(required)
