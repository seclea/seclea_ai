# -*- coding: utf-8 -*-

"""
    src.__init__
    ~~~~~~~~~~~~

    Python is having special type of methods called magic methods named with
    preceded and double underscores.

    if we want to talk about magic method __new__ then obviously will also need
    to talk about __init__ method. The magic method __new__ will be called when
    instance is being created.where as __init__ method will be called to
    initialize instance when you are creating instance.

"""

from src.__version__ import (
    # Call author of the package
    __author__,

    # Call the package version
    __version__,

    # Call created date
    __date__,

    # Name of the package
    __name__,

    # Global package name
    __package__,

    # The package description
    __doc__,
)

from .core import scelea_ai as Seclea_AI
