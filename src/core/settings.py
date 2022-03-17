# -*- coding: utf-8 -*-

__name__ = 'settings'

__doc__ = f""" introduction

"""


#
# TODO: why and what needs to be added here
#
from src.core.__settings__.base import __PATH__, __DATA__


# initialize db path for the local storage
def DB_PATH() -> 'pathlib.PosixPath':
    """ TODO: Description

        description body
    """

    __path__ = __DATA__() / str(__import__('os').getenv('DB_PATH',
                                                        'database/json'))

    return __path__
