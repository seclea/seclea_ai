# -*- coding: utf-8 -*-

__date__ = 'Mar 15, 2022'
__author__ = 'Melkon Hovhannisyan <Melkon.Hovhanisian@mail.com>'

# Name of the package
__name__ = 'seclea_ai'

# The major proposed change is the introduction of a new module level
# attribute, __package__. When it is present, relative imports will be based on
# this attribute rather than the module __name__ attribute.
__package__ = 'seclea_ai'


#
# TODO: Write introduction for the imported lib
#
from src.core.settings import __PATH__


#
# Documentation of the package
#
__doc__ = f"""Introduction

{open(__PATH__('DOCS') / 'AI/introduction.txt', 'r').read()}

"""


__version__ = 1.0
