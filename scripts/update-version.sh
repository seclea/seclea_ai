#!/usr/bin/env bash

sed -i "s|$(grep -m 1 __version__ __version__.py)|__version__ = \"$1\"| " __version__.py
