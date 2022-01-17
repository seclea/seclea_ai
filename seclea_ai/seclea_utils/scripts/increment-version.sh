#!/usr/bin/env bash

version=$(grep version setup.py | cut -d '"' -f2)

bumpversion "$1" --current-version "$version" setup.py
