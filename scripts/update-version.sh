#!/usr/bin/env bash

sed -i "s|$(grep version setup.py)|    version=\"$1\",| " setup.py
