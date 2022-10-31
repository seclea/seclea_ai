#!/usr/bin/env bash

sed -i "s|$(grep -m 1 version setup.py)|version = \"$1\"| " setup.py
