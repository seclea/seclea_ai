#!/usr/bin/env bash
sleep 30
. /venv/bin/activate
pip3 freeze
python3 --version
ldd --version
rm -rf seclea_ai/lib/seclea_utils/clib/compiled/*
python3 -m unittest discover test/test_integration_portal
