#!/usr/bin/env bash

# Build Sphinx Documentation
# Run in Env where requirements.txt is active
(
  cd docs || exit
  make clean
  make html

  # Copy favicon.js (generate favicon) into build
  cp static/js/favicon.js build/html/_static/js

  # Copy favicons into build
  cp static/assets/favicon.ico static/assets/favicon-dark-theme.ico build/html/_static/
)
