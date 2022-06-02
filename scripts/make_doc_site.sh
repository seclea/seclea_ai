#!/usr/bin/env bash

# Build Sphinx Documentation
# Run in Env where requirements.txt is active - requires pandoc to be installed
(

  cd docs/examples || exit

  jupyter nbconvert Getting_Started_with_Seclea.ipynb --to markdown

  mv Getting_Started_with_Seclea.md ../source/Getting_Started_with_Seclea.md

)

(
  cd docs || exit
  make clean
  make html

  # Copy favicon.js (generate favicon) into build
  cp static/js/favicon.js build/html/_static/js

  # Copy favicons into build
  cp static/assets/favicon.ico static/assets/favicon-dark-theme.ico build/html/_static/
)
