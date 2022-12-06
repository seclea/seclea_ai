pwd
poetry env use python3.10
if [[ $(uname -m) == 'arm64' ]]; then
  echo "macos"
  unset CC
  unset CXX
  poetry install
  export CC=gcc-11
  export CXX=g++-11
  poetry install
else
  poetry install
fi
yarn install
