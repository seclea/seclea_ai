pwd
poetry env use python3.10
if [[ $(uname -m) == 'arm64' ]]; then
  echo "========[ arm64 ]==========="
  brew install openssl
  brew install libomp
  brew install gcc@11
  export GRPC_PYTHON_BUILD_SYSTEM_OPENSSL=1
  export GRPC_PYTHON_BUILD_SYSTEM_ZLIB=1
  export LDFLAGS="-L/opt/homebrew/opt/openssl@3/lib"
  export CPPFLAGS="-I/opt/homebrew/opt/openssl@3/include"

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
