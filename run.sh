#!/bin/bash

# Shortcut: LD_LIBRARY_PATH="/usr/local/lib:$LD_LIBRARY_PATH"

# If cnpy is not installed on the colab/vm, first install it on the colab/vm

CNPY_LIB_DIR="/usr/local/lib"

# Check if weights directory was provided
if [ -z "$1" ]; then
    echo "Usage: $0 <path_to_weights_dir>"
    exit 1
fi

# Run the program with proper library path
export LD_LIBRARY_PATH="$CNPY_LIB_DIR:$LD_LIBRARY_PATH"
./bin/gpt2 "$1"
