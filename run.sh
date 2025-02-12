#!/bin/bash

# run.sh
# Make script executable with: chmod +x run.sh

# Directory containing cnpy library - adjust these paths as needed
CNPY_LIB_DIR="$HOME/local/lib"
EIGEN_INCLUDE_DIR="/usr/local/include/eigen3"

# Build the project with library path
export CPLUS_INCLUDE_PATH="$EIGEN_INCLUDE_DIR:$CPLUS_INCLUDE_PATH"
make clean && make

# Check if make was successful
if [ $? -ne 0 ]; then
    echo "Build failed!"
    exit 1
fi

# Check if weights directory was provided
if [ -z "$1" ]; then
    echo "Usage: $0 <path_to_weights_dir>"
    exit 1
fi

# Run the program with proper library path
export LD_LIBRARY_PATH="$CNPY_LIB_DIR:$LD_LIBRARY_PATH"
./bin/gpt2_weight_loader "$1"

# Reset LD_LIBRARY_PATH (optional, since script environment won't persist anyway)
# export LD_LIBRARY_PATH="$OLD_LD_LIBRARY_PATH"
