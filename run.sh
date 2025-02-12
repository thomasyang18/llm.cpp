#!/bin/bash

# run.sh
# Make script executable with: chmod +x run.sh

# Directory containing cnpy library - adjust these paths as needed
# We load this at runtime, I guess? I mean like, its such a small library that it prolly doesn't matter.

# Build the project with library path
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
./bin/gpt2_weight_loader "$1"

# Reset LD_LIBRARY_PATH (optional, since script environment won't persist anyway)
# export LD_LIBRARY_PATH="$OLD_LD_LIBRARY_PATH"
