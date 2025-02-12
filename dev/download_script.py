# Ran this on a google colab and exported weights from there. 
# The colab has a filesystem, so once this is finished, zip up all the *.npy files and download it locally.

from transformers import GPT2LMHeadModel
import torch
import numpy as np

# Load pretrained model
model = GPT2LMHeadModel.from_pretrained("gpt2")
state_dict = model.state_dict()

# Ignore these buffers
ignored_keys = {k for k in state_dict if k.endswith(('.attn.bias', '.attn.masked_bias'))}

# What the fuck? Why did karpathy even do this? 
# The original attention mechanism makes so much more sense, jesus fucking christ.
# The issue with ML is that, there *is* an intuitive sense for what these matrices are doing, 
# but nobody actually bothers to explain what's going on. 
# Maybe I'm just a stupid slow kid? Like jesus christ.
# This is just horrible, horrible, un-intuitive data munging. 

# Layers that need transposition (Conv1D to Linear)
transposed_layers = {"attn.c_attn.weight", "attn.c_proj.weight", "mlp.c_fc.weight", "mlp.c_proj.weight"}

# Save tensors as numpy arrays
for key, tensor in state_dict.items():
    if key in ignored_keys:
        continue  # Skip masked biases

    # Handle transposed weights
    if any(key.endswith(layer) for layer in transposed_layers):
        tensor = tensor.T  # Transpose if necessary

    filename = f"{key}.npy"
    np.save(filename, tensor.numpy())