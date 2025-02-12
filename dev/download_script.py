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

# Save tensors as numpy arrays
for key, tensor in state_dict.items():
    filename = f"{key}.npy"
    np.save(filename, tensor.numpy())