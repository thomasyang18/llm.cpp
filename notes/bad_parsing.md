# Lol? 2-12-2024

In python code, I see:

```python
# Layers that need transposition (Conv1D to Linear)
transposed_layers = {"attn.c_attn.weight", "attn.c_proj.weight", "mlp.c_fc.weight", "mlp.c_proj.weight"}
```

But like why. #1 this broke all my code. #2, the way the paper does it makes so much more sense, the inputs are the left dimension, output is the right.

Idk why kaparthy swapped it. Probably performance magic reasons.