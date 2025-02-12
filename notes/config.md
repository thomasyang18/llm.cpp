# Config

## Smol GPT

```python
@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
```

We use [cnpy](https://github.com/rogersce/cnpy) to serialize/parse stuff. 

The weights are all float32.

We use `tiktoken` and bind to python, because downloading weights was already a pain...

We expose the C++ with pybind. 

