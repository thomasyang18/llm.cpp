# Attention

## Why Project Back?

ChatGPT OP jesus.

Self-attention doesn’t just output the same shape as the input—it temporarily expands the dimensionality before needing to bring it back.

    Input tokens start as embeddings of shape:
        [batch_size, seq_len, n_embd]
    The Q, K, V projections expand this into 3 separate vectors:
        [batch_size, seq_len, n_heads, head_dim] (where n_heads * head_dim = n_embd)
    After self-attention, the heads are concatenated:
        [batch_size, seq_len, n_heads * head_dim]
    
    **This is still n_embd, but it's a different learned representation.**

Now, before we add it back to the residual connection, we need to ensure it has the same format as the original input embeddings.

Cracked. 