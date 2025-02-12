import tiktoken

enc = tiktoken.get_encoding('gpt2')
text = "Hello, I'm a language model."
tokens = enc.encode(text)

print("Token IDs:", tokens)

print(enc.eot_token)

print("Token Strings:", [enc.decode_single_token_bytes(token).decode('utf-8', errors='replace') for token in tokens])