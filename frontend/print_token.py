import tiktoken

enc = tiktoken.get_encoding('gpt2')
print(enc.decode_single_token_bytes(int(input())).decode('utf-8', errors='replace'))