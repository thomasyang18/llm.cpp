import tiktoken


# s = "18672 21475 28988 4413 35477 43318 19413"

# tokens = list(map(int, s.split()))

tokens = [int(input())]

# Load the encoding
enc = tiktoken.get_encoding('gpt2')

# Decode the tokens and print the result
decoded_text = enc.decode(tokens)
print(decoded_text)
