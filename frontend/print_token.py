import tiktoken

tokens = eval(input())

# Load the encoding
enc = tiktoken.get_encoding('gpt2')

# Decode the tokens and print the result
decoded_text = enc.decode(tokens)
print(decoded_text)

# The canonical greedy sample text, for debugging purposes
# (yes the model is really stupid here lmao, its also the same on hugging face online)
res =  [15496, 11, 314, 1101, 257, 3303, 2746, 13, 314, 1101, 257, 3303, 2746, 13, 314, 1101, 257, 3303, 2746, 13, 314, 1101, 257, 3303, 2746, 13, 314, 1101, 257, 3303, 2746, 13, 314, 1101, 257, 3303, 2746, 13, 314, 1101, 257, 3303, 2746, 13, 314, 1101, 257, 3303, 2746, 13, 314, 1101, 257, 3303, 2746, 13, 314, 1101]

print("\nCheck if enc is canonical:", res == tokens)