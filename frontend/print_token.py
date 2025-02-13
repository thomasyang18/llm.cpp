import tiktoken


# s = "18672 21475 28988 4413 35477 43318 19413"

s =  """11703
  28988
  4413
  35477
  43777
  11703
  35141
  30742
  35541
  34900
  43318
  1958
  28988
  26325
  43318
  18793
  15910
  26325
  26325
  26325
  26325
  26325
  27604
  23022
  43318
  35541
  37679"""

#tokens = list(map(int, s.split()))

# tokens = [int(input())]

tokens = [15496, 11, 314, 1101, 257, 3303, 2746, 13, 314, 1101, 407, 257, 2746, 329, 661, 284, 1382, 319, 13, 843, 314, 1101, 407, 1016, 284, 1382, 281, 2104, 2746]

# Load the encoding
enc = tiktoken.get_encoding('gpt2')

# Decode the tokens and print the result
decoded_text = enc.decode(tokens)
print(decoded_text)
