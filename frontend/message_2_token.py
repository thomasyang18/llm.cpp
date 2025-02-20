import tiktoken

# Load the GPT-2 encoding
enc = tiktoken.get_encoding('gpt2')

# Take input as a string
text = input("Enter text: ")

# Encode the text into token IDs
tokens = enc.encode(text)

tokens.append(50256) # eos

# Print the result as a C++ vector
print("\nCopy-paste this into C++:\n")
print(f"std::vector<int> tokens = {{{', '.join(map(str, tokens))}}};")

# Also print encoded tokens for verification
print("\nEncoded token IDs:\n", tokens)

"""
Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.
"""