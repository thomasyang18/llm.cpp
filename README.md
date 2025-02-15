# GPT-2 Implementation in C++ for CPU Inference

This repository contains an implementation of the GPT-2 language model in C++. The project aims to provide a basic understanding of how large language models work by implementing the core components of GPT-2. Additionally, I want to implement some basic passes such as: 

- **K-V caching**

- **Flash Attention**

### Dependencies

- Eigen 
- cnpy

## Running the project (for colab or anything etc.)

load model weights into /dev/model_weights, just run download script in the /dev directory (in colab)

(or if running locally and don't want to install torch or something beefy, just load just the download script in a colab, zip up the files, and manually copy paste it in)

run `make clean` then `make -j 8`

then ./run.sh should just work out of the box. 

(cnpy needs to be dynamically linked :/, by default its in /usr/local/lib after its install script)

---

### This is likely to be discontinued :/ 

There are some interesting things I can do here, and the more I've learned about transformers, the more appreciative of how insane they are 

(they're theroretically infinite context, if you replace `wpe` with something like sine waves or RoPE!)

but idk. Maybe I can transition this to a big mono-repo where I can freely explore ML projects. 