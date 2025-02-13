# GPT-2 Implementation in C++ for CPU Inference

This repository contains an implementation of the GPT-2 language model in C++. The project aims to provide a basic understanding of how large language models work by implementing the core components of GPT-2. Additionally, I want to implement some basic passes such as: 
    - **K-V caching**
    - **Kernel Fusion**. Not exactly sure how this will work without a GPU. Maybe I can just rent a GPU and add GPU support... but theoretically this should still be possible since cache hierarchies are a thing too. Well, it's educational at the end of the day regardless.

### Dependencies

- Eigen 
- cnpy

### How to get the weights

I serialized the weights from numpy by downloading them from google colab. My linux partition is extremely small and my PC is small in general, so downloading torch was out of the question :/. Pretty handy. 