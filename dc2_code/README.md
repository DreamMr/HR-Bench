# <img src="../resources/logo.webp" style="vertical-align: -10px;" :height="50px" width="50px"> Divide, Conquer and Combine: A Training-Free Framework for High-Resolution Image Perception in Multimodal Large Language Models

# Contents
- [Overview](#overview)
- [Install](#install)
- [$DC^2$ Demo](#demo)

# Overview
We propose Divide, Conquer and Combine (DC$^2$), a novel training-free framework for enhancing MLLM perception of HR images. DC$^2$ follows a three-staged approach: 1) Divide: recursively partitioning the HR image into patches and merging similar patches to minimize computational overhead, 2) Conquer: leveraging the MLLM to generate accurate textual descriptions for each image patch, and 3) Combine: utilizing the generated text descriptions to enhance the MLLM's understanding of the overall HR image.

# Install
```
pip install requirement.txt
```

# $DC^2$ Demo
To run a demo of the project, execute the following command:
```
python main.py
```
We also provide a Jupyter Notebook [demo](./main.ipynb). You can view and interact with it.
