# <img src="../resources/logo.webp" style="vertical-align: -10px;" :height="50px" width="50px"> Divide, Conquer and Combine: A Training-Free Framework for High-Resolution Image Perception in Multimodal Large Language Models

# Contents
- [Overview](#overview)
- [Install](#install)
- [$DC^2$ Demo](#demo)

# Overview
We propose Divide, Conquer and Combine (DC$^2$), a novel training-free framework for enhancing MLLM perception of HR images. DC$^2$ follows a three-staged approach: 1) Divide: recursively partitioning the HR image into patches and merging similar patches to minimize computational overhead, 2) Conquer: leveraging the MLLM to generate accurate textual descriptions for each image patch, and 3) Combine: utilizing the generated text descriptions to enhance the MLLM's understanding of the overall HR image. Extensive experiments show that: 1) the SOTA MLLM achieves 63% accuracy, which is markedly lower than the 87% accuracy achieved by humans on HR-Bench; 2) our DC$^2$ brings consistent and significant improvements (a relative increase of +6% on HR-Bench and +8% on general multimodal benchmarks). The benchmark and code will be released to facilitate the multimodal R&D community.

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
