# <img src="resources/logo.webp" style="vertical-align: -10px;" :height="50px" width="50px"> Divide, Conquer and Combine: A Training-Free Framework for High-Resolution Image Perception in Multimodal Large Language Models

[**🤗 Dataset**](https://huggingface.co/datasets/DreamMr/HR-Bench) | [**📖 Paper**](http://arxiv.org/abs/2408.15556)

This repo contains the official code and dataset for the paper "[Divide, Conquer and Combine: A Training-Free Framework for High-Resolution Image Perception in Multimodal Large Language Models](http://arxiv.org/abs/2408.15556)"

## 💡 Highlights
- 🔥 We introduce **_HR-Bench_** to systematically evaluate the perception ability of MLLMs in high-resolution (8K resolution) images.
- 🔥 We propose a training-free framework **$DC^2$** to effectively enhance the MLLM's perceive ability on high-resolution images.

## 📜 News
**[2025.02.08]**  🚀 HRBench has been supported in the [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) repository.

**[2024.12.10]**   🥳 Our work was accepted by AAAI 2025.

**[2024.09.09]**  🚀 HRBench has been supported in the [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) repository.

**[2024.08.29]** 🚀 We released the [ArXiv paper](http://arxiv.org/abs/2408.15556).

**[2024.08.23]** 🚀 [Huggingface Dataset](https://huggingface.co/datasets/DreamMr/HR-Bench) and $DC^2$ code are available!


## 👀 Introduction

### **HR-Bench**

We find that the highest resolution in existing multimodal benchmarks is only 2K. To address the current lack of high-resolution multimodal benchmarks, we construct **_HR-Bench_**. **_HR-Bench_** consists two sub-tasks: **_Fine-grained Single-instance Perception (FSP)_** and **_Fine-grained Cross-instance Perception (FCP)_**. The **_FSP_** task includes 100 samples, which includes tasks such as attribute recognition, OCR, visual prompting. The **_FCP_** task also comprises 100 samples which encompasses map analysis, chart analysis and spatial relationship assessment. We visualize examples of our **_HR-Bench_**.👇

<img src="resources/case_study_dataset_13.png">


**_HR-Bench_** is available in two versions: **_HR-Bench 8K_** and **_HR-Bench 4K_**. The **_HR-Bench 8K_** includes images with an average resolution of 8K. Additionally, we manually annotate the coordinates of objects relevant to the questions within the 8K image and crop these image to 4K resolution.

### <img src="resources/logo.webp" style="vertical-align: -10px;" :height="25px" width="25px">  Divide, Conquer and Combine

We observe that most current MLLMs (e.g., LLaVA-v1.5) perceive images in a fixed resolution (e.g., $336\times336$). This simplification often leads to greater visual information loss. Based on this finding, we propose a novel training-free framework —— **D**ivide, **C**onquer and **C**ombine (**$DC^2$**). We  first recursively split an image into image patches until they reach the resolution defined by the pretrained vision encoder (e.g., $336\times 336$), merging similar patches for efficiency (**Divide**). Next, we utilize MLLM to generate text description for each image patch and extract objects mentioned in the text descriptions (**Conquer**). Finally, we filter out hallucinated objects resulting from image division and store the coordinates of the image patches which objects appear (**Combine**). During the inference stage, we retrieve the related image patches according to the user prompt to provide accurate text descriptions.

<img src="resources/framework_version_8.png">

## 🏆 Mini-Leaderboard

We show a mini-leaderboard here and please find more information in our paper. (👏🏻Any new results are welcome. Please add your results and model/paper links through an issue or pull request.)

| Model | **_HR-Bench 4K_** (**_Acc._**) | **_HR-Bench 8K_** (**_Acc._**) | **_Avg._** |
|-------|:--------:|:--------:|:-------:|
|Human Baseline 🥇 | **82.0** | **86.8** | **84.4** |
|Qwen3-VL 32B (instruct) 🥈|  84.6    |   81.6     |   83.0     |
|Qwen3-VL 30B-A3B (instruct) 🥉 |   82.5    |  79.3  |  80.9   |
|MiMo-VL-SFT 7B |    69.4     |   67.8    |  68.6   |
|InternVL-2-llama3-76B w/ our $DC^2$ | 70.4 | 63.3 | 66.9 |
|Qwen2VL-7B | 66.8 | 66.5 | 66.6 |
|InternVL-2-llama3-76B | 71.0 | 61.4 | 66.2 |
|GPT4o (gpt-4o-2024-05-13) | 68.0 | 63.9 | 66.0 |
|Gemini 1.5 Flash | 66.8 | 62.8 | 64.8 |
|InternVL-1.5-26B w/ $DC^2$ | 63.4 | 61.3 | 62.3 |
|Qwen2VL-2B | 64.0 | 58.6 | 61.3 |
|InternVL-1.5-26B | 60.6 | 57.9 | 59.3 |
|QWen-VL-max | 58.5 | 52.5 | 55.5 |
|Xcomposer2-4kHD-7B | 57.8 | 51.3 | 54.6 |
|LLaVA-HR-X-13B | 53.6 | 46.9 | 50.3 |
|LLaVA-1.6-34B | 52.9 | 47.4 | 50.2 |
|QWen-VL-plus | 53.0 | 46.5 | 49.8 |
|LLaVA-HR-X-7B | 52.0 | 41.6 | 46.8 |


## 📧 Contact
- Wenbin Wang: wangwenbin97@whu.edu.cn 

## ✒️ Citation
```
@inproceedings{hrbench,
  title={Divide, conquer and combine: A training-free framework for high-resolution image perception in multimodal large language models},
  author={Wang, Wenbin and Ding, Liang and Zeng, Minyan and Zhou, Xiabin and Shen, Li and Luo, Yong and Yu, Wei and Tao, Dacheng},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={8},
  pages={7907--7915},
  year={2025}
}
```

## Acknowledgement
- This work is built upon the [VLMEvalKit](https://github.com/open-compass/VLMEvalKit)
- We sincerely thank [Ailin Huang](https://github.com/P2Oileen) for providing the GPT-4o (gpt-4o-2024-05-13) results for our benchmark.
