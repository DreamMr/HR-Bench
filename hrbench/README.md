# Evaluation Guidlines

## Environment

You should prepare the environment as follows:
```
conda create -n hrbench python=3.10
conda activate hrbench

cd hrbench
pip install -r requirements.txt
```

## Evaluation

We develop an easy-to-use evaluation pipeline based on the [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) repository. This pipeline is built to assess the accuracy of different MLLMs using our HR-Bench benchmark.

We use the evaluation of InternVL-Chat-V1.5 as an example. 

```
#---------scripts/run.sh----------------
#!/bin/bash

export LMUData=~
export llm_path=YOUR_LLM_PATH # use to extract final answer
export TOKENIZERS_PARALLELISM=true
cd ../
torchrun --nproc-per-node=4 run.py --data HRBench4K --model InternVL-Chat-V1-5
```