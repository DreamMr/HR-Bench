#!/bin/bash
export LMUData=~
export llm_path=YOUR_LLM_PATH # use to extract final answer
export TOKENIZERS_PARALLELISM=true
cd ../
torchrun --nproc-per-node=4 run.py --data HRBench4K --model InternVL-Chat-V1-5