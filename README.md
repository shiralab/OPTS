# OPTS
Optimizing Prompts with Strategy Selection (OPTS).<br>
This repo provides the experimental code for the paper "Bandit-Based Prompt Design Strategy Selection Improves Prompt Optimizers".

## Abstract of the paper
Prompt optimization aims to search for effective prompts that enhance the performance of large language models (LLMs).
Although existing prompt optimization methods have discovered effective prompts, they often differ from sophisticated 
prompts carefully designed by human experts. Prompt design strategies, representing best practices for improving prompt 
performance, can be key to improving prompt optimization. Recently, a method termed the Autonomous Prompt Engineering 
Toolbox (APET) has incorporated various prompt design strategies into the prompt optimization process. In APET, the LLM 
is needed to implicitly select and apply the appropriate strategies because prompt design strategies can have negative 
effects. This implicit selection may be suboptimal due to the limited optimization capabilities of LLMs. This paper 
introduces Optimizing Prompts with sTrategy Selection (OPTS), which implements explicit selection mechanisms for prompt 
design. We propose three mechanisms, including a Thompson sampling-based approach, and integrate them into EvoPrompt, a 
well-known prompt optimizer. Experiments optimizing prompts for two LLMs, Llama-3-8B-Instruct and GPT-4o mini, were 
conducted using BIG-Bench Hard. Our results show that the selection of prompt design strategies improves the performance 
of EvoPrompt, and the Thompson sampling-based mechanism achieves the best overall results.


## Experimental Environment
Our experiments were conducted on a computer running Ubuntu 22.04 with an AMD EPYC 7502P CPU and an NVIDIA A100 GPU, and 
on another computer running Ubuntu 22.04 with an AMD EPYC 7702P CPU and an NVIDIA A100 GPU. We used openai 1.40.8 as the 
python library to access GPT-4o mini, and vllm 0.6.3.post1 as the python library to access llama-3-8B-Instruct.


## Preparation
### Installation
```command1
pip install .
```

### Login to Hugging Face
```command2
huggingface-cli login --token [YOUR_ACCESS_TOKEN]
```
If you have not applied to use Llama-3-8B-Instruct in the Hugging Face, please apply first.

### Add OpenAI API key
```command3
export OPENAI_API_KEY=[YOUR_OPENAI_API_KEY]
```


## Quick Start
The following command is used to optimize the task descriptions fed into the llama-3-8B-Instruct using EvoPrompt(DE)-OPTS(TS).

### Command
```run
cd expt
bash script/optimize_prompt_llama/EvoPromptDE-OPTS_TS.sh
```

## Acknowledgements
### Code
Our code is based on the following papers and repos.
1. Qingyan Guo, Rui Wang, Junliang Guo, Bei Li, Kaitao Song, Xu Tan, Guoqing Liu, Jiang Bian, and Yujiu Yang. 2024. 
Connecting large language models with evolutionary algorithms yields powerful prompt optimizers. In The Twelfth 
International Conference on Learning Representations.
2. https://github.com/beeevita/EvoPrompt
3. Daan Kepel and Konstantina Valogianni. 2024. Autonomous prompt engineering in large language models. Preprint, arXiv:2407.11000.
4. https://github.com/daankepel/APET
5. https://github.com/EleutherAI/lm-evaluation-harness

### Dataset
The dataset BBH, included in the "dataset" directory, is based on the following paper and repo:
1. Mirac Suzgun, Nathan Scales, Nathanael Schärli, Sebastian Gehrmann, Yi Tay, Hyung Won Chung, Aakanksha Chowdhery, Quoc V. Le, Ed H. Chi, 
Denny Zhou, and Jason Wei. 2022. Challenging big-bench tasks and whether chain-of-thought can solve them. Preprint, arXiv:2210.09261.
2. https://github.com/suzgunmirac/BIG-Bench-Hard


