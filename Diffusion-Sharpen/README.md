# Diffusion-Sharpen

This repository provides resources for the paper "Diffusion-Sharpen: Fine-tuning Diffusion Models with Denoising Trajectory Sharpening"
![alt text](./figs/fig3.png)

![alt text](./figs/fig2.png)




## Introduction

![alt text](./figs/fig1.png)
Diffusion-Sharpening is a trajectory-level fine-tuning framework for diffusion models that autonomously optimizes sampling paths through reward-guided iterative refinement. By integrating self-alignment with a path integral approach, it eliminates the reliance on predefined datasets and amortizes inference costs into training, achieving superior efficiency in both convergence and inference.


## Installation

```shell
git clone https://github.com/Gen-Verse/Diffusion-Sharpen
cd Diffusion-Sharpen
conda create -n Diffusion-Sharpen python==3.8.10
conda activate Diffusion-Sharpen
pip install -r requirements.txt
```

## SFT-Diffusion-Sharpen Training Command

```shell
bash scripts/sft.sh
```

## RLHF-Diffusion-Sharpen Training Command

```shell
bash scripts/rlhf.sh
```

## Reward Model Configure
Please refer to `reward.py` for more details. If you wish to use your own reward model, you can modify the `reward.py` file. For MLLM Grader, you can set your api key in the training command with `--api_key`.


