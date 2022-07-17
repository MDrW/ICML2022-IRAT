# ICML2022-IRAT
This repository implements IRAT which is the codes of the paper "Individual Reward Assisted Multi-Agent Reinforcement Learning" (https://proceedings.mlr.press/v162/wang22ao/wang22ao.pdf).

This codes are implemented based on the repository https://github.com/marlbenchmark/on-policy.

## Train
This repository contains the experiments in MPE (continue and discrete), Multiwalker (SISL) and StarCraftII (SC2.4.10).

There are two ways to conduct an experiment and train.

1. The bash scripts of these experiments are presented in directory "irat_code/scripts". We can run these bash scripts directly to start an experiments. 

2. We also can run the files in directory"irat_code/scripts/train" use python3. 

Adding or changing the parameters in the .sh files or in command line will conduct different experiments and get different results. The parameters in .sh files are default parameters used in experiments of paper "Individual Reward Assisted Multi-Agent Reinforcement Learning".

## Citation
If you find this repository useful, please cite our paper:

@inproceedings{DBLP:conf/icml/WangZHWZGHLF22,
  author    = {Li Wang and
               Yupeng Zhang and
               Yujing Hu and
               Weixun Wang and
               Chongjie Zhang and
               Yang Gao and
               Jianye Hao and
               Tangjie Lv and
               Changjie Fan},
  editor    = {Kamalika Chaudhuri and
               Stefanie Jegelka and
               Le Song and
               Csaba Szepesv{\'{a}}ri and
               Gang Niu and
               Sivan Sabato},
  title     = {Individual Reward Assisted Multi-Agent Reinforcement Learning},
  booktitle = {International Conference on Machine Learning, {ICML} 2022, 17-23 July
               2022, Baltimore, Maryland, {USA}},
  series    = {Proceedings of Machine Learning Research},
  volume    = {162},
  pages     = {23417--23432},
  publisher = {{PMLR}},
  year      = {2022},
  url       = {https://proceedings.mlr.press/v162/wang22ao.html}
}
