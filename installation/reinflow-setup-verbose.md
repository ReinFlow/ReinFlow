### 0. About

This document records all the steps needed to install packages related to reproducing the results in `ReinFlow: Fine-tuning Flow Matching Policy with Online Reinforcement Learning`. 


### 1. Configure pip source and conda source channels as THU tuna
```bash
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

#paste these to pip.conf
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
[install]
trusted-host = pypi.tuna.tsinghua.edu.cn
```

```bash
vim ~/.condarc
# copy this to ~/.condarc and then exit.
channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  - conda-forge
  - defaults
show_channel_urls: true
default_channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2
custom_channels:
  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  msys2: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  bioconda: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  menpo: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  pytorch: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  pytorch-lts: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  simpleitk: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  deepmodeling: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/
  nvidia: https://mirrors.sustech.edu.cn/anaconda-extra/cloud
# clear buffer
conda clean -i
conda clean -p
conda clean -a
```

### 6.Login your WandB, GitHub account, and initialize your git repo.
(omitted)

### 7.Setup a VPN if you are blocked
(omitted)