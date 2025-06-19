### 0. About

This document records all the steps needed to install packages related to reproducing the results in `ReinFlow: Fine-tuning Flow Matching Policy with Online Reinforcement Learning`. 


### 1. Configure pip source, conda source channels, and create environment
* Pip and conda sources
You are free to skip this part if you are satisfied with the speed of python package downloading. 
Otherwise, if you want to use THU TUNA as the conda source, please refer to Section 1. in [reinflow-setup-verbose.md](reinflow-setup-verbose.md). Other sources are also acceptable. 

* Create environment
```bash
# create environment
conda create -n reinflow python=3.8 -y
conda activate reinflow
# rename your repo folder as ReinFlow
# (omitted)
```


### 2. Setup Environment for OpenAI Gym
Here we list one possible approach to install mujoco_py. 

```bash
# make mujoco directory
mkdir $HOME/.mujoco
cd ~/.mujoco 
# download mujoco210 and mujoco-py from this source, or other places you like
https://drive.google.com/drive/folders/15fcrWlTCwFxZkxpMPSrcNzCfTxjj8WE4?usp=sharing
# unzip them to your root directory
unzip ./mujoco210.zip
unzip ./mujoco-py.zip
# add link to mujoco
echo -e '# link to mujoco
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia 
export PATH="$LD_LIBRARY_PATH:$PATH" ' >> ~/.bashrc
source ~/.bashrc
conda activate reinflow
# install cython
pip install 'cython<3.0.0' -i https://pypi.tuna.tsinghua.edu.cn/simple

# if you don't have root privilege or cannot update the driver (e.g. in a container)
conda install -c conda-forge glew
conda install -c conda-forge mesalib
conda install -c menpo glfw3
echo -e 'CPATH=$CONDA_PREFIX/include' >> ~/.bashrc
source ~/.bashrc
# else, if you have root privilege: 
sudo apt-get install patchelf
sudo apt-get install python3-dev build-essential libssl-dev libffi-dev libxml2-dev  
sudo apt install libosmesa6-dev libgl1-mesa-glx libglfw3

# install mujoco-py
cd ~/.mujoco/mujoco-py
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install -r requirements.dev.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
pip3 install -e . --no-cache -i https://pypi.tuna.tsinghua.edu.cn/simple
# test mujoco-py installation 
python3
import mujoco_py # there should be no import errors. 
dir(mujoco_py)   # you should see a lot of methods if you successfully installed mujoco_py. 
```

#### [Debug Helper] If you don't have root privilege, or meet the error '#include <GL/glew.h>'...
```bash
    4 | #include <GL/glew.h>
      |          ^~~~~~~~~~~
Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.
```
You can still install mujoco without using sudo commands by the following bash commands, according to https://github.com/openai/mujoco-py/issues/627
```bash
conda install -c conda-forge glew
conda install -c conda-forge mesalib
conda install -c menpo glfw3
echo -e 'CPATH=$CONDA_PREFIX/include' >> ~/.bashrc
source ~/.bashrc
```

#### [Debug Helper] If you meet this error: version `GLIBCXX_3.4.30‘ not found...
```bash
# link GLIBCXX_3.4.30. reference: https://blog.csdn.net/L0_L0/article/details/129469593
strings /usr/lib/x86_64-linux-gnu/libstdc++.so.6 | grep GLIBCXX # 1. first check if the thing exists: 
# you should be ablt to see GLIBCXX_3.4 ~ GLIBCXX_3.4.30
cd <PATH_TO_YOUR_ANACONDA3>/envs/reinflow/bin/../lib/ # 2. then create soft links
mv libstdc++.so.6 libstdc++.so.6.old
ln -s /usr/lib/x86_64-linux-gnu/libstdc++.so.6 libstdc++.so.6
```


### 3. Setup environment for Franka Kitchen
```bash
# install mjrl
# down load mjrl from https://github.com/aravindr93/mjrl to mjrl-master.zip
cd ~/ReinFlow/
unzip mjrl-master.zip
rm mjrl-master.zip
cd ./mjrl-master
pip install -e .

# install D4RL, DeepMind control suite, and MuJoCo
pip install d4rl
pip install dm_control==1.0.16 mujoco==3.1.6         # for franka kitchen
```

### 4. Setup environment for Robomimic
```bash
# install robomimic==0.3.0. Warning: this action lets you download torch and cudann, be careful of the space consumption. 
pip install robomimic==0.3.0

# install robosuite==1.4.1
sudo apt-get install libegl-dev # to prevent eglQueryString bug.
pip install robosuite==1.4.1 

python <PATH_TO_YOUR_ANACONDA3>/envs/reinflow/lib/python3.8/site-packages/robosuite/scripts/setup_macros.py # add links
# then you will see this "copied $HOME/anaconda3/envs/reinflow/lib/python3.8/site-packages/robosuite/macros.py to $HOME/anaconda3/envs/reinflow/lib/python3.8/site-packages/robosuite/macros_private.py"
```

#### [Debug Helper] if you see `libEGL warning: failed to open /dev/dri/renderD135: Permission denied'
If you are in a container and not an administrator, it is okay to see this warning, but it means you can only use osmesa to train robomimic instead of egl, 
which is often three times slower than using EGL rendering. 

### 5. Install dependencies for visualization and logging 
These packages are just for visualization. Feel free to skip them if you are only interested in training. 
```bash
pip install wandb==0.17.7 tqdm==4.66.5 pandas==2.0.3 pypdf>=3.9.0 seaborn==0.13.2
```

### 6.Set environment paths
These commands specify the paths to your logging directory, data folders and create verbose debugging information, which is beneficial for development. 
```bash
# set REINFLOW paths
cd ~/ReinFlow 
conda activate reinflow
bash ./script/set_path.sh  # set log, data, and env paths. # wandb entity: 
source ~/.bashrc
```

### 7.Install ReinFlow package
```bash
# install reinflow package
conda activate reinflow
pip install -e .
```

### 8. Download the visualize folder (optional)
If you want to obtain the training record of our experiments, consider download the visualize folder from our repository and put the contents in your `<REINFLOW_DIR>/visualize`




