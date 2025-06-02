# MIT License

# Copyright (c) 2025 ReinFlow Authors

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import numpy as np
color_dict = {
        'Gaussian': '#228B22',
        'QSM': '#b98c1b',
        'DIPO': '#808080',
        'IDQL': '#000000', 
        'DQL': '#800080',  
        'DRWR': '#1E90B2', 
        'DAWR': "#0984C7", 
        'DPPO': '#F4AD08', 
        'ReinFlow-S (ours)': '#FF2400', 
        'ReinFlow-R (W2=0.01)': "#0FBFE6", 
        'ReinFlow-R (W2=0.1)': "#0F76DD", 
        'ReinFlow-R (W2=1.0)': "#1022E8", 
        'ReinFlow-R (sigma(s,t))': "#FF0000", 
        'ReinFlow-R (sigma(s))': "#990066", 
        'ReinFlow-S (2 steps)': "#3B34FF",
        'ReinFlow-S (1 step)': "#5F1F20",
        'ReinFlow-S, 32 (ours)': '#FF2400', 
        'ReinFlow-S, 64 (ours)': '#FF2400', 
        'ReinFlow-S, 100 (ours)': '#FF2400', 
        'ReinFlow-R (ours)': '#E30DEA', 
        'ReinFlow-beta (ours)': '#1E90FF', 
        'ReinFlow-logitnormal (ours)': '#4B0082', 
        'uniform': '#E30DEA',  
        'beta': "#6E0887",     
        'logitnormal': '#4B0082',
        'FQL': "#0F8EE3"
    }


# Time/step ratios for each environment and method (in seconds/step)
time_step_ratios = {
        'hopper':{ 
            'ReinFlow-R (ours)': 11.715,
            'ReinFlow-S (ours)': 12.290,
            'DPPO': 99.046,
            'FQL': 4.418,            
        },
        'hopper-d4rl':{ 
            'ReinFlow-R (ours)': 11.715,
            'ReinFlow-S (ours)': 12.290,
            'DPPO': 99.046,
            'FQL': 4.418,            
        },
        'walker':{ 
            'ReinFlow-R (ours)': 11.563 ,
            'ReinFlow-S (ours)': 13.019 ,
            'DPPO': 101.915 ,
            'FQL': 5.017,            
        },
        'walker-d4rl':{ 
            'ReinFlow-R (ours)': 11.563 ,
            'ReinFlow-S (ours)': 13.019 ,
            'DPPO': 101.915 ,
            'FQL': 5.017,            
        },
        'ant':{ 
            'ReinFlow-R (ours)': 17.473  ,
            'ReinFlow-S (ours)': 17.734  ,
            'DPPO': 102.012  ,
            'FQL': 5.167 ,            
        },
        'ant-d4rl':{ 
            'ReinFlow-R (ours)': 17.473  ,
            'ReinFlow-S (ours)': 17.734  ,
            'DPPO': 102.012  ,
            'FQL': 5.167 ,            
        },
        'humanoid':{ 
            'ReinFlow-R (ours)': 30.916   ,
            'ReinFlow-S (ours)': 30.529   ,
            'DPPO': 109.566   ,
            'FQL': 5.249  ,            
        },
        'humanoid-d4rl':{ 
            'ReinFlow-R (ours)': 30.916   ,
            'ReinFlow-S (ours)': 30.529   ,
            'DPPO': 109.566   ,
            'FQL': 5.249  ,            
        },
        'can-img': {
            'ReinFlow-R (ours)': 218.061,
            'ReinFlow-S (ours)': 218.061,
            'DPPO': 308.933
        },
        'square-img': {
            'ReinFlow-R (ours)': 313.206,
            'ReinFlow-S (ours)': 313.206,
            'DPPO': 437.830
        },
        'transport-img': {
            'ReinFlow-S (ours)': 662.121,
            'DPPO': 701.712
        },
        'kitchen-complete-v0':{
            'ReinFlow-S (ours)': 26.537 ,
            'DPPO': 83.158,
            'FQL': 5.249
        },
    }
    

method_name_dict = {
        'hopper':[
            {'original_name': 'ppo_reflow_mlp_ta4_td4', 'display_name': 'ReinFlow-R (ours)', 'color': color_dict['ReinFlow-R (ours)']},
            {'original_name': 'ppo_shortcut_mlp_ta4_td4', 'display_name': 'ReinFlow-S (ours)', 'color': color_dict['ReinFlow-S (ours)']},
            {'original_name': 'ppo_diffusion_mlp_ta4_td20_tdf10', 'display_name': 'DPPO', 'color': color_dict['DPPO']},
            {'original_name': 'fql_mlp_ta4_td4', 'display_name': 'FQL', 'color': color_dict['FQL']},
        ],
        'hopper-d4rl':[
            {'original_name': 'ppo_reflow_mlp_ta4_td4_d4rl', 'display_name': 'ReinFlow-R (ours)', 'color': color_dict['ReinFlow-R (ours)']},
            {'original_name': 'ppo_shortcut_mlp_ta4_td4_d4rl', 'display_name': 'ReinFlow-S (ours)', 'color': color_dict['ReinFlow-S (ours)']},
            {'original_name': 'ppo_diffusion_mlp_ta4_td20_d4rl', 'display_name': 'DPPO', 'color': color_dict['DPPO']},
            {'original_name': 'fql_mlp_ta4_td4', 'display_name': 'FQL', 'color': color_dict['FQL']},
        ],
        'walker':[
            {'original_name': 'ppo_reflow_mlp_ta4_td4', 'display_name': 'ReinFlow-R (ours)', 'color': color_dict['ReinFlow-R (ours)']},
            {'original_name': 'ppo_shortcut_mlp_ta4_td4', 'display_name': 'ReinFlow-S (ours)', 'color': color_dict['ReinFlow-S (ours)']},
            {'original_name': 'ppo_diffusion_mlp_ta4_td20_tdf10', 'display_name': 'DPPO', 'color': color_dict['DPPO']},
            {'original_name': 'fql_mlp_ta4_td4', 'display_name': 'FQL', 'color': color_dict['FQL']},
        ],
        'walker-d4rl':[
            {'original_name': 'ppo_reflow_mlp_ta4_td4_d4rl', 'display_name': 'ReinFlow-R (ours)', 'color': color_dict['ReinFlow-R (ours)']},
            {'original_name': 'ppo_shortcut_mlp_ta4_td4_d4rl', 'display_name': 'ReinFlow-S (ours)', 'color': color_dict['ReinFlow-S (ours)']},
            {'original_name': 'ppo_diffusion_mlp_ta4_td20_d4rl', 'display_name': 'DPPO', 'color': color_dict['DPPO']},
            {'original_name': 'fql_mlp_ta4_td4', 'display_name': 'FQL', 'color': color_dict['FQL']},
        ],
        'ant':[
            {'original_name': 'ppo_reflow_mlp_ta4_td4', 'display_name': 'ReinFlow-R (ours)', 'color': color_dict['ReinFlow-R (ours)']},
            {'original_name': 'ppo_shortcut_mlp_ta4_td4', 'display_name': 'ReinFlow-S (ours)', 'color': color_dict['ReinFlow-S (ours)']},
            {'original_name': 'ppo_diffusion_mlp_ta4_td20_tdf10', 'display_name': 'DPPO', 'color': color_dict['DPPO']},
            {'original_name': 'fql_mlp_ta4_td4', 'display_name': 'FQL', 'color': color_dict['FQL']},
        ],
        'ant-d4rl':[
            {'original_name': 'ppo_reflow_mlp_ta4_td4_d4rl', 'display_name': 'ReinFlow-R (ours)', 'color': color_dict['ReinFlow-R (ours)']},
            {'original_name': 'ppo_shortcut_mlp_ta4_td4_d4rl', 'display_name': 'ReinFlow-S (ours)', 'color': color_dict['ReinFlow-S (ours)']},
            {'original_name': 'ppo_diffusion_mlp_ta4_td20_d4rl', 'display_name': 'DPPO', 'color': color_dict['DPPO']},
            {'original_name': 'fql_mlp_ta4_td4', 'display_name': 'FQL', 'color': color_dict['FQL']},
        ],
        'humanoid':[
            {'original_name': 'ppo_reflow_mlp_ta4_td4', 'display_name': 'ReinFlow-R (ours)', 'color': color_dict['ReinFlow-R (ours)']},
            {'original_name': 'ppo_shortcut_mlp_ta4_td4', 'display_name': 'ReinFlow-S (ours)', 'color': color_dict['ReinFlow-S (ours)']},
            {'original_name': 'ppo_diffusion_mlp_ta4_td20_tdf10', 'display_name': 'DPPO', 'color': color_dict['DPPO']},
            {'original_name': 'fql_mlp_ta4', 'display_name': 'FQL', 'color': color_dict['FQL']},
        ],
        'humanoid-d4rl':[
            {'original_name': 'ppo_reflow_mlp_ta4_td4_d4rl', 'display_name': 'ReinFlow-R (ours)', 'color': color_dict['ReinFlow-R (ours)']},
            {'original_name': 'ppo_shortcut_mlp_ta4_td4_d4rl', 'display_name': 'ReinFlow-S (ours)', 'color': color_dict['ReinFlow-S (ours)']},
            {'original_name': 'ppo_diffusion_mlp_ta4_td20_d4rl', 'display_name': 'DPPO', 'color': color_dict['DPPO']},
            {'original_name': 'fql_mlp_ta4', 'display_name': 'FQL', 'color': color_dict['FQL']},
        ],
        'humanoid-regularize-compare':[
            # {'original_name': 'ppo_reflow_mlp_ta4_td4', 'display_name': r'Entropy $\alpha=0.03$', 'color': color_dict['ReinFlow-R (ours)']},
            # {'original_name': 'ppo_reflow_Wdiv_regularize_bc0.01_mlp_ta4_td4', 'display_name': r'Wasserstein $\alpha=0.01$', 'color': color_dict['ReinFlow-R (W2=0.01)']},
            # {'original_name': 'ppo_reflow_Wdiv_regularize_bc0.1_mlp_ta4_td4', 'display_name': r'Wasserstein $\alpha=0.1$', 'color': color_dict['ReinFlow-R (W2=0.1)']},
            # {'original_name': 'ppo_reflow_Wdiv_regularize_bc1.0_mlp_ta4_td4', 'display_name': r'Wasserstein $\alpha=1.0$', 'color': color_dict['ReinFlow-R (W2=1.0)']},
            {'original_name': 'ppo_reflow_mlp_ta4_td4', 'display_name': r'$\mathbf{\alpha=0.03}$', 'color': color_dict['ReinFlow-R (ours)']},
            {'original_name': 'ppo_reflow_Wdiv_regularize_bc0.01_mlp_ta4_td4', 'display_name': r'$\mathbf{\beta=0.01}$', 'color': color_dict['ReinFlow-R (W2=0.01)']},
            {'original_name': 'ppo_reflow_Wdiv_regularize_bc0.1_mlp_ta4_td4', 'display_name': r'$\mathbf{\beta=0.1}$', 'color': color_dict['ReinFlow-R (W2=0.1)']},
            {'original_name': 'ppo_reflow_Wdiv_regularize_bc1.0_mlp_ta4_td4', 'display_name': r'$\mathbf{\beta=1.0}$', 'color': color_dict['ReinFlow-R (W2=1.0)']},
        ],
        'gym-state-all': [
            {'original_name': 'qsm_diffusion_mlp_ta4_td20', 'display_name': 'QSM', 'color': color_dict['QSM']},
            {'original_name': 'dipo_diffusion_mlp_ta4_td20', 'display_name': 'DIPO', 'color': color_dict['DIPO']},
            {'original_name': 'idql_diffusion_mlp_ta4_td20', 'display_name': 'IDQL', 'color': color_dict['IDQL']},
            {'original_name': 'dql_diffusion_mlp_ta4_td20', 'display_name': 'DQL', 'color': color_dict['DQL']},
            {'original_name': 'rwr_diffusion_mlp_ta4_td20', 'display_name': 'DRWR', 'color': color_dict['DRWR']},
            {'original_name': 'awr_diffusion_mlp_ta4_td20', 'display_name': 'DAWR', 'color': color_dict['DAWR']},
            {'original_name': 'shortcut_mlp_img_ta4_td1', 'display_name': 'ReinFlow-S (ours)', 'color': color_dict['ReinFlow-S (ours)']},
            {'original_name': 'shortcut_mlp_img_td4_td1_datascale_32', 'display_name': 'ReinFlow-S, 32 (ours)', 'color': color_dict['ReinFlow-S, 32 (ours)']},
            {'original_name': 'shortcut_mlp_img_td4_td1_datascale_64', 'display_name': 'ReinFlow-S, 64 (ours)', 'color': color_dict['ReinFlow-S, 64 (ours)']},
            {'original_name': 'shortcut_mlp_img_td4_td1_datascale_100', 'display_name': 'ReinFlow-S, 100 (ours)', 'color': color_dict['ReinFlow-S, 100 (ours)']},
            {'original_name': 'reflow_mlp_img_ta4_td1', 'display_name': 'ReinFlow-R (ours)', 'color': color_dict['ReinFlow-R (ours)']},
            {'original_name': 'reflow_beta_mlp_img_1', 'display_name': 'ReinFlow-beta (ours)', 'color': color_dict['ReinFlow-beta (ours)']},
            {'original_name': 'reflow_logitnormal_mlp_img_td4_td1', 'display_name': 'ReinFlow-logitnormal (ours)', 'color': color_dict['ReinFlow-logitnormal (ours)']},
            {'original_name': 'diffusion_mlp_img_ta4_td100_tdf5', 'display_name': 'DPPO', 'color': color_dict['DPPO']},
        ],
        'can-img': [
            {'original_name': 'shortcut_mlp_ta4_td1_tdf1', 'display_name': 'ReinFlow-S (ours)', 'color': color_dict['ReinFlow-S (ours)']},
            {'original_name': 'flow_mlp_img_ta4_td1_tdf1', 'display_name': 'ReinFlow-R (ours)', 'color': color_dict['ReinFlow-R (ours)']},
            {'original_name': 'gaussian_mlp_img_ta4', 'display_name': 'Gaussian', 'color': color_dict['Gaussian']},
            {'original_name': 'diffusion_mlp_img_ta4_td100_tdf5', 'display_name': 'DPPO', 'color': color_dict['DPPO']},
        ],
        'square-img': [
            {'original_name': 'reflow', 'display_name': 'ReinFlow-R (ours)', 'color': color_dict['ReinFlow-R (ours)']},
            {'original_name': 'shortcut_mlp_img_td4_td1_datascale_100', 'display_name': 'ReinFlow-S (ours)', 'color': color_dict['ReinFlow-S (ours)']},
            {'original_name': 'gaussian_mlp_img_ta4', 'display_name': 'Gaussian', 'color': color_dict['Gaussian']},
            {'original_name': 'diffusion_mlp_img_ta4_td100_tdf5', 'display_name': 'DPPO', 'color': color_dict['DPPO']},
        ],
        'square-img-logitbeta': [
            {'original_name': 'reflow_mlp_img_td4_td1_tdf1', 'display_name': 'uniform', 'color': color_dict['uniform']},
            {'original_name': 'reflow_logitnormal_mlp_img_td4_td1', 'display_name': 'logitnormal', 'color': color_dict['logitnormal']},
            {'original_name': 'reflow_mlp_img_ta4_td1_tdf1', 'display_name': 'beta', 'color': color_dict['beta']},
        ],
        'transport-img': [
            {'original_name': 'gaussian_mlp_img_ta8', 'display_name': 'Gaussian', 'color': color_dict['Gaussian']},
            {'original_name': 'diffusion_mlp_img_ta8_td100_tdf5', 'display_name': 'DPPO', 'color': color_dict['DPPO']},
            {'original_name': 'shortcut_mlp_img_ta8_td4_tdf4', 'display_name': 'ReinFlow-S (ours)', 'color': color_dict['ReinFlow-S (ours)']},
        ],
        'kitchen-complete-v0': [
            {'original_name': 'fql_mlp_ta4_td4', 'display_name': 'FQL', 'color': color_dict['FQL']},
            {'original_name': 'ppo_diffusion_mlp_ta4_td20_tdf10', 'display_name': 'DPPO', 'color': color_dict['DPPO']},
            {'original_name': 'ppo_shortcut_mlp_ta4_td4_tdf4', 'display_name': 'ReinFlow-S (ours)', 'color': color_dict['ReinFlow-S (ours)']},
        ],
        'kitchen-complete-v0-denoise_step': [
            {'original_name': 'ppo_shortcut_mlp_ta4_td4_tdf4', 'display_name': 'ReinFlow-S (4 steps)', 'color': color_dict['ReinFlow-S (ours)']},
            {'original_name': 'ppo_shortcut_mlp_ta4_td2_tdf2', 'display_name': 'ReinFlow-S (2 steps)', 'color': color_dict['ReinFlow-S (2 steps)']},
            {'original_name': 'ppo_shortcut_mlp_ta4_td1_tdf1', 'display_name': 'ReinFlow-S (1 step)', 'color': color_dict['ReinFlow-S (1 step)']},
        ],
        'kitchen-partial-v0': [
            {'original_name': 'fql_mlp_ta4_td4', 'display_name': 'FQL', 'color': color_dict['FQL']},
            {'original_name': 'ppo_diffusion_mlp_ta4_td20_tdf10', 'display_name': 'DPPO', 'color': color_dict['DPPO']},
            {'original_name': 'ppo_shortcut_mlp_ta4_td4_tdf4', 'display_name': 'ReinFlow-S (ours)', 'color': color_dict['ReinFlow-S (ours)']},
        ],
        'kitchen-mixed-v0': [
            {'original_name': 'fql_mlp_ta4_td4', 'display_name': 'FQL', 'color': color_dict['FQL']},
            {'original_name': 'ppo_diffusion_mlp_ta4_td20_tdf10', 'display_name': 'DPPO', 'color': color_dict['DPPO']},
            {'original_name': 'ppo_shortcut_mlp_ta4_td4_tdf4', 'display_name': 'ReinFlow-S (ours)', 'color': color_dict['ReinFlow-S (ours)']},
        ],                     
        'kitchen-partial-v0-denoise_step':[
            {'original_name': 'ppo_shortcut_mlp_ta4_td4_tdf4', 'display_name': 'ReinFlow-S (4 steps)', 'color': color_dict['ReinFlow-S (ours)']},
            {'original_name': 'ppo_shortcut_mlp_ta4_td2_tdf2', 'display_name': 'ReinFlow-S (2 steps)', 'color': color_dict['ReinFlow-S (2 steps)']},
            {'original_name': 'ppo_shortcut_mlp_ta4_td1_tdf1', 'display_name': 'ReinFlow-S (1 step)', 'color': color_dict['ReinFlow-S (1 step)']},
        ],
        'kitchen-partial-v0-sigma_s_t':[
            {'original_name': 'ppo_shortcut_learn_s_mlp_ta4_td4_tdf4', 'display_name': r'ReinFlow-S: $\sigma(s)$', 'color': color_dict['ReinFlow-R (sigma(s,t))']},
            {'original_name': 'ppo_shortcut_mlp_ta4_td4_tdf4', 'display_name': r'ReinFlow-S: $\sigma(s,t)$', 'color': color_dict['ReinFlow-R (sigma(s))']},
        ]
    }

max_n_step_dict={
        'can-img': 100,
        'square-img': 300,
        'square-img-logitbeta': 300,
        'transport-img': 180,
        'hopper': np.float32('inf'),
        'hopper-d4rl': 1_000,
        'walker': 1_000,
        'walker-d4rl': 1_000,
        'ant': 1_000,
        'ant-d4rl': 1_000,
        'humanoid': 2_00,
        'humanoid-d4rl': 2_00,
        'humanoid-regularize-compare': 2_00,
        'kitchen-complete-v0': 1765000,
        'kitchen-complete-v0-denoise_step': np.float32('inf'),
        'kitchen-partial-v0': np.float32('inf'),
        'kitchen-partial-v0-denoise_step': np.float32('inf'),
        'kitchen-partial-v0-sigma_s_t': np.float32('inf'),
        'kitchen-mixed-v0': np.float32('inf'),
    }