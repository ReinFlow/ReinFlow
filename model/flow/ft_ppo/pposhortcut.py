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





import logging
log = logging.getLogger(__name__)
from model.flow.mlp_shortcut import NoisyShortCutFlowMLP
from model.flow.ft_ppo.ppoflow import PPOFlow
import torch
from torch import Tensor as Tensor
from torch.distributions import Normal

class PPOShortCut(PPOFlow):
    def __init__(self, 
                 device,
                 policy,
                 critic,
                 actor_policy_path,
                 act_dim,
                 horizon_steps,
                 act_min, 
                 act_max,
                 obs_dim,
                 cond_steps,
                 noise_scheduler_type,
                 inference_steps,
                 ft_denoising_steps,
                 randn_clip_value,
                 min_sampling_denoising_std,
                 min_logprob_denoising_std,
                 logprob_min,
                 logprob_max,
                 clip_ploss_coef,
                 clip_ploss_coef_base,
                 clip_ploss_coef_rate,
                 clip_vloss_coef,
                 denoised_clip_value,
                 max_logprob_denoising_std,
                 time_dim_explore,
                 learn_explore_time_embedding,
                 use_time_independent_noise,
                 noise_hidden_dims,
                 logprob_debug_sample,
                 logprob_debug_recalculate,
                 explore_net_activation_type
                 ):
        
        super().__init__(
                 device,
                 policy,
                 critic,
                 actor_policy_path,
                 act_dim,
                 horizon_steps,
                 act_min, 
                 act_max,
                 obs_dim,
                 cond_steps,
                 noise_scheduler_type,
                 inference_steps,
                 ft_denoising_steps,
                 randn_clip_value,
                 min_sampling_denoising_std,
                 min_logprob_denoising_std,
                 logprob_min,
                 logprob_max,
                 clip_ploss_coef,
                 clip_ploss_coef_base,
                 clip_ploss_coef_rate,
                 clip_vloss_coef,
                 denoised_clip_value,
                 max_logprob_denoising_std,
                 time_dim_explore,
                 learn_explore_time_embedding,
                 use_time_independent_noise,
                 noise_hidden_dims,
                 logprob_debug_sample,
                 logprob_debug_recalculate,
                 explore_net_activation_type
        )
    
    # overload
    def init_actor_ft(self, policy_copy):
        self.actor_ft = NoisyShortCutFlowMLP(policy=policy_copy,
                                    denoising_steps=self.inference_steps,
                                    learn_explore_noise_from = self.inference_steps - self.ft_denoising_steps,
                                    inital_noise_scheduler_type=self.noise_scheduler_type,
                                    min_logprob_denoising_std = self.min_logprob_denoising_std,
                                    max_logprob_denoising_std = self.max_logprob_denoising_std,
                                    learn_explore_time_embedding=self.learn_explore_time_embedding,
                                    time_dim_explore=self.time_dim_explore,
                                    use_time_independent_noise=self.use_time_independent_noise,
                                    device=self.device,
                                    noise_hidden_dims=self.noise_hidden_dims,
                                    activation_type=self.explore_net_activation_type
                                    )
        
    
    ################################ debug eval-finetune mismatch #########################################
    # overload 
    # the major difference between shortcut and flow is how we generte time. 
    # in reflow we use linspace(0,1,K) 
    # while in shortcut we use linspace(0,1-1/K, K)
    def get_logprobs(self, 
                     cond:dict, 
                     x_chain:Tensor, 
                     get_entropy =False, 
                     normalize_denoising_horizon=False, 
                     normalize_act_space_dimension=False,
                     clip_intermediate_actions=True,
                     verbose_entropy_stats=True,
                     debug=True,
                     account_for_initial_stochasticity=False,
                     get_chains_stds=True
                     ):
        '''
        inputs:
            x_chain: torch.Tensor of shape `[batchsize, self.inference_steps+1, self.horizon_steps, self.act_dim]`
           
        outputs:
            log_prob. tensor of shape `[batchsize]`
            entropy_rate_est: tensor of shape `[batchsize]`
            chains_stds.mean(): tensor of shape `[batchsize]`
            
        explanation:
            p(x0|s)       = N(x0|0, 1)
            p(xt+1|xt, s) = N(xt+1 | xt + v(xt, s)1/K; sigma_t^2)
            
            log p(xK|s) = log p(x0) + \sum_{t=0}^{K-1} log p(xt+1|xt, s)
            H(X0:K)     = H(x0|s)     + \sum_{t=0}^{K-1} H(Xt+1|X_t, s)
            entropy rate H(X) = H(X0:K)/(K+1) asymptotically converges to the entropy per symbol when K goes to infinity.
            we view the actions at each dimension and horizon as conditionally independent on the state s and previous action. 
        '''
        if self.inference_steps != self.actor_ft.denoising_steps:
            raise ValueError(f"self.inference_steps({self.inference_steps}) != self.actor_ft.denoising_steps={self.actor_ft.denoising_steps}, check how your {self.__class__.__name__} assigns that to your {self.actor_ft.__class__.__name__}!")

        logprob = 0.0
        joint_entropy=0.0 
        entropy_rate_est=0.0
        logprob_steps = 0
        
        B = x_chain.shape[0]
        chains_prev = x_chain[:, :-1,:, :].flatten(-2,-1)                       # [batchsize, self.inference_steps, self.horizon_steps x self.act_dim]
        chains_next = x_chain[:, 1:, :, :].flatten(-2,-1)                       # [batchsize, self.inference_steps, self.horizon_steps x self.act_dim]
        chains_stds = torch.zeros_like(chains_prev, device=self.device)         # [batchsize, self.inference_steps, self.horizon_steps x self.act_dim]
        
        # initial probability
        init_dist = Normal(torch.zeros(B, self.horizon_steps* self.action_dim, device=self.device), 1.0)
        logprob_init = init_dist.log_prob(x_chain[:,0].reshape(B,-1)).sum(-1)   # [batchsize]
        if get_entropy:
            entropy_init = init_dist.entropy().sum(-1)                          # [batchsize]
        if account_for_initial_stochasticity:
            logprob+=logprob_init
            if get_entropy:
                joint_entropy+=entropy_init
            logprob_steps+=1
        
        # transition probabilities
        chains_vel  = torch.zeros_like(chains_prev, device=self.device)         # [batchsize, self.inference_steps, self.horizon_steps x self.act_dim]
        dt = 1.0/self.inference_steps
        steps = torch.linspace(0, 1-dt, self.inference_steps).repeat(B, 1).to(self.device)  # [batchsize, self.inference_steps]
        for i in range(self.inference_steps):
            t       = steps[:,i]
            xt      = x_chain[:,i]                                              # [batchsize, self.horizon_steps , self.act_dim]
            d = torch.full((B,), dt, device=self.device)
            # this is different from 1-ReFlow
            vt, nt  =self.actor_ft.forward(xt, t, d, cond, True, i)             # [batchsize, self.horizon_steps, self.act_dim]
            chains_vel[:,i]  = vt.flatten(-2,-1)                                # [batchsize, self.horizon_steps x self.act_dim]
            chains_stds[:,i] = nt                                               # [batchsize, self.horizon_steps x self.act_dim]
            logprob_steps+=1
        chains_mean = (chains_prev + chains_vel * dt)                           # [batchsize, self.inference_steps, self.horizon_steps x self.act_dim]
        if clip_intermediate_actions:
            chains_mean = chains_mean.clamp(-self.denoised_clip_value, self.denoised_clip_value)
        
        # transition distribution
        chains_dist = Normal(chains_mean, chains_stds)
        
        # logprobability and entropy of the transitions
        logprob_trans = chains_dist.log_prob(chains_next).sum(-1)               # [batchsize, self.inference_steps] sum up self.horizon_steps x self.act_dim 
        if get_entropy:
            entropy_trans = chains_dist.entropy().sum(-1)                       # [batchsize, self.inference_steps] Sum over all dimensions
        
        # logprobability of the whole markov chain.
        logprob += logprob_trans.sum(-1)                          # [batchsize] accumulate over inference steps (Markov property)
        if self.logprob_debug_recalculate: 
            log.info(f"logprob_init={logprob_init.mean().item()}, logprob_trans={logprob_trans.mean().item()}")
        # entropy rate estimate of the whole markov chain
        if get_entropy:
            joint_entropy +=entropy_trans.sum(-1)
        
        if get_entropy:
            entropy_rate_est = joint_entropy/logprob_steps
        if normalize_denoising_horizon:
            logprob = logprob / logprob_steps
        if normalize_act_space_dimension:
            logprob = logprob/self.act_dim_total
            if get_entropy:
                entropy_rate_est = entropy_rate_est/self.act_dim_total
        if verbose_entropy_stats and get_entropy:
            log.info(f"entropy_rate_est={entropy_rate_est.shape} Entropy Percentiles: 10%={entropy_rate_est.quantile(0.1):.2f}, 50%={entropy_rate_est.median():.2f}, 90%={entropy_rate_est.quantile(0.9):.2f}")
        
        if get_entropy:
            if get_chains_stds:
                return logprob, entropy_rate_est, chains_stds.mean()
            return logprob, entropy_rate_est, 
        else:
            if get_chains_stds:
                return logprob, chains_stds.mean()
            return logprob 
    
    # overload
    @torch.no_grad()
    def get_actions(self, 
                    cond:dict, 
                    eval_mode:bool, 
                    save_chains=False, 
                    normalize_denoising_horizon=False, 
                    normalize_act_space_dimension=False,
                    clip_intermediate_actions=True,
                    account_for_initial_stochasticity=True,
                    ret_logprob=True
                    ):
        '''
        inputs:
            cond: dict, contatinin...
                'state': obs. observation in robotics. torch.Tensor(batchsize, cond_steps, obs_dim)
            deterministic: bool, whether use deterministic inference or stochastic interpolate
            deterministic: bool, whether use deterministic inference or stochastic interpolate
            save_chains: whether to return predictions at each step
            normalize_denoising_horizon: bool, whether to normalize time horizon when calculating the log probability.  When toggled, could reduce some hyper parameter tuning when the action space changes. 
            normalize_act_space_dimension: bool, whether to normalize action dimension when calculating the log probability. When toggled, could reduce some hyper parameter tuning when the action space changes. 
            clip_intermediate_actions: bool, whether to clip intermediate actions during the flow. 
        outputs:
            1. (xt, x_chain, log_prob) when `save_chains` =True
            2. (xt, log_prob) when `save_chains` = False
            xt. tensor of shape `[batchsize, self.horizon_steps, self.action_dim]`
            x_chains. tensor of shape `[self.inference_steps +1 ,self.data_shape]`: x0, x1, x2, ... xK
            logprob. tensor of shape `[batchsize]` or None
        '''
        if self.inference_steps != self.actor_ft.denoising_steps:
            raise ValueError(f"self.inference_steps({self.inference_steps}) != self.actor_ft.denoising_steps={self.actor_ft.denoising_steps}, check how your {self.__class__.__name__} assigns that to your {self.actor_ft.__class__.__name__}!")

        # when doing deterministic sampling should calculate logprob again.
        B=cond["state"].shape[0]
        dt = 1/self.inference_steps
        # this is different from flows! 1-dt  if you don't add 1-dt there will be a performance gap.
        steps = torch.linspace(0,1-dt,self.inference_steps).repeat(B, 1).to(self.device)  # [batchsize, num_steps]
        if save_chains:
            x_chain=torch.zeros((B, self.inference_steps+1, self.horizon_steps, self.action_dim), device=self.device)
        if ret_logprob:
            log_prob=0.0 
            log_prob_steps=0
            if self.logprob_debug_sample: 
                log_prob_list = []
        
        # sample first point
        xt, log_prob_init = self.sample_first_point(B)
        if ret_logprob and account_for_initial_stochasticity:
            log_prob+=log_prob_init
            log_prob_steps+=1
            if self.logprob_debug_sample:
                log_prob_list.append(log_prob_init.mean().item())
        
        xt:torch.Tensor
        if save_chains:
            x_chain[:, 0] = xt
        
        for i in range(self.inference_steps):
            t = steps[:,i]
            d = torch.full((B,), dt, device=self.device)
            # different from 1-ReFlow
            vt, nt =self.actor_ft.forward(xt, t, d, cond, learn_exploration_noise=False, step=i)
            xt += vt* dt
            if clip_intermediate_actions: # Discourage excessive exploration
                xt = xt.clamp(-self.denoised_clip_value, self.denoised_clip_value)
            
            # add noise during training, also prevent too deterministic policies
            std = nt.unsqueeze(-1).reshape(xt.shape)
            std = torch.clamp(std, min=self.min_sampling_denoising_std)    # each value in [self.min_sampling_denoising_std, self.max_logprob_denoising_std]
            dist = Normal(xt, std)
            if not eval_mode:
                xt = dist.sample().clamp_(dist.loc - self.randn_clip_value * dist.scale,
                                          dist.loc + self.randn_clip_value * dist.scale).to(self.device)
            
            # prevent last action overflow
            if i == self.inference_steps-1:
                xt = xt.clamp_(self.act_min, self.act_max)
            if ret_logprob:
                # compute logprobs for eval or train
                logprob_transition = dist.log_prob(xt).sum(dim=(-2,-1)).to(self.device)
                if self.logprob_debug_sample: 
                    log_prob_list.append(logprob_transition.mean().item())
                log_prob += logprob_transition
                log_prob_steps+=1
            if save_chains:
                x_chain[:, i+1] = xt
        
        if ret_logprob:
            if normalize_denoising_horizon:
                log_prob = log_prob/log_prob_steps
            if normalize_act_space_dimension:
                log_prob = log_prob/self.act_dim_total
            if self.logprob_debug_sample:
                transform_logprob=torch.log(1-torch.tanh(x_chain[-1])**2+1e-7).sum(dim=(-2,-1)).mean().item()
                print(f"log_prob_list={log_prob_list}, transform={transform_logprob}")
            if save_chains:
                return (xt, x_chain, log_prob)  
            return (xt, log_prob)
        else:
            if save_chains:
                return (xt, x_chain)
            return xt
    