
import torch
device='cuda:5'
finetuned_ckpt_path='log/robomimic/finetune/Transport-shortcut-img-ft/seed42/last.pt'
net=torch.load(finetuned_ckpt_path, weights_only=True, map_location=device)
# print(f"net={net['model'].keys()}")
noise_net_ckpt={
    key.replace('actor_ft.explore_noise_net.', ''):value for key, value in net['model'].items() if key.startswith('actor_ft.explore_noise_net.')
}
print(f"noise_net={noise_net_ckpt.keys()}")

from model.flow.mlp_flow import ExploreNoiseNet

action_dim=14
horizon_steps=8
obs_dim=18
cond_steps=1
act_dim_total = action_dim * horizon_steps

noise_input_dim = 32
min_logprob_denoising_std=0.05
max_logprob_denoising_std=0.10
noise_hidden_dims=[384,384,384]
noise_activation_type='Tanh'

noise_net=ExploreNoiseNet(in_dim=noise_input_dim, 
                out_dim=act_dim_total,
                logprob_denoising_std_range=[min_logprob_denoising_std, max_logprob_denoising_std], 
                device=device,
                hidden_dims=noise_hidden_dims,
                activation_type=noise_activation_type)
noise_net.load_state_dict(noise_net_ckpt)
print(f"noise_net={noise_net}")
""" 
noise_net=ExploreNoiseNet(
  (mlp_logvar): MLP(
    (moduleList): ModuleList(
      (0): Sequential(
        (linear_1): Linear(in_features=32, out_features=384, bias=True)
        (act_1): Tanh()
      )
      (1-2): 2 x Sequential(
        (linear_1): Linear(in_features=384, out_features=384, bias=True)
        (act_1): Tanh()
      )
      (3): Sequential(
        (linear_1): Linear(in_features=384, out_features=112, bias=True)
        (act_1): Identity()
      )
    )
  )
)
"""



import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Assuming noise_net is already defined and loaded as in your query
device = 'cuda:5'

def embed_scalar(x, dim=16):
    """Generate a sinusoidal embedding for a scalar value."""
    embed = []
    for i in range(dim // 2):
        freq = 2 ** i
        embed.append(np.sin(2 * np.pi * x * freq))
        embed.append(np.cos(2 * np.pi * x * freq))
    return np.array(embed)

# Create a grid of time (t) and state (s) values
t_values = np.linspace(0, 1, 50)
s_values = np.linspace(0, 1, 50)
T, S = np.meshgrid(t_values, s_values)
Z = np.zeros_like(T)

# Compute average noise standard deviation for each (t, s) pair
for i in range(50):
    for j in range(50):
        t = T[i, j]
        s = S[i, j]
        embed_t = embed_scalar(t)
        embed_s = embed_scalar(s)
        noise_feature = np.concatenate([embed_t, embed_s])  # [32]
        noise_feature_tensor = torch.tensor(noise_feature, dtype=torch.float32, device=device).unsqueeze(0)  # [1, 32]
        with torch.no_grad():
            noise_std = noise_net(noise_feature_tensor)  # [1, 112]
        Z[i, j] = noise_std.mean().item()

# Create the 3D surface plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(T, S, Z, cmap='viridis')
ax.set_xlabel('Time (t)')
ax.set_ylabel('State (s)')
ax.set_zlabel('Average Noise Std')
ax.set_title('ExploreNoiseNet Visualization')
fig.colorbar(surf)
plt.savefig('noise_net_visualization.png')