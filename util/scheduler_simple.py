import numpy as np
import torch
from torch.optim.lr_scheduler import _LRScheduler

class CustomScheduler(_LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        schedule_type: str,
        **kwargs
    ):
        self.schedule_type = schedule_type
        self.kwargs = kwargs
        self.scheduler_func = self.get_scheduler(schedule_type, **kwargs)
        super(CustomScheduler, self).__init__(optimizer)

    def get_scheduler(self, schedule_type: str, **kwargs):
        return get_scheduler(schedule_type, **kwargs)

    def get_lr(self):
        return [self.scheduler_func(self.last_epoch) for _ in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr
    
    def reset(self):
        self.last_epoch=-1

def get_scheduler(schedule_type: str, **kwargs):
    '''
    examples:
        get_schedule('constant', level=0.2))
        get_schedule('linear', max=0.2, hold_steps=100, anneal_steps=300, min=0.1))
        get_schedule('cosine', max=0.2, hold_steps=100, anneal_steps=300, min=0.1))
        
        get_schedule('constant_warmup', min=0.1, warmup_steps=100, max=0.2))
        get_schedule('linear_warmup', min=0.1, warmup_steps=100, max=0.2, hold_steps=50, anneal_steps=300))
        get_schedule('cosine_warmup', min=0.1, warmup_steps=100, max=0.2, hold_steps=50, anneal_steps=300))
    '''
    if schedule_type == 'constant':
        assert len(kwargs) == 1
        return lambda x: float(kwargs['level'])
    elif schedule_type == 'constant_warmup':
        assert len(kwargs) == 3
        eps_min, warmup, eps_max = float(kwargs['min']), int(kwargs['warmup_steps']), float(kwargs['max'])
        if warmup == 0:
            return lambda x: eps_max
        return lambda x: (eps_max-eps_min)/warmup * x + eps_min if x < warmup else eps_max
    elif schedule_type == 'linear':
        assert len(kwargs) == 4
        eps_min, hold_steps, eps_max, anneal_steps = float(kwargs['min']), int(kwargs['hold_steps']), float(kwargs['max']), int(kwargs['anneal_steps'])
        return lambda x: np.clip(eps_max - (eps_max - eps_min) * (x - hold_steps) / anneal_steps, eps_min, eps_max)
    elif schedule_type == 'linear_warmup':
        assert len(kwargs) == 5
        eps_min, warmup, eps_max, hold_steps, anneal_steps = float(kwargs['min']), int(kwargs['warmup_steps']), float(kwargs['max']), int(kwargs['hold_steps']), int(kwargs['anneal_steps'])
        if warmup == 0:
            return lambda x: np.clip(eps_max - (eps_max - eps_min) * (x - hold_steps) / anneal_steps, eps_min, eps_max)
        return lambda x: (eps_max-eps_min)/warmup * x + eps_min if x < warmup else np.clip(eps_max - (eps_max - eps_min) * (x - hold_steps - warmup) / anneal_steps, eps_min, eps_max)
    elif schedule_type == 'cosine':
        assert len(kwargs) == 4
        eps_max, hold_steps, anneal_steps, eps_min= float(kwargs['max']), int(kwargs['hold_steps']), int(kwargs['anneal_steps']), float(kwargs['min'])
        return lambda x: eps_min + (eps_max - eps_min) * 0.5 * (1 + np.cos(np.clip((x - hold_steps) / anneal_steps, 0, 1) * np.pi))
    elif schedule_type == 'cosine_warmup':
        assert len(kwargs) == 5
        eps_min, warmup, eps_max, hold_steps, anneal_steps = float(kwargs['min']), int(kwargs['warmup_steps']), float(kwargs['max']), int(kwargs['hold_steps']), int(kwargs['anneal_steps'])
        if warmup == 0:
            return lambda x: eps_min + (eps_max - eps_min) * 0.5 * (1 + np.cos(np.clip((x - hold_steps-warmup) / anneal_steps, 0, 1) * np.pi))
        return lambda x: (eps_max-eps_min)/warmup * x + eps_min if x < warmup else eps_min + (eps_max - eps_min) * 0.5 * (1 + np.cos(np.clip((x - hold_steps-warmup) / anneal_steps, 0, 1) * np.pi))
    else:
        raise ValueError('Unknown schedule: %s' % schedule_type)


if __name__ == "__main__":
    
    import matplotlib.pyplot as plt
    # Define the schedulers
    schedulers = [
        ('Constant', get_scheduler('constant', level=0.08)),
        ('Linear', get_scheduler('linear', min=0.016, hold_steps=200, max=0.08, anneal_steps=800)),
        ('Cosine', get_scheduler('cosine', max=0.08, min=0.016, hold_steps=200, anneal_steps=800)),
        ('Constant Warmup', get_scheduler('constant_warmup', min=0.016, warmup_steps=100, max=0.08)),
        ('Linear Warmup', get_scheduler('linear_warmup', max=0.08, min=0.016, warmup_steps=50, hold_steps=250, anneal_steps=700)),
        ('Cosine Warmup', get_scheduler('cosine_warmup', max=0.08, min=0.016, warmup_steps=100, hold_steps=200, anneal_steps=700))
    ]

    # Create the plot
    fig, axs = plt.subplots(2, 3, figsize=(12, 8))
    fig.suptitle('Scheduler Visualizations', fontsize=16)

    # Generate x values
    x = np.arange(0, 1001)

    # Plot each scheduler
    for i, (name, scheduler) in enumerate(schedulers):
        row = i // 3
        col = i % 3
        
        y = [scheduler(t) for t in x]
        axs[row, col].plot(x, y)
        axs[row, col].set_title(name)
        axs[row, col].set_xlabel('Steps')
        axs[row, col].set_ylabel('Value')
        axs[row, col].grid(True)

    # Adjust layout and display
    plt.tight_layout()
    plt.savefig('scheduler_visualization.png')
    print(f"figure saved to {'scheduler_visualization.png'}")