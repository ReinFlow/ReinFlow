def set_seed_everywhere(seed:int):
    import random 
    import torch
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    
    