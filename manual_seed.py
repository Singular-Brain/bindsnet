import random 
import numpy as np
import torch 

def manual_seed(seed):
    random.seed(seed)
    gpu = True
    if gpu and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
    else:
        torch.manual_seed(seed)
        np.random.seed(seed)
        device = "cpu"
        if gpu:
            gpu = False