import os

from bindsnet.datasets import MNIST
from bindsnet.encoding import PoissonEncoder

import numpy as np


# Load MNIST data.
dataset = MNIST(
    None,
    None,
    root=os.path.join("..", "..", "data", "MNIST"),
    download=False,
    # transform=transforms.Compose(
    #     [transforms.ToTensor(),
    #     transforms.Lambda(lambda x: x * intensity),
    #     transforms.CenterCrop(crop_size)]
)

np.savez(f'a.npz', np.zeros(10))

target_classes = (2,4)
mask = np.array([1 if dataset[i]['label'] in target_classes else 0 for i in range(len(dataset))])
np.savez(f'mask_{target_classes[0]}_{target_classes[1]}.npz', mask)