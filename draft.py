import torch
import numpy as np


a = torch.FloatTensor(np.random.random((4, 4)))
b, c = a.topk(1)
e, f = a.topk(2)
print()
