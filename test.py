import RNA
import numpy as np
import torch

from utils.rna_lib import structure_dotB2Edge, structure_edge2DotB

t = torch.Tensor([0,0,0,1])

flag = torch.any(t == 1.)

print(flag)