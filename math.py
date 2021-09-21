from .defaults import *
from .conversions import *

import torch

def isoftmax(x:torch.Tensor, dim:int, dtype:torch.dtype = None) -> torch.Tensor:
    # Normal softmax
    if not x.is_complex(): return torch.softmax(x, dim=dim, dtype=dtype)

    # Imaginary softmax
    inter = torch.view_as_real(x)
    softInter = torch.softmax(inter, dim=(dim-1), dtype=dtype)

    # Turn back into imaginary
    return torch.view_as_complex(softInter)
