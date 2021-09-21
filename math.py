from .defaults import *
from .conversions import *

import torch

def isoftmax(x:torch.Tensor, dim:int, dtype:torch.dtype = None) -> torch.Tensor:
    # Normal softmax
    if not x.is_complex(): return torch.softmax(x, dim=dim, dtype=dtype)

    # Imaginary softmax
    angle = torch.atan(x.imag/x.real)
    magnitude = torch.sqrt(x.real.pow(2) + x.imag.pow(2))
    softMagnitude = torch.softmax(magnitude, dim=dim, dtype=dtype)
    
    # Convert back to imaginary
    newReal = softMagnitude * torch.cos(angle)
    newImag = softMagnitude * torch.sin(angle)
    
    # Return in proper datatype
    newReal.unsqueeze(-1)
    newImag.unsqueeze(-1)
    return torch.view_as_complex(torch.stack((newReal, newImag), dim=-1))
