from .defaults import *
from .conversions import *

import torch

@torch.jit.script
def isoftmax(x:torch.Tensor, dim:int, dtype:torch.dtype = -1) -> torch.Tensor:
    # Normal softmax
    if not x.is_complex(): 
        return torch.softmax(x, dim=dim, dtype=dtype)

    # Imaginary softmax
    angle:torch.Tensor = torch.atan(x.imag/x.real)
    magnitude:torch.Tensor = torch.sqrt(x.real.pow(2) + x.imag.pow(2))
    softMagnitude:torch.Tensor = torch.softmax(magnitude, dim=dim, dtype=dtype)
    
    # Convert back to imaginary
    newReal:torch.Tensor = softMagnitude * torch.cos(angle)
    newImag:torch.Tensor = softMagnitude * torch.sin(angle)
    
    # Return in proper datatype
    newReal.unsqueeze(-1)
    newImag.unsqueeze(-1)
    return torch.view_as_complex(torch.stack((newReal, newImag), dim=-1))
