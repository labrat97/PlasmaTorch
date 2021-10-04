from .defaults import *
from .conversions import *

import torch
import math

@torch.jit.script
def imagnitude(x:torch.Tensor) -> torch.Tensor:
    if not x.is_complex():
        return x

    # Main conversion
    return torch.sqrt(x.real.pow(2) + x.imag.pow(2))

@torch.jit.script
def ipolarization(x:torch.Tensor) -> torch.Tensor:
    # Main conversion
    return torch.angle(x)

@torch.jit.script
def isoftmax(x:torch.Tensor, dim:int) -> torch.Tensor:
    # Normal softmax
    if not x.is_complex(): 
        return torch.softmax(x, dim=dim)

    # Imaginary softmax
    angle:torch.Tensor = ipolarization(x)
    magnitude:torch.Tensor = imagnitude(x)
    softMagnitude:torch.Tensor = torch.softmax(magnitude, dim=dim)
    
    # Convert back to imaginary
    newReal:torch.Tensor = softMagnitude * torch.cos(angle)
    newImag:torch.Tensor = softMagnitude * torch.sin(angle)
    
    # Return in proper datatype
    newReal.unsqueeze(-1)
    newImag.unsqueeze(-1)
    return torch.view_as_complex(torch.stack((newReal, newImag), dim=-1))

@torch.jit.script
def icos(x:torch.Tensor) -> torch.Tensor:
    # Normal cos
    if not x.is_complex():
        return torch.cos(x)

    # Main conversion
    real = torch.cos(x.real)
    imag = torch.cos(x.imag)
    return torch.view_as_complex(torch.stack((real, imag), dim=-1))

@torch.jit.script
def isin(x:torch.Tensor) -> torch.Tensor:
    # Normal sin
    if not x.is_complex():
        return torch.sin(x)

    # Main conversion
    real = torch.sin(x.real)
    imag = torch.sin(x.imag)
    return torch.view_as_complex(torch.stack((real, imag), dim=-1))
