from .defaults import *
from .conversions import *

import torch

def pi() -> torch.Tensor:
    return (torch.ones((1)) * 3.141592653589793238462643383279502).detach()

@torch.jit.script
def golden() -> torch.Tensor:
    one = torch.ones((1)).detach()
    square = torch.sqrt(one * 5)

    return (one + square) / 2

@torch.jit.script
def i() -> torch.Tensor:
    return torch.view_as_complex(torch.stack(
        (torch.zeros(1), torch.ones(1)), \
        dim=-1)).detach()

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
    I = i()
    return torch.exp(I * x.real) + torch.exp(I * x.imag) - 1

@torch.jit.script
def isin(x:torch.Tensor) -> torch.Tensor:
    # Normal sin
    if not x.is_complex():
        return torch.sin(x)

    # Main conversion
    return i() * icos(x)
