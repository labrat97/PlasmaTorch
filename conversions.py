from .defaults import *

import torch
import torch.nn as nn


# Turn a pointwise signal into a smearwise one
class Smear(nn.Module):
  def __init__(self, samples:int = DEFAULT_FFT_SAMPLES, lowerScalar:float = 1./16, 
    upperScalar:float = 1./16, dtype:torch.dtype = DEFAULT_DTYPE):
    super(Smear, self).__init__()

    self.samples:int = samples
    self.smearBias:nn.Parameter = nn.Parameter(torch.zeros(1, dtype=dtype))
    self.smearWindow:nn.Parameter = nn.Parameter(torch.tensor([-lowerScalar, upperScalar]).type(dtype))

    self.__iter = torch.Tensor(
      [builder / (self.samples-1) for builder in range(self.samples)]
    ).type(dtype).detach()
  
  def forward(self, x:torch.Tensor) -> torch.Tensor:
    xBias:torch.Tensor = x + self.smearBias
    if self.samples <= 1:
      return xBias

    lowerSmear:torch.Tensor = self.smearWindow[0]
    upperSmear:torch.Tensor = self.smearWindow[1]
    xRange:torch.Tensor = (upperSmear - lowerSmear) * xBias
    xLow:torch.Tensor = ((1 + lowerSmear) * xBias)

    return (xRange * self.__iter) + xLow

@torch.jit.script
def nantonum(x:torch.Tensor) -> torch.Tensor:
  # Already implemented
  if not x.is_complex(): return torch.nan_to_num(x)

  # Do it on a per element basis
  real = x.real.nan_to_num()
  imag = x.imag.nan_to_num()
  
  # Create the stablized output and return
  return torch.view_as_complex(torch.stack((real, imag), dim=-1))

@torch.jit.script
def toComplex(x:torch.Tensor) -> torch.Tensor:
  # Already done
  if x.is_complex(): return x
  
  # Turn into a complex number
  complexProto = torch.stack((x, torch.zeros_like(x)), dim=-1)
  return torch.view_as_complex(complexProto)

@torch.jit.script
def strToTensor(x:str) -> torch.Tensor:
  # Prepare memory for construction
  rawstr = torch.zeros((len(x)), dtype=torch.int32, requires_grad=False, device='cpu')

  # Copy string
  for idx, char in enumerate(rawstr):
    rawstr[idx] = ord(char)
  
  return rawstr

@torch.jit.script
def tensorToStr(x:torch.Tensor) -> List[str]:
  # Make sure it can be represented in python natively
  if len(x.size()) == 1:
    wx = x.unsqueeze(0)
  else:
    wx = x.flatten(end_dim=-2)

  # Prepare python traced output
  pystr:str = []

  # Copy the string out of the tensor into Python's format
  for idx in range(wx.size(0)):
    # Isolate
    target:torch.Tensor = wx[idx]
    build:str = ''

    # Copy element by element
    for jdx in range(target.size(0)):
      build += target[jdx]
    
    # Add the string to the output list
    pystr.append(build)
  
  # Return all embedded strings in a list
  return pystr


class RealObserver(nn.Module):
    def __init__(self, units:int = 1, dtype:torch.dtype = DEFAULT_DTYPE):
        super(RealObserver, self).__init__()

        # Create the polarization parameter and type check
        self.polarization:nn.Parameter = nn.Parameter(torch.zeros((units), dtype=dtype))
        assert self.polarization.is_complex() == False
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        # Type checking
        assert x.is_complex()

        # Apply the polarization between the complex signal domains
        return (torch.cos(self.polarization) * x.real) \
            + (torch.sin(self.polarization) * x.imag)


class ComplexObserver(nn.Module):
    def __init__(self, units:int = 1, dtype:torch.dtype = DEFAULT_DTYPE):
        super(ComplexObserver, self).__init__()

        # Create the polarization parameter then type check
        self.polarization:nn.Parameter = nn.Parameter(torch.zeros((units), dtype=dtype))
        assert self.polarization.is_complex() == False

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        # Type checking
        assert x.is_complex() == False

        # Apply polarization to pull into complex plane
        xReal:torch.Tensor = torch.cos(self.polarization) * x
        xImag:torch.Tensor = torch.sin(self.polarization) * x
        
        # Resize and turn complex
        xReal.unsqueeze_(-1)
        xImag.unsqueeze_(-1)
        return torch.view_as_complex(torch.cat((xReal, xImag), dim=-1))
