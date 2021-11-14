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
def resampleSmear(x:torch.Tensor, samples:int, dim:int = -1) -> torch.Tensor:
  # Sample the constructing frequencies and phases, zero padding
  xfft:torch.Tensor = torch.fft.fft(x, dim=dim, n=samples)
  
  # Put the samples back to smearwise where no zero padding exists
  # This can be done because this is a natural signal
  # No data is lost or obscured in theory during upsampling, downsampling loses higher frequencies
  y:torch.Tensor = torch.fft.ifft(xfft, dim=dim, n=samples)
  if not torch.is_complex(x):
    y = y.abs()

  return y

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
def resampleContinuous(x:torch.Tensor, dim:int=-1, msi:int=-1) -> torch.Tensor:
  # Get rid of values that can propagate through horribly (+-inf, nan)
  xclean:torch.Tensor = nantonum(x)
  samples = x.size()[dim]
  xfft:torch.Tensor = torch.fft.fft(xclean, n=samples, dim=dim)
  ixfft:torch.Tensor = torch.fft.ifft(xfft, n=samples, dim=dim)

  # Make sure the most significant index's scale remains consistent
  resultScalar = xclean[..., msi] / ixfft[..., msi]
  return resultScalar.unsqueeze(-1) * ixfft

@torch.jit.script
def toComplex(x:torch.Tensor) -> torch.Tensor:
  # Already done
  if x.is_complex(): return x
  
  # Turn into a complex number
  complexProto = torch.stack((x, torch.zeros_like(x)), dim=-1)
  return torch.view_as_complex(complexProto)


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
