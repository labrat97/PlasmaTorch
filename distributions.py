import torch
import torch.nn as nn
import torch.nn.functional as nnf

from .defaults import *
from .conversions import *
from .math import *

@torch.jit.script
def irregularGauss(x:torch.Tensor, mean:torch.Tensor, lowStd:torch.Tensor, highStd:torch.Tensor, reg:bool=False) -> torch.Tensor:
  """Generates an piecewise Gaussian curve according to the provided parameters.

  Args:
      x (torch.Tensor): The sampling value for the curve with indefinite size.
      mean (torch.Tensor): The means that generate the peaks of the function which
        has a shape that is broadcastable upon x.
      lowStd (torch.Tensor): The standard deviation to use when the function is below
        the defined mean. The size must be broadcastable upon x.
      highStd (torch.Tensor): The standard deviation to use when the function is
        above the defined mean. The size must be broadcastable upon x.
      reg (bool): Apply the regularization needed for the cdf to approach 1. Defaults to False.

  Returns:
      torch.Tensor: A sampled set of values with the same size as the input.
  """
  # Grab the correct side of the curve
  belowMean = torch.le(x, mean)
  std = (belowMean.int() * lowStd) + ((1 - belowMean.int()) * highStd)

  # Calculate the gaussian curve
  top = x - mean

  # Never hits 0 or inf
  bottom = nnf.softplus(std, beta=phi()[0])
  
  # Calculate the normal distribution
  factor = top / bottom
  result = torch.exp(-0.5 * torch.pow(factor, 2))
  if not reg:
    return result
  
  # Regulate the output so that the cdf approaches 0 at inf
  regulator = (bottom * torch.sqrt(2. * pi()))
  return result / regulator

class LinearGauss(nn.Module):
  """
  A linearly tuned irregular gaussian function to be used as an activation layer of sorts.
  """
  def __init__(self, channels:int, dtype:torch.dtype = DEFAULT_DTYPE):
    """Builds a new LinearGauss structure.

    Args:
        channels (int): The amount of linear gausses to build together. Must be broadcastable to
          the provided set of channels.
        dtype (torch.dtype): The type of the parameters used to calculate the gaussian curves.
    """
    super(LinearGauss, self).__init__()

    self.channels:int = channels

    self.mean:nn.Parameter = nn.Parameter(torch.zeros((self.channels), dtype=dtype))
    self.lowStd:nn.Parameter = nn.Parameter(torch.zeros((self.channels), dtype=dtype))
    self.highStd:nn.Parameter = nn.Parameter(torch.zeros((self.channels), dtype=dtype))
    
    self.isComplex:bool = torch.is_complex(self.mean)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    # Handle the evaluation of a complex number in a non-complex system`
    inputComplex:bool = torch.is_complex(x)

    # Move channels if needed
    if self.channels > 1:
      x = x.transpose(-2, -1)

    if inputComplex and not self.isComplex:
      real:torch.Tensor = irregularGauss(x=x.real, mean=self.mean, lowStd=self.lowStd, highStd=self.highStd)
      imag:torch.Tensor = irregularGauss(x=x.imag, mean=self.mean, lowStd=self.lowStd, highStd=self.highStd)
      
      # Move channels if needed for reconstruction
      if self.channels > 1:
        real = real.transpose(-1, -2)
        imag = imag.transpose(-1, -2)

      return torch.view_as_complex(torch.stack((real, imag), dim=-1))
    
    # Handle evaluation in a complex system
    if self.isComplex:
      if not inputComplex:
        x = toComplex(x)
      real:torch.Tensor = irregularGauss(x=x.real, mean=self.mean.real, lowStd=self.lowStd.real, highStd=self.highStd.real)
      imag:torch.Tensor = irregularGauss(x=x.imag, mean=self.mean.imag, lowStd=self.lowStd.imag, highStd=self.highStd.imag)

      # Move channels if needed for the reconstruction
      if self.channels > 1:
        real = real.transpose(-1, -2)
        imag = imag.transpose(-1, -2)

      return torch.view_as_complex(torch.stack((real, imag), dim=-1))
    

    # Calculate most default result
    result = irregularGauss(x=x, mean=self.mean, lowStd=self.lowStd, highStd=self.highStd)
    
    # Move channels if needed for return
    if self.channels > 1:
      return result.transpose(-1, -2)
    return result
