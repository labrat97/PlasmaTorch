import torch
import torch.nn as nn

from .defaults import *

@torch.jit.script
def irregularGauss(x: torch.Tensor, mean: torch.Tensor, lowStd: torch.Tensor, highStd: torch.Tensor) -> torch.Tensor:
  """Generates an piecewise Gaussian curve according to the provided parameters.

  Args:
      x (torch.Tensor): The sampling value for the curve with indefinite size.
      mean (torch.Tensor): The means that generate the peaks of the function which
        has a shape that is broadcastable upon x.
      lowStd (torch.Tensor): The standard deviation to use when the function is below
        the defined mean. The size must be broadcastable upon x.
      highStd (torch.Tensor): The standard deviation to use when the function is
        above the defined mean. The size must be broadcastable upon x.

  Returns:
      torch.Tensor: A sampled set of values with the same size as the input.
  """
  # Grab the correct side of the curve
  if x <= mean: std = lowStd
  else: std = highStd

  # Never hits 0 or inf., easy to take derivative
  std = torch.exp(std)

  # Calculate the gaussian curve
  top = torch.square(x - mean)
  bottom = torch.square(std)
  return torch.exp((-0.5) * (top / bottom))

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

    self.channels = channels
    self.mean = nn.Parameter(torch.zeros((channels), dtype=dtype))
    self.lowStd = nn.Parameter(torch.zeros((channels), dtype=dtype))
    self.highStd = nn.Parameter(torch.zeros((channels), dtype=dtype))

  def forward(self, x: torch.Tensor):
    return irregularGauss(x=x, mean=self.mean, lowStd=self.lowStd, highStd=self.highStd)
