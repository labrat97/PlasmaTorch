import torch
import torch.nn as nn

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
  def __init__(self, size: torch.Size):
    """Builds a new LinearGauss structure.

    Args:
        size (torch.Size): This size must be broadcastable towards the later used
          input tensor.
    """
    super(LinearGauss, self).__init__()

    self.size = size
    self.mean = nn.Parameter(torch.zeros(size), dtype=torch.float16)
    self.lowStd = nn.Parameter(torch.zeros(size), dtype=torch.float16)
    self.highStd = nn.Parameter(torch.zeros(size), dtype=torch.float16)

  def forward(self, x: torch.Tensor):
    return irregularGauss(x=x, mean=self.mean, lowStd=self.lowStd, highStd=self.highStd)
