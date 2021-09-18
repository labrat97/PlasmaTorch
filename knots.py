import torch
import torch.nn as nn

from .defaults import *

class Lissajous(nn.Module):
  """
  Holds a Lissajous-like curve to be used as a sort of activation layer as a unit
    of knowledge.
  """
  def __init__(self, size:int, dtype:torch.dtype = DEFAULT_DTYPE):
    """Builds a new Lissajous-like curve structure.

    Args:
        size (int): The amount of dimensions encoded in the curve.
    """
    super(Lissajous, self).__init__()

    self.size = size
    self.frequency = nn.Parameter(torch.zeros([1, size], dtype=dtype))
    self.phase = nn.Parameter(torch.zeros([1, size], dtype=dtype))

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Gets a sample or batch of samples from the contained curve.

    Args:
        x (torch.Tensor): The sample or sampling locations.

    Returns:
        torch.Tensor: The evaluted samples.

          [::,Samples] -> [::,Samples,Curves]
    """
    # Add another dimension to do the batch of encodes
    xFat = torch.unsqueeze(x, -1)
    xOnes = torch.ones_like(xFat)

    # Activate inside of the curve's embedding space
    cosinePosition = (xFat @ self.frequency) + (xOnes @ self.phase)
    evaluated = torch.cos(cosinePosition)

    return evaluated


class Knot(nn.Module):
  """
  Creates a Lissajous-Knot-like structure for encoding information. All information
    stored in the knot is stored in the form of a multidimensional fourier series,
    which allows the knot to have its parameters later entangled, modulated, and
    transformed through conventional methods.
  """
  def __init__Helper(self, lissajousCurves:nn.ModuleList, dtype:torch.dtype):
    """Does the actual __init__ work for super() call reasons.

    Args:
        lissajousCurves (nn.ModuleList): The curves to add together to create the
          knot.
        dtype (torch.dtype): The type of the housed parameters used for modifying
          the value of the contained lissajous structures.
    """
    # Set up the curves for the function
    self.curves = lissajousCurves
    self.curveSize = self.curves[0].size

    # Size assertion
    for curve in self.curves:
      assert curve.size == self.curveSize

    self.dtype = dtype
    paramSize = (len(self.curves), self.curveSize)
    self.regWeights = nn.Parameter(torch.zeros(paramSize, dtype=dtype))
    self.knotRadii = nn.Parameter(torch.zeros(self.curveSize, dtype=dtype))

  def __init__(self, lissajousCurves:nn.ModuleList, dtype:torch.dtype=DEFAULT_DTYPE):
    """Constructs a Knot for later use from previously constructed Lissajous curves.

    Args:
        lissajousCurves (nn.ModuleList): The Lissajous curves to add together to make the knot.
        dtype (torch.dtype): The type of the housed parameters used for modifying
          the value of the contained lissajous structures.
    """
    super(Knot, self).__init__()

    # Call helper init function
    self.__init__Helper(lissajousCurves=lissajousCurves, dtype=dtype)    

  def __init__(self, knotSize:int, knotDepth:int, dtype:torch.dtype=DEFAULT_DTYPE):
    """Constructs a Knot for later use generating all weights and storing internally.

    Args:
        knotSize (int): The dimensionality of the contained lissajous-like curves.
        knotDepth (int): The amount of lissajous-like curves to be added together.
        dtype (torch.dtype): The type of the housed parameters used for modifying
          the value of the contained lissajous structures.
    """
    super(Knot, self).__init__()

    # Construct and call helper function
    curves = nn.ModuleList([Lissajous(size=knotSize) for _ in range(knotDepth)])
    self.__init__Helper(lissajousCurves=curves, dtype=dtype)

  # TODO: Add a method to add more curves, it would be cool to have a hyperparameter
  #   that makes the neural network hold more data in almost the same space

  def forward(self, x:torch.Tensor) -> torch.Tensor:
    """Pushed forward the same way as the Lissajous module. This is just an array
    of Lissajous modules summed together in a weighted way.

    Args:
        x (torch.Tensor): The points to sample on the curves.

    Returns:
        torch.Tensor: The original size tensor, but every point has a Lissajous curve
          activated upon it. There will be one extra dimension that is the same in size
          as the dimensions of the curve.

          [Batches,::,Samples] -> [Batches,::,Curves,Samples]
    """
    # Create the expanded dimensions required in the output tensor
    outputSize = torch.Size(list(x.size()) + [self.curveSize])
    result = torch.zeros(outputSize, dtype=self.dtype)

    # Add all of the curves together
    for idx, lissajous in enumerate(self.curves):
      # Each lissajous curve-like structure has different weights, and therefore 
      curve = self.regWeights[idx] * lissajous.forward(x)
      result.add_(curve)
    
    # Swap the position of the curve and the sample (so the samples are on the rear)
    return result.transpose(-1, -2) + self.knotRadii