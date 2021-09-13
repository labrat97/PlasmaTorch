import torch
import torch.nn as nn

class Lissajous(nn.Module):
  """
  Holds a Lissajous-like curve to be used as a sort of activation layer as a unit
    of knowledge.
  """
  def __init__(self, size: int):
    """Builds a new Lissajous-like curve structure.

    Args:
        size (int): The amount of dimensions encoded in the curve.
    """
    super(Lissajous, self).__init__()

    self.size = size
    self.frequency = nn.Parameter(torch.zeros([1, size]), dtype=torch.float16)
    self.phase = nn.Parameter(torch.zeros([1, size]), dtype=torch.float16)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Gets a sample or batch of samples from the contained curve.

    Args:
        x (torch.Tensor): The sample or sampling locations.

    Returns:
        torch.Tensor: The evaluted samples.
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
  def ___init__Helper(self, lissajousCurves: nn.ModuleList):
    """Does the actual __init__ work for super() call reasons.

    Args:
        lissajousCurves (nn.ModuleList): The curves to add together to create the
          knot.
    """
    # Set up the curves for the function
    self.curves = lissajousCurves
    self.curveSize = self.curves[0].size

    # Size assertion
    for curve in self.curves:
      assert curve.size == self.curveSize

    paramSize = (len(self.curves), self.curveSize)
    self.regWeights = nn.Parameter(torch.ones(paramSize), dtype=torch.float16)
    self.knotRadii = nn.Parameter(torch.zeros(self.curveSize), dtype=torch.float16)

  def __init__(self, lissajousCurves: nn.ModuleList):
    """Constructs a Knot for later use from previously constructed Lissajous curves.

    Args:
        lissajousCurves (nn.ModuleList): The Lissajous curves to add together to make the knot.
    """
    super(Knot, self).__init__()

    # Call helper init function
    self.___init___Helper(lissajousCurves=lissajousCurves)    

  def __init__(self, knotSize: int, knotDepth: int):
    """Constructs a Knot for later use generating all weights and storing internally.

    Args:
        knotSize (int): The dimensionality of the contained lissajous-like curves.
        knotDepth (int): The amount of lissajous-like curves to be added together.
    """
    super(Knot, self).__init__()

    # Construct and call helper function
    curves = nn.ModuleList([Lissajous(size=knotSize) for _ in range(knotDepth)])
    self.___init___Helper(lissajousCurves=curves)

  # TODO: Add a method to add more curves, it would be cool to have a hyperparameter
  #   that makes the neural network hold more data in almost the same space

  @torch.jit.script
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Pushed forward the same way as the Lissajous module. This is just an array
    of Lissajous modules summed together in a weighted way.

    Args:
        x (torch.Tensor): The points to sample on the curves.

    Returns:
        torch.Tensor: The original size tensor, but every point has a Lissajous curve
          activated upon it. There will be one extra dimension that is the same in size
          as the dimensions of the curve.
    """
    # Create the expanded dimensions required in the output tensor
    outputSize = torch.Size(list(x.size()).append(self.curveSize))
    result = torch.Tensor(torch.zeros(outputSize), dtype=torch.float16)

    # Add all of the curves together
    for idx, lissajous in enumerate(self.curves):
      # Each lissajous curve-like structure has different weights, and therefore 
      curve = lissajous.forward(x)
      curve = self.regWeights[idx] * curve
      result = result + curve
    
    return result + self.knotRadii