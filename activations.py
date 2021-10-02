import torch
import torch.nn as nn

from .defaults import *
from .math import *

from typing import List


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

    self.size:int = size
    self.frequency:nn.Parameter = nn.Parameter(torch.zeros([1, size], dtype=dtype))
    self.phase:nn.Parameter = nn.Parameter(torch.zeros([1, size], dtype=dtype))

  def forward(self, x:torch.Tensor, oneD:bool = True) -> torch.Tensor:
    """Gets a sample or batch of samples from the contained curve.

    Args:
        x (torch.Tensor): The sample or sampling locations. If dim[-2] == self.size,
          the input curve is believed to have the same amount of curves as the function.
          When this is the case, instead of taking a 1D input

    Returns:
        torch.Tensor: The evaluted samples.

          [BATCHES...,Samples] -> [BATCHES...,Curves,Samples]
    """
    if oneD:
      # Manipulate dimensions to broadcast in 1D sense
      x = torch.unsqueeze(x, -1)
      cosPos:torch.Tensor = (x @ self.frequency) + (torch.ones_like(x) @ self.phase)
    else:
      # Put curves in the right spot
      assert x.size()[-2] == self.size
      x = x.transpose(-1,-2)

      # Maniupulate dimensions to broadcast in per-curve sense
      freq:torch.Tensor = self.frequency.squeeze(0)
      phase:torch.Tensor = self.phase.squeeze(0)
      cosPos:torch.Tensor = (x * freq) + (torch.ones_like(x) * phase)

    # Activate in curve's embedding space depending on the working datatype.
    # This is done due to the non-converging nature of the non-convergence of the
    # cos function during the operation on complex numbers. To solve this, a sin function
    # is called in the imaginary place to emulate the e^ix behavior for sinusoidal signals.
    return icos(cosPos).transpose(-1, -2)


class Knot(nn.Module):
  """
  Creates a Lissajous-Knot-like structure for encoding information. All information
    stored in the knot is stored in the form of a multidimensional fourier series,
    which allows the knot to have its parameters later entangled, modulated, and
    transformed through conventional methods.
  """
  def __init__(self, knotSize:int, knotDepth:int, dtype:torch.dtype=DEFAULT_DTYPE):
    """Constructs a Knot for later use generating all weights and storing internally.

    Args:
        knotSize (int): The dimensionality of the contained lissajous-like curves.
        knotDepth (int): The amount of lissajous-like curves to be added together.
        dtype (torch.dtype): The type of the housed parameters used for modifying
          the value of the contained lissajous structures.
    """
    super(Knot, self).__init__()

    # Set up the curves for the function
    self.curves:nn.ModuleList = nn.ModuleList([Lissajous(size=knotSize, dtype=dtype) for _ in range(knotDepth)])
    self.curveSize:int = self.curves[0].size

    # Size assertion
    for curve in self.curves:
      assert curve.size == self.curveSize

    # Add some linearly trained weighted goodness
    self.dtype:torch.dtype = dtype
    paramSize:List[int] = [len(self.curves), self.curveSize, 1]
    self.regWeights:nn.Parameter = nn.Parameter(torch.zeros(paramSize, dtype=dtype))
    self.knotRadii:nn.Parameter = nn.Parameter(torch.zeros(paramSize[1:], dtype=dtype))

  def forward(self, x:torch.Tensor, oneD:bool = True) -> torch.Tensor:
    """Pushed forward the same way as the Lissajous module. This is just an array
    of Lissajous modules summed together in a weighted way.

    Args:
        x (torch.Tensor): The points to sample on the curves.
        oneD (bool): Evaluate the tensor as if it is one dimensional (curves from 1 curve). Defaults to True.

    Returns:
        torch.Tensor: The original size tensor, but every point has a Lissajous curve
          activated upon it. There will be one extra dimension that is the same in size
          as the dimensions of the curve.

          [Batches,::,Samples] -> [Batches,::,Curves,Samples]
    """
    # Create the expanded dimensions required in the output tensor
    if oneD:
      outputSize:torch.Size = torch.Size(list(x.size()) + [self.curveSize])
      result:torch.Tensor = torch.zeros(outputSize, dtype=self.dtype).transpose(-1, -2)
    else:
      outputSize:torch.Size = x.size()
      result:torch.Tensor = torch.zeros(outputSize, dtype=self.dtype)

    # Add all of the curves together
    for idx, lissajous in enumerate(self.curves):
      # Each lissajous curve-like structure has different weights, and therefore 
      curve:torch.Tensor = self.regWeights[idx] * lissajous.forward(x, oneD=oneD)
      result.add_(curve)

    # Add the radius of the knot to the total of the sum of the curves
    result.add_(self.knotRadii)
    
    # Swap the position of the curve and the sample (so the samples are on the rear)
    return result