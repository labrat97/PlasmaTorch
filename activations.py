import torch
import torch.nn as nn

from .defaults import *
from .conversions import *
from .math import *

from typing import List


def lissajous(x:torch.Tensor, freqs:torch.Tensor, phases:torch.Tensor, oneD:bool = True):
  assert freqs.size() == phases.size()
  
  if oneD:
    # Manipulate dimensions to broadcast in 1D sense
    x = torch.unsqueeze(x, -1)
    cosPos:torch.Tensor = (x @ freqs) + (torch.ones_like(x) @ phases)
  else:
    # Put curves in the right spot
    assert x.size()[-2] == freqs.size()[-1]
    x = x.transpose(-1,-2)

    # Maniupulate dimensions to broadcast in per-curve sense
    cosPos:torch.Tensor = (x * freqs.unsqueeze(0)) + (torch.ones_like(x) * phases.unsqueeze(0))

  # Activate in curve's embedding space depending on the working datatype.
  # This is done due to the non-converging nature of the non-convergence of the
  # cos function during the operation on complex numbers. To solve this, a sin function
  # is called in the imaginary place to emulate the e^ix behavior for sinusoidal signals.
  return icos(cosPos).transpose(-1, -2)

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
          When this is the case, instead of taking a 1D input.
        oneD (bool): If true, expand every leaf logit into the required amount of
          internal signals.

    Returns:
        torch.Tensor: The evaluted samples.

          [BATCHES...,Samples] -> [BATCHES...,Curves,Samples]
    """
    return lissajous(x, freqs=self.frequency, phases=self.phase, oneD=oneD)


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
    self.knotDepth = knotDepth
    self.knotSize = knotSize

    # Add some linearly trained weighted goodness
    self.dtype:torch.dtype = dtype
    paramSize:List[int] = [self.knotDepth, self.knotSize, 1]
    self.regWeights:nn.Parameter = nn.Parameter(torch.ones(paramSize, dtype=dtype) / self.knotDepth)
    
    self.frequencies:nn.Parameter = nn.Parameter(torch.zeros((self.knotSize, self.knotDepth), dtype=dtype))
    self.phases:nn.Parameter = nn.Parameter(torch.zeros((self.knotSize, self.knotDepth), dtype=dtype))
    self.__triu:torch.Tensor = torch.triu(torch.ones((self.knotDepth, self.knotDepth), dtype=dtype), diagonal=0).detach()
    self.__latticeParams:torch.Tensor = latticeParams(self.knotDepth)

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
      outputSize:torch.Size = torch.Size(list(x.size()) + [self.knotSize])
      result:torch.Tensor = torch.zeros(outputSize, dtype=self.dtype).transpose(-1, -2)
    else:
      outputSize:torch.Size = x.size()
      result:torch.Tensor = torch.zeros(outputSize, dtype=self.dtype)
    
    # Add the frequencies together
    freqs:torch.Tensor = ((self.frequencies * self.__latticeParams) @ self.__triu).transpose(0, 1)
    phases:torch.Tensor = ((self.phases * self.__latticeParams) @ self.__triu).transpose(0, 1)

    # Add all of the curves together
    for idx in range(self.knotDepth):
      # Pass the frequencies to the curves
      freqn:torch.Tensor = freqs[idx].unsqueeze(0)
      phasen:torch.Tensor = phases[idx].unsqueeze(0)
      regn:torch.Tensor = self.regWeights[idx]

      # Each lissajous curve-like structure has different weights, and therefore 
      curve:torch.Tensor = regn * lissajous(x=x, freqs=freqn, phases=phasen, oneD=oneD)
      result.add_(curve)

    # Add the radius of the knot to the total of the sum of the curves
    result.add_(self.knotRadii)
    
    # Swap the position of the curve and the sample (so the samples are on the rear)
    return result

class Ringing(nn.Module):
  """
  Creates a structure that acts as a set of tuning forks, dampening over time. Because
    time is not really relevant here, this is actually dampening over forward iteration
    unless specified not to.
  """
  def __init__(self, forks:int=DEFAULT_FFT_SAMPLES, dtype:torch.dtype=DEFAULT_COMPLEX_DTYPE):
    super(Ringing, self).__init__()

    # The positions and values of the enclosed forks
    forks = int(forks)
    DECAY_SEED = (-torch.log(phi() - 1)).type(dtype) # After a sigmoid eval this should come to 1/phi()
    self.forkPos = nn.Parameter(toComplex(torch.zeros((forks), dtype=dtype)).real)
    self.forkVals = toComplex(torch.zeros((forks), dtype=dtype, requires_grad=False))
    self.forkDecay = nn.Parameter(torch.ones((forks), dtype=dtype) * DECAY_SEED)
    self.signalDecay = nn.Parameter(torch.ones((1), dtype=dtype) * DECAY_SEED)

  def __createOutputSignal(self, xfft:torch.Tensor, posLow:torch.Tensor, posHigh:torch.Tensor, posMix:torch.Tensor) -> torch.Tensor:
    yfft = torch.zeros_like(xfft)
    yfft[..., posLow] = (1 - posMix) * self.forkVals
    yfft[..., posHigh] = posMix * self.forkVals
    yfft.add_(xfft * isigmoid(self.signalDecay))

    return yfft
  
  def dampen(self, stop:bool=False):
    # Don't modify in-place
    result:torch.Tensor = self.forkVals

    # If stopping, fully decaying
    if stop:
      self.forkVals = self.forkVals * 0
    # Regular 1/phi() decay
    else:
      self.forkVals = self.forkVals * isigmoid(self.forkDecay)

    return result
    

  def view(self, samples:int=DEFAULT_FFT_SAMPLES, irfft:bool=False) -> torch.Tensor:
    # Generate metadata needed to create the output signal
    positions = isigmoid(self.forkPos) * (samples - 1)
    posLow = positions.type(torch.int64)
    posHigh = posLow + 1
    posMix = positions - posLow
    xfft = torch.zeros((samples), dtype=self.forkVals.dtype)

    # Generate the output signal
    yfft = self.__createOutputSignal(xfft=xfft, posLow=posLow, posHigh=posHigh, posMix=posMix)

    # Real or complex output
    if irfft:
      return torch.fft.irfft(yfft, n=samples, dim=-1)
    return torch.fft.ifft(yfft, n=samples, dim=-1)

  def forward(self, x:torch.Tensor, irfft:bool=False, stopTime:bool=False) -> torch.Tensor:
    # Gather parameters needed to have some light attention to the tunes coming in
    xfft = torch.fft.fft(x, dim=-1)
    xsamples = x.size()[-1]
    positions = isigmoid(self.forkPos) * (xsamples - 1)

    # Extract the target parameters from the signal
    posLow = positions.type(torch.int64)
    posHigh = posLow + 1
    posMix = positions - posLow # [1, 0] -> [HIGH, 1-LOW]
    xvals = ((1 - posMix) * xfft[..., posLow]) + (posMix * xfft[..., posHigh])

    # Add the input signals to the enclosed signals
    forkVals = (self.forkVals * isigmoid(self.forkDecay)) + xvals
    if not stopTime:
      self.forkVals = forkVals
    
    # Create the output signal
    yfft = self.__createOutputSignal(xfft=xfft, posLow=posLow, posHigh=posHigh, posMix=posMix)

    # Real or complex output
    if irfft:
      return torch.fft.irfft(yfft, n=xsamples, dim=-1)
    return torch.fft.ifft(yfft, n=xsamples, dim=-1)
