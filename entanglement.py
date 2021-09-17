import torch
import torch.nn as nn

from .distributions import *
from .knots import *
from .defaults import *

from enum import Enum

# TODO: Continue adding size safety from [ HERE MARK SAFETY SIZES ]

# Turn a pointwise signal into a smearwise one
class Smear(nn.Module):
  def __init__(self, samples:int = DEFAULT_FFT_SAMPLES, lowerScalar:float = 1./16, 
    upperScalar:float = 1./16, dtype:torch.dtype = DEFAULT_DTYPE):
    super(Smear, self).__init__()

    self.__iter = torch.Tensor([builder / (self.samples-1) for builder in range(self.samples)], \
      dtype=dtype).detach()

    self.samples = samples
    self.smearBias = nn.Parameter(torch.zeros(1, dtype=dtype))
    self.smearWindow = nn.Parameter(torch.Tensor([lowerScalar, upperScalar], dtype=dtype))
  
  def forward(self, x:torch.Tensor) -> torch.Tensor:
    xBias = x + self.smearBias
    if self.samples <= 1:
      return xBias

    lowerSmear = self.smearWindow[0]
    upperSmear = self.smearWindow[1]
    xRange = (upperSmear - lowerSmear) * xBias
    xLow = ((1 - lowerSmear) * xBias)

    return (xRange * self.__iter) + xLow

class EntangleOutputMode(Enum):
  SUPERPOSITION = 1 << 0
  COLLAPSE = 1 << 1
  BOTH = SUPERPOSITION | COLLAPSE

class Entangle(nn.Module):
  def __init__(self, inputSignals:int, curveChannels:int = DEFAULT_SPACE_PRIME, \
    samples:int = DEFAULT_FFT_SAMPLES, useKnowledgeMask:bool = True, \
    outputMode:EntangleOutputMode = EntangleOutputMode.BOTH, dtype:torch.dtype = DEFAULT_DTYPE):
    super(Entangle, self).__init__()

    # Store data about the signals going into/out of the module
    self.signalCount = inputSignals
    self.curveChannels = curveChannels
    self.samples = samples
    self.outputMode = outputMode

    # Hold the entanglement parameters
    self.entangleActivation = [LinearGauss(1, dtype=dtype) for _ in range(inputSignals)]
    self.entanglePolarization = nn.Parameter(torch.zeros(
      (inputSignals), dtype=dtype
    ))

    # If requested, use a knowledge mask at the end of the forward() call
    self.knowledgeMask:nn.Parameter = None
    if useKnowledgeMask:
      self.knowledgeMask = nn.Parameter(torch.view_as_complex(torch.ones(
        (inputSignals, curveChannels, samples, samples), dtype=dtype
      )))
  
  def forward(self, x:torch.Tensor) -> torch.Tensor:
    # Check to make sure that x is of compatible shape
    inputSize = x.shape
    inputSizeLen = len(inputSize)
    if inputSizeLen == 3: 
      x = x.unsqueeze(0)
    assert inputSizeLen == 4
    assert inputSize[-1] == self.samples
    assert inputSize[-2] == self.curveChannels
    assert inputSize[-3] == self.signalCount
    isComplex = torch.is_complex(x)

    # Find out what the signals are made of
    signals = torch.fft.fft(x, n=self.samples, dim=-1)

    # Store where the signals are going
    y = torch.zeros_like(x)
    s = torch.zeros((inputSize[0], self.signalCount, self.curveChannels, self.samples, self.samples))
    for idx in range(self.signalCount):
      signal = signals[:,idx]
      polarization = self.entanglePolarization[idx]

      for jdx in range(self.signalCount):
        # See how similar each signal is
        subsig = signals[:,jdx]
        subconj = torch.conj(subsig)
        correlation = torch.mean(torch.mean(
          torch.irfft(signal * subconj, n=self.samples, dim=-1)
        ))

        # Create a superposition through a tensor product
        superposition = signal @ torch.transpose(signal)

        # Apply knowledge to the superposition of the subsignals if requested
        if self.knowledgeMask is not None:
          superposition = superposition * self.knowledgeMask

        # Save superposition for output if needed
        if (self.outputMode & EntangleOutputMode.SUPERPOSITION) != 0:
          s[:,idx].add_(superposition)

        # No need to collapse
        if (self.outputMode & EntangleOutputMode.COLLAPSE) == 0:
          continue

        # Act on correlation for collapse
        entangleMix = self.entangleActivation[idx].forward(correlation)
        classicalMix = 1 - entangleMix

        # Collapse
        collapseSignal = (torch.sum(superposition), torch.sum(torch.transpose(superposition)))
        if isComplex:
          collapseSmear = (torch.ifft(collapseSignal[0], n=self.samples, dim=-1), \
            torch.ifft(collapseSignal[1], n=self.samples, dim=-1))
        else:
          collapseSmear = (torch.irfft(collapseSignal[0], n=self.samples, dim=-1), \
            torch.irfft(collapseSignal[1], n=self.samples, dim=-1))
        entangledSmear = (torch.cos(polarization) * collapseSmear[0]) \
          + (torch.sin(polarization) * collapseSmear[1])

        # Put into output for signals
        y[:,idx].add_(
          ((entangleMix * entangledSmear) + (classicalMix * x))
        )
    
    # Regularize
    if (self.outputMode & EntangleOutputMode.COLLAPSE) != 0:
      y.div_(self.signalCount)
    if (self.outputMode & EntangleOutputMode.SUPERPOSITION) != 0:
      s.div_(self.signalCount)

    # Return
    if self.outputMode == EntangleOutputMode.COLLAPSE:
      return y
    if self.outputMode == EntangleOutputMode.SUPERPOSITION:
      return s
    return y, s


class Enzyme(nn.Module):
  def __init__(self, knots:nn.ModuleList = None, windowSize:tuple = (32, 32), stepSize:int = 4, samples:int = 128, knotSize:int = 3):
    super(Enzyme, self).__init__()

    self.windowSize = torch.Tensor(windowSize)
    self.stepSize = stepSize * torch.ones_like(self.windowSize)
    self.samples = samples
    self.knotSize = knotSize

    flatWindow = 1
    for n in windowSize:
      flatWindow = flatWindow * n

    self.knots = knots
    if self.knots != None:
      assert len(self.knots) == flatWindow
      for knot in self.knots:
        assert knot.curveSize == self.knotSize
    else:
      self.knots = nn.ModuleList([Knot(knotSize=knotSize, knotDepth=samples/4.) for i in range(flatWindow)])
