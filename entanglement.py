import torch
import torch.nn as nn

from .distributions import *
from .knots import *
from .defaults import *
from .conversions import *
from .math import *

from enum import Flag


class EntangleOutputMode(int, Flag):
  SUPERPOSITION:int = 1 << 0
  COLLAPSE:int = 1 << 1
  BOTH:int = SUPERPOSITION | COLLAPSE

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
      # This should broadcast an identity matrix over the knowledge mask for collapsing
      iEye = toComplex(torch.eye(samples, dtype=dtype))
      self.knowledgeMask = nn.Parameter(
        toComplex(torch.zeros((inputSignals, curveChannels, samples, samples), dtype=dtype)) \
        + iEye)
  
  def forward(self, x:torch.Tensor) -> torch.Tensor:
    # Define some constants
    SAMPLE_POS = -1
    CURVE_POS = -2
    COUNT_POS = -3

    # Check to make sure that x is of compatible shape
    inputSize = x.shape
    inputSizeLen = len(inputSize)
    if inputSizeLen == 3: 
      x = x.unsqueeze(0)
    assert inputSizeLen == 4
    assert inputSize[SAMPLE_POS] == self.samples
    assert inputSize[CURVE_POS] == self.curveChannels
    assert inputSize[COUNT_POS] == self.signalCount
    isComplex = torch.is_complex(x)

    # Find out what the signals are made of
    signals = torch.fft.fft(x, n=self.samples, dim=SAMPLE_POS)

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
        correlation = torch.mean(
          torch.fft.irfft(signal * subconj, n=self.samples, dim=SAMPLE_POS),
        dim=SAMPLE_POS)

        # Create a superposition through a tensor product
        superposition = signal.unsqueeze(-1) @ torch.transpose(subsig.unsqueeze(-1), -2, -1)

        # Apply knowledge to the superposition of the subsignals if requested
        if self.knowledgeMask is not None:
          superposition = superposition * isoftmax(self.knowledgeMask[jdx], dim=-2)

        # Save superposition for output if needed
        if (int(self.outputMode) & int(EntangleOutputMode.SUPERPOSITION)) != 0:
          s[:,idx].add_(superposition)

        # No need to collapse
        if (int(self.outputMode) & int(EntangleOutputMode.COLLAPSE)) == 0:
          continue

        # Act on correlation for collapse
        entangleMix = self.entangleActivation[idx].forward(correlation).unsqueeze(-1)
        classicalMix = 1 - entangleMix

        # Collapse
        collapseSignal = (torch.sum(superposition, dim=-2), torch.sum(torch.transpose(superposition, -2, -1), dim=-2))
        if isComplex:
          collapseSmear = (torch.fft.ifft(collapseSignal[0], n=self.samples, dim=SAMPLE_POS), \
            torch.fft.ifft(collapseSignal[1], n=self.samples, dim=SAMPLE_POS))
        else:
          collapseSmear = (torch.fft.irfft(collapseSignal[0], n=self.samples, dim=SAMPLE_POS), \
            torch.fft.irfft(collapseSignal[1], n=self.samples, dim=SAMPLE_POS))
        entangledSmear = (torch.cos(polarization) * collapseSmear[0]) \
          + (torch.sin(polarization) * collapseSmear[1])

        # Put into output for signals
        y[:,idx].add_(
          ((entangleMix * entangledSmear) + (classicalMix * x[:,idx]))
        )
    
    # Regularize
    if (int(self.outputMode) & int(EntangleOutputMode.COLLAPSE)) != 0:
      y.div_(self.signalCount)
    if (int(self.outputMode) & int(EntangleOutputMode.SUPERPOSITION)) != 0:
      s.div_(self.signalCount)

    # Return
    if self.outputMode == EntangleOutputMode.COLLAPSE:
      return y
    if self.outputMode == EntangleOutputMode.SUPERPOSITION:
      return s
    return y, s
