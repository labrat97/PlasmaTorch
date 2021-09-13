import torch
import torch.nn as nn

from .distributions import *
from .knots import *

#https://pytorch.org/docs/stable/jit_language_reference.html
from typing import Dict, List, Tuple

DEFAULT_FFT_SAMPLES = 128


# TODO: Continue adding size safety from [ HERE MARK SAFETY SIZES ]


# Entangle a whole bunch of knots into one singular signal
class KnotEntangle(nn.Module):
  def __init__(self, knots:nn.ModuleList, samples:int = DEFAULT_FFT_SAMPLES, lowerSmear:float = 1./8,
    upperSmear:float = 1./8, attn:bool = True, linearPolarization:bool = False, shareSmears:bool = False):
    """Generates the complex required to entangle two seperate knotted signals together.

    Args:
        knots (nn.ModuleList): The knots that define the array to be entangled into one knotted signal.
        samples (int, optional): The amount of FFT samples to use. Defaults to DEFAULT_FFT_SAMPLES.
        lowerSmear (float, optional): The proportion of the input that is smeared backwards. Defaults to 1./8.
        upperSmear (float, optional): The proportion of the input that is smeared forwards. Defaults to 1./8.
        attn (bool, optional): Try to make the input values more represented in the output signal. Defaults to True.
        linearPolarization (bool, optional): Embed a the entangled values with an elementwise knowledgegraph. Defaults to False.
        shareSmears (bool, optional): Share the smear windows between the knots. Defaults to False.
    """
    super(KnotEntangle, self).__init()

    # Set up the knots and assert size constraints
    self.knots = knots
    tCurveSize = self.knots[0].curveSize
    for knot in self.knots:
      assert knot.curveSize == tCurveSize

    # Define FFT and IFFT lead up and execution
    self.samples = samples
    if shareSmears:
      lowerProto = lowerSmear * torch.ones(len(self.knots))
      upperProto = upperSmear * torch.ones(len(self.knots))
      self.smearWindow = nn.Parameter(torch.Tensor([lowerProto, upperProto]), dtype=torch.float16)
    else:
      self.smearWindow = nn.Parameter(torch.Tensor([lowerSmear, upperSmear]), dtype=torch.float16)

    # Provide signal entanglement weighting
    self.entangleActivation = [LinearGauss(1.) for _ in range(self.knots)]
    self.entanglePolarization = nn.Parameter(torch.zeros(len(self.knots)), dtype=torch.float16)

    # If defined, this turns the entanglement function into something that is
    # initially, essentially, a dot product. This knowledge graph on the entanglemeant structure is
    # done per knot, and is referenced from the external entangling knot (as opposed to the
    # local entangled knot).
    self.linPolarization = linearPolarization
    if self.linPolarization:
      self.polKnowledge = nn.Parameter([torch.eye(self.samples) for _ in self.knots], dtype=torch.complex32)

    # Try to pay attention to the input values more than anything, adding some light weighting
    self.attn = attn
    if self.attn:
      self.attnWeight = nn.Parameter(torch.ones(len(self.knots)), dtype=torch.float16)
      self.attnBias = nn.Parameter(torch.zeros(len(self.knots)), dtype=torch.float16)
      self.attnScope = nn.Parameter(torch.ones(len(self.knots)), dtype=torch.float16)

  def knotCount(self) -> int:
    """Gets the amount of knots locked into the entanglement structure.

    Returns:
        int: The amount of knots to be locked into entanglement.
    """
    return len(self.knots)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    # Sizing layout
    inputSize = x.size()
    assert inputSize[-1] == 1 or inputSize[-1] == self.knotCount()
    # Shouldn't need to do any squeezing as the knot propogation forces an unsqueeze

    # Create standardized sampling locations
    lowerSmear = self.smearWindow[0]
    upperSmear = self.smearWindow[1]
    xRange = (upperSmear - lowerSmear) * x
    xStep = xRange / self.samples
    xLow = ((1 - lowerSmear) * x)
    xIter = torch.Tensor([(builder + 1) / self.samples for builder in range(self.samples)], dtype=torch.float16).detach()

    # TODO: Somewhere around here

    # Smear input across the constructed sampling ranges
    knotSmears = []
    knotSignals = []
    for idx, knot in enumerate(self.knots):
      smear = knot.forward((xStep[idx] * xIter) + xLow[idx])
      knotSmears.append(smear)
      knotSignals.append(torch.fft.rfft(smear, self.samples))

    # Entangle
    entangledSmears = []
    for idx, signal in enumerate(knotSignals):
      # Find the entanglements
      smear = knotSmears[idx]
      signal = knotSignals[idx]
      resultSmear = torch.zeros_like(smear)

      for jdx in range(len(self.knots)):
        if idx == jdx: continue

        # Check signal correlation
        subsig = knotSignals[jdx]
        subsigConj = torch.conj(subsig)
        correlation = torch.mean(torch.fft.irfft(signal * subsigConj, self.samples))

        # Entangle signals
        # Note that the weighted activations are tied to each target knot
        entangleMix = self.entangleActivation[jdx].forward(correlation)
        classicalMix = 1 - entangleMix

        # Basing the entangling process of off the use of a tensor product mixed
        # with a sum. To collapse each entangled state, the view from each particle is
        # assessed and the more important one is superimposed into the final signal.
        superposition = (subsig @ torch.transpose(signal)) * self.polKnowledge
        collapseSignal = (torch.sum(superposition), torch.sum(torch.transpose(superposition)))
        collapseSmear = (torch.irfft(collapseSignal[0], self.samples), torch.irfft(collapseSignal[1], self.samples))
        polarization = self.entanglePolarization[jdx]
        entangledSmear = (torch.cos(polarization) * collapseSmear[0]) \
          + (torch.sin(polarization) * collapseSmear[1])

        # Mix the signals together and ensure normalization for what is entangled.
        resultSmear = resultSmear + (entangleMix * entangledSmear)
        resultSmear = resultSmear + (classicalMix * smear)

      # Push to end of calculation
      entangledSmears.append(resultSmear)

    # Collapse into a single knotted time-domain signal definition
    result = torch.sum(entangledSmears)

    # Don't pay attention if that's how you roll
    if not self.attn:
      return result

    # Mix the signal with a gaussian curve to signify the original importance of x 
    # if obscured. The idea is to use this as a way to pay attention to a specific
    # portion of the curve.
    allMeans = (x * self.attnWeight) + self.attnBias
    meansMean = torch.mean(x)
    allLows = (1. - (lowerSmear * self.attnScope)) * meansMean 
    allHighs = (1. + (upperSmear * self.attnScope)) * meansMean
    gaussSamples = ((allHighs - allLows) * xIter) + allLows
    gaussians = []
    for idx in range(len(self.knots)):
      # Pull out specific gaussian parameters
      activeMean = allMeans[idx]
      activeMeans = activeMean * torch.ones_like(result)
      gaussians.append(irregularGauss(x=gaussSamples, mean=activeMeans, lowStd=allLows, highStd=allHighs))
    
    # Apply the psuedo-attention and return
    return torch.sum(gaussians) * result


class KnotConv(nn.Module):
  def __init__(self, knots:nn.ModuleList = None, windowSize:tuple = (32, 32), stepSize:int = 4, samples:int = 128, knotSize:int = 3):
    super(KnotConv, self).__init__()

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
