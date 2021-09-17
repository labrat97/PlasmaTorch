from .knots import *
from .distributions import *
from .entanglement import *

import torch
import torch.nn as nn

class Turbulence(nn.Module):
    def __init__(self, samples:int=DEFAULT_FFT_SAMPLES, internalDimensions:int=11, internalWaves:int=int(DEFAULT_FFT_SAMPLES/4)):
        super(Turbulence, self).__init__()

        self.egoKnowledge = Knot(knotSize=internalDimensions, knotDepth=internalWaves)
        self.worldKnowledge = Knot(knotSize=internalDimensions, knotDepth=internalWaves)
        self.mixingKnowledge = Knot(knotSize=internalDimensions, knotDepth=internalWaves)
        self.warpingKnowledge = Knot(knotSize=internalDimensions, knotDepth=internalWaves)

        self.parietalCollapser = KnotEntangle([self.egoKnowledge, self.worldKnowledge], \
            samples=samples, attn=True, linearPolarization=True, shareSmears=False)

    
    def forward(self, queries:torch.Tensor, states:torch.Tensor) -> torch.Tensor:
        orientation = self.parietalCollapser.forward(torch.tensor([queries, states]))
        # TODO: Go back to entanglement and expose entangle functionality without weighting

