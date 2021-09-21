from .knots import *
from .distributions import *
from .entanglement import *
from .defaults import *
from .math import *

import torch
import torch.nn as nn
import torch.nn.functional as nnf

class Turbulence(nn.Module):
    def __init__(self, samples:int=DEFAULT_FFT_SAMPLES, internalDimensions:int=DEFAULT_SPACE_PRIME, \
        internalWaves:int=int(DEFAULT_FFT_SAMPLES/2), dtype:torch.dtype=DEFAULT_DTYPE):
        super(Turbulence, self).__init__()

        # Entangle the signals together to get higher order knowledge in smaller spots
        self.samples = samples
        self.parietalEntangler = Entangle(inputSignals=4, curveChannels=internalDimensions, \
            samples=samples, outputMode=EntangleOutputMode.COLLAPSE, useKnowledgeMask=True, \
            dtype=dtype)

        # Knot up the world signal and ego signal
        self.egoKnot = Knot(knotSize=internalDimensions, knotDepth=internalWaves, dtype=dtype)
        self.worldKnot = Knot(knotSize=internalDimensions, knotDepth=internalWaves, dtype=dtype)
        complexType = self.parietalEntangler.knowledgeMask.dtype
        self.integralKnot = Knot(knotSize=internalDimensions, knotDepth=internalWaves, dtype=complexType)
        self.basisKnot = Knot(knotSize=internalDimensions, knotDepth=internalWaves, dtype=complexType)

        # Figure out how to mix the past present and future
        self.warpEntangler = Entangle(inputSignals=internalDimensions, curveChannels=1, \
            samples=samples, useKnowledgeMask=True, outputMode=EntangleOutputMode.COLLAPSE, \
            dtype=dtype)
        self.warpKnot = Knot(knotSize=2, knotDepth=internalWaves, dtype=complexType)

        # Figure out how to mix n dimensions
        self.dimEntangler = Entangle(inputSignals=internalDimensions, curveChannels=1, \
            samples=samples, useKnowledgeMask=True, outputMode=EntangleOutputMode.COLLAPSE, \
            dtype=complexType)

        # Amplify things in the things that are brought into clearer view
        self.compressorKnot = Knot(knotSize=internalDimensions, knotDepth=internalWaves, dtype=complexType)
        self.compressorGain = nn.Parameter(torch.ones(1, dtype=complexType))
    
    def forward(self, queries:torch.Tensor, states:torch.Tensor, inter:str='bicubic', padding:str='border') -> torch.Tensor:
        inputSize = queries.size()
        assert states.size() == inputSize
        
        # Shift both directions in computational elaboration
        integralStates = torch.fft.ifft(states, dim=-1)
        basisStates = torch.fft.fft(states, dim=-1)

        # Entangle the queries and the states together
        egoKnot = self.egoKnot.forward(queries)
        worldKnot = self.worldKnot.forward(states)
        integralKnot = self.integralKnot.forward(integralStates)
        basisKnot = self.basisKnot.forward(basisStates)
        parietalEntanglements = self.parietalEntangler.forward(
            torch.stack([egoKnot, basisKnot, worldKnot, integralKnot], dim=1)
        )
        print(f'entanglement size: \t{parietalEntanglements.size()}')
        entangleSum = parietalEntanglements.sum(dim=1)
        print(f'entanglement sum: \t{entangleSum.size()}')
        superTangle = self.dimEntangler.forward(
            entangleSum.unsqueeze(-2)
        )
        print(f'super tangle: \t{superTangle.size()}')
        superSum = superTangle.sum(dim=1)
        print(f'super sum: \t{superSum.size()}')

        
        # Pay attention using spatial warping and basis vector compression
        warpKnot = isoftmax(self.warpKnot.forward(superSum), dim=-2)
        print(f'warp knot size: \t {warpKnot.size()}')
        compressorKnot = isoftmax(self.compressorKnot.forward(superSum), dim=-2) * self.compressorGain
        compressorKnot.squeeze_(dim=1)
        print(f'compressor size: \t{compressorKnot.size()}')

        # Warping as if the state vector is 4D image data as seen here:
        # https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html#torch.nn.functional.grid_sample
        # Because the data is currently in [BATCH, Hin, C, Win], transpose is needed
        stateEntanglements = parietalEntanglements.transpose(1, 2)
        print(f'state entangle: \t{stateEntanglements.size()}')
        # Now in [BATCH, C, Hin, Win]

        # To warp the view of all levels of iterable signal complexity,
        # a tensor of size [BATCH, 1, samples, 2] is needed in the space of the
        # 'grid' param. This is done with the warp knot ([BATCH, curve, samples])
        # which needs to be in the shape [BATCH, 1, samples, curve].
        warpGrid = warpKnot.transpose(-1, -2)
        print(f'warp grid: \t{warpGrid.size()}')
        warpedStateReal = nnf.grid_sample(stateEntanglements.real, grid=warpGrid.real, mode=inter, align_corners=False).unsqueeze(-1)
        warpedStateImag = nnf.grid_sample(stateEntanglements.imag, grid=warpGrid.imag, mode=inter, align_corners=False).unsqueeze(-1)
        warpedState = torch.view_as_complex(torch.cat((warpedStateReal, warpedStateImag), dim=-1))
        print(f'warped state: \t{warpedState.size()}')

        # Now <warpedState> must be translated back to the format that the network
        # is expecting. The current format should be still be [BATCH, C, 1, samples],
        # but it needs to be [BATCH, C, samples], so just squeeze.
        warpedState.squeeze_(-2)
        print(f'squeezed warped state: \t{warpedState.size()}')

        # Find what makes the variably 'zoomed' signal, and modify it with the
        # compressor-like signal evaluated earlier. If things are done right here,
        # comrpessor should broadcast across the warped signal.
        warpedSignal = torch.fft.fft(warpedState, n=self.samples, dim=-1)
        print(f'warped signal: \t{warpedSignal.size()}')
        warpCompSignal = warpedSignal * compressorKnot
        print(f'warp comp signal:\t{warpCompSignal.size()}')

        # Return the signal that is constructed from the final computations. This
        # signal is back into the constructed 'current' domain.
        return torch.fft.ifft(warpCompSignal, n=self.samples, dim=-1)
