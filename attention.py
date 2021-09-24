from .activations import *
from .distributions import *
from .entanglement import *
from .defaults import *
from .math import *

import torch
import torch.nn as nn
import torch.nn.functional as nnf

class Turbulence(nn.Module):
    """
    Pays attention to a continuous signal by multiple methods. Three methods are primarily used, the
    first being calculus based modulation through an observation of the feeding signal in the normal domain,
    fft domain, and ifft domain. The second layer implemented is frequency and phase modulation through grid remapping.
    By using this mapping method, the network can essentially choose to zoom in on certain subsections, giving downstream
    parameters invariance over the entire observation space of the signal (potentially). The final chunk of computation used
    for attention is amplitude modulation of the basis vector of the signal (derived by a pass through
    a fft, an element-wise multiplication, and a pass back through an ifft).
    """
    def __init__(self, samples:int=DEFAULT_FFT_SAMPLES, internalDimensions:int=DEFAULT_SPACE_PRIME, \
        internalWaves:int=int(DEFAULT_FFT_SAMPLES/2), sameDimOut:bool=False, sameDimWarpEntangle:bool=False, \
        dtype:torch.dtype=DEFAULT_DTYPE):
        """Constructs a new Turbulence attention module.

        Args:
            samples (int, optional): The amount of samples to use for the natural signal. DefaultDefaults to DEFAULT_FFT_SAMPLES.
            internalDimensions (int, optional): The amount of curves contained in each knot. DefaultDefaults to DEFAULT_DEFAULT_SPACE_PRIME.
            sameDimOut (bool, optional): Use another entanglement step prior to the final output. DefaultDefaults to False.
            sameDimWarpEntangle (bool, optional): Make previously stated step the same one that generates the warp. Defaults to False.
        """
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

        # Entangle the final signals if requested
        self.finalEntangle = None
        if sameDimOut:
            if sameDimWarpEntangle:
                self.finalEntangle = self.warpEntangler
            else:
                self.finalEntangle = Entangle(inputSignals=internalDimensions, curveChannels=1, \
                    samples=samples, useKnowledgeMask=True, outputMode=EntangleOutputMode.COLLAPSE, \
                    dtype=dtype)

        # Amplify things in the things that are brought into clearer view
        self.compressorKnot = Knot(knotSize=internalDimensions, knotDepth=internalWaves, dtype=complexType)
        self.compressorGain = nn.Parameter(torch.ones(1, dtype=complexType))
    
    def forward(self, queries:torch.Tensor, states:torch.Tensor, inter:str='bicubic', padding:str='border', oneD:bool=True) -> torch.Tensor:
        """Run a forward computation through the module. Defaults to

        Args:
            queries (torch.Tensor): The queries for what to pay attention to (of size
                [BATCHES...,CURVES,SAMPLES]).
            states (torch.Tensor): The signal to look at (of size
                [BATCHES...,CURVES,SAMPLES]).
            inter (str, optional): The interpolation to use during the grid_sample() call. Defaults to 'bicubic'.
            padding (str, optional): The padding_mode to use during the grid_sample() call. Defaults to 'border'.
            oneD (bool, optional): Whether or not the knots evaluate the input signal as a multichannel curve. Defaults to True.

        Returns:
            torch.Tensor: An entangled and attended to attention vector. By default,
                this vector is returned in knot form (so [BATCHES...,CURVES,SAMPLES]), 
                but with prior configuration in the __init__() call, this can be changed to
                just be [BATCHES..., samples].
        """
        inputSize = queries.size()
        assert states.size() == inputSize
        
        # Shift both directions in computational elaboration
        integralStates = torch.fft.ifft(states, dim=-1)
        basisStates = torch.fft.fft(states, dim=-1)

        # Entangle the queries and the states together
        egoKnot = self.egoKnot.forward(queries, oneD=oneD)
        worldKnot = self.worldKnot.forward(states, oneD=oneD)
        integralKnot = self.integralKnot.forward(integralStates, oneD=oneD)
        basisKnot = self.basisKnot.forward(basisStates, oneD=oneD)
        parietalEntanglements, _ = self.parietalEntangler.forward(
            torch.stack([egoKnot, basisKnot, worldKnot, integralKnot], dim=1)
        )
        entangleSum = parietalEntanglements.sum(dim=1)
        superTangle, _ = self.warpEntangler.forward(
            entangleSum.unsqueeze(-2)
        )
        superSum = superTangle.sum(dim=1)

        
        # Pay attention using spatial warping and basis vector compression
        warpKnot = isoftmax(self.warpKnot.forward(superSum), dim=-2)
        compressorKnot = isoftmax(self.compressorKnot.forward(superSum), dim=-2) * self.compressorGain
        compressorKnot.squeeze_(dim=1)

        # Warping as if the state vector is 4D image data as seen here:
        # https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html#torch.nn.functional.grid_sample
        # Because the data is currently in [BATCH, Hin, C, Win], transpose is needed
        stateEntanglements = parietalEntanglements.transpose(1, 2)
        # Now in [BATCH, C, Hin, Win]

        # To warp the view of all levels of iterable signal complexity,
        # a tensor of size [BATCH, 1, samples, 2] is needed in the space of the
        # 'grid' param. This is done with the warp knot ([BATCH, curve, samples])
        # which needs to be in the shape [BATCH, 1, samples, curve].
        warpGrid = warpKnot.transpose(-1, -2)
        warpedStateReal = nnf.grid_sample(stateEntanglements.real, grid=warpGrid.real, \
            mode=inter, align_corners=False, padding_mode=padding).unsqueeze(-1)
        warpedStateImag = nnf.grid_sample(stateEntanglements.imag, grid=warpGrid.imag, \
            mode=inter, align_corners=False, padding_mode=padding).unsqueeze(-1)
        warpedState = torch.view_as_complex(torch.cat((warpedStateReal, warpedStateImag), dim=-1))

        # Now <warpedState> must be translated back to the format that the network
        # is expecting. The current format should be still be [BATCH, C, 1, samples],
        # but it needs to be [BATCH, C, samples], so just squeeze.
        warpedState.squeeze_(-2)

        # Find what makes the variably 'zoomed' signal, and modify it with the
        # compressor-like signal evaluated earlier. If things are done right here,
        # comrpessor should broadcast across the warped signal.
        warpedSignal = torch.fft.fft(warpedState, n=self.samples, dim=-1)
        warpCompSignal = warpedSignal * compressorKnot

        # Return the signal that is constructed from the final computations. This
        # signal is back into the constructed 'current' domain.
        result = torch.fft.ifft(warpCompSignal, n=self.samples, dim=-1)
        if self.finalEntangle is None:
            return result
        result, _ = self.finalEntangle.forward(result.unsqueeze(-2))
        return result.squeeze(-2).sum(-2)
