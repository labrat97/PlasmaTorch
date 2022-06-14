from .defaults import *
from .conversions import *
from .math import *


@ts
def lissajous(x:t.Tensor, freqs:t.Tensor, phases:t.Tensor, oneD:bool = True) -> t.Tensor:
    """Create a lissajous curve sampled at position `x` with the associated frequencies
    and phases.

    Args:
            x (t.Tensor): The sampling positions for the curve(s).
            freqs (t.Tensor): The frequencies for the curve(s).
            phases (t.Tensor): The phase offsets for the curve(s).
            oneD (bool, optional): All sampling positions are considered independent for the frequencies
            and phases provided. Defaults to True.

    Returns:
            t.Tensor: The sampled lissajous curve.
    """
    assert freqs.size() == phases.size()
    
    if oneD:
        # Manipulate dimensions to broadcast in 1D sense
        x = t.unsqueeze(x, -1)
        sinpos:t.Tensor = (x @ freqs) + (t.ones_like(x) @ phases)
    else:
        # Put curves in the right spot
        assert x.size()[-2] == freqs.size()[-1]
        x = x.transpose(-1,-2)

        # Maniupulate dimensions to broadcast in per-curve sense
        sinpos:t.Tensor = (x * freqs.unsqueeze(0)) + (t.ones_like(x) * phases.unsqueeze(0))

    # Activate in curve's embedding space depending on the working datatype.
    # This is done due to the non-converging nature of the non-convergence of the
    # cos function during the operation on complex numbers. To solve this, a sin function
    # is called in the imaginary place to emulate the e^ix behavior for sinusoidal signals.
    return isin(sinpos).transpose(-1, -2)

class Lissajous(nn.Module):
    """
    Holds a Lissajous-like curve to be used as a sort of activation layer as a unit
        of knowledge.
    """
    def __init__(self, size:int, dtype:t.dtype = DEFAULT_DTYPE):
        """Builds a new Lissajous-like curve structure.

        Args:
                size (int): The amount of dimensions encoded in the curve.
        """
        super(Lissajous, self).__init__()

        self.size:int = size
        self.frequency:nn.Parameter = nn.Parameter(t.zeros([1, size], dtype=dtype))
        self.phase:nn.Parameter = nn.Parameter(t.zeros([1, size], dtype=dtype))

    def forward(self, x:t.Tensor, oneD:bool = True) -> t.Tensor:
        """Gets a sample or batch of samples from the contained curve.

        Args:
                x (t.Tensor): The sample or sampling locations. If dim[-2] == self.size,
                    the input curve is believed to have the same amount of curves as the function.
                    When this is the case, instead of taking a 1D input.
                oneD (bool): If true, expand every leaf logit into the required amount of
                    internal signals.

        Returns:
                t.Tensor: The evaluted samples.

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
    def __init__(self, knotSize:int, knotDepth:int, dtype:t.dtype=DEFAULT_DTYPE):
        """Constructs a Knot for later use generating all weights and storing internally.

        Args:
                knotSize (int): The dimensionality of the contained lissajous-like curves.
                knotDepth (int): The amount of lissajous-like curves to be added together.
                dtype (t.dtype): The type of the housed parameters used for modifying
                    the value of the contained lissajous structures.
        """
        super(Knot, self).__init__()

        # Set up the curves for the function
        self.knotDepth = knotDepth
        self.knotSize = knotSize

        # Add some linearly trained weighted goodness
        self.dtype:t.dtype = dtype
        paramSize:List[int] = [self.knotDepth, self.knotSize, 1]
        self.regWeights:nn.Parameter = nn.Parameter(t.ones(paramSize, dtype=dtype) / self.knotDepth)
        
        self.frequencies:nn.Parameter = nn.Parameter(t.zeros((self.knotSize, self.knotDepth), dtype=dtype))
        self.phases:nn.Parameter = nn.Parameter(t.zeros((self.knotSize, self.knotDepth), dtype=dtype))
        self.__triu:t.Tensor = t.triu(t.ones((self.knotDepth, self.knotDepth), dtype=dtype), diagonal=0).detach()
        self.__latticeParams:t.Tensor = latticeParams(self.knotDepth)

        self.knotRadii:nn.Parameter = nn.Parameter(t.zeros(paramSize[1:], dtype=dtype))

    def forward(self, x:t.Tensor, oneD:bool = True) -> t.Tensor:
        """Pushed forward the same way as the Lissajous module. This is just an array
        of Lissajous modules summed together in a weighted way.

        Args:
                x (t.Tensor): The points to sample on the curves.
                oneD (bool): Evaluate the tensor as if it is one dimensional (curves from 1 curve). Defaults to True.

        Returns:
                t.Tensor: The original size tensor, but every point has a Lissajous curve
                    activated upon it. There will be one extra dimension that is the same in size
                    as the dimensions of the curve.

                    [Batches,::,Samples] -> [Batches,::,Curves,Samples]
        """
        # Create the expanded dimensions required in the output tensor
        if oneD:
            outputSize:t.Size = t.Size(list(x.size()) + [self.knotSize])
            result:t.Tensor = t.zeros(outputSize, dtype=self.dtype).transpose(-1, -2)
        else:
            outputSize:t.Size = x.size()
            result:t.Tensor = t.zeros(outputSize, dtype=self.dtype)
        
        # Add the frequencies together
        freqs:t.Tensor = ((self.frequencies * self.__latticeParams) @ self.__triu).transpose(0, 1)
        phases:t.Tensor = ((self.phases * self.__latticeParams) @ self.__triu).transpose(0, 1)

        # Add all of the curves together
        for idx in range(self.knotDepth):
            # Pass the frequencies to the curves
            freqn:t.Tensor = freqs[idx].unsqueeze(0)
            phasen:t.Tensor = phases[idx].unsqueeze(0)
            regn:t.Tensor = self.regWeights[idx]

            # Each lissajous curve-like structure has different weights, and therefore 
            curve:t.Tensor = regn * lissajous(x=x, freqs=freqn, phases=phasen, oneD=oneD)
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
    def __init__(self, forks:int=DEFAULT_FFT_SAMPLES, dtype:t.dtype=DEFAULT_COMPLEX_DTYPE):
        """Initialize the ringing module.

        Args:
                forks (int, optional): The amount of forks to ring in the module. Defaults to DEFAULT_FFT_SAMPLES.
                dtype (t.dtype, optional): The default datatype for ringing parameters; supports complex values. Defaults to DEFAULT_COMPLEX_DTYPE.
        """
        super(Ringing, self).__init__()

        # The positions and values of the enclosed forks
        forks = int(forks)
        DECAY_SEED = (asigphi()).type(dtype) # After a sigmoid eval this should come to 1/phi()
        self.forkPos = nn.Parameter(toComplex(t.zeros((forks), dtype=dtype)).real)
        self.forkVals = toComplex(t.zeros((forks), dtype=dtype, requires_grad=False))
        self.forkDecay = nn.Parameter(t.ones((forks), dtype=dtype) * DECAY_SEED)
        self.signalDecay = nn.Parameter(t.ones((1), dtype=dtype) * DECAY_SEED)

    def __createOutputSignal__(self, forks:t.Tensor, xfft:t.Tensor, posLow:t.Tensor, posHigh:t.Tensor, posMix:t.Tensor) -> t.Tensor:
        # Create tensor for constructing output
        yfft = t.zeros_like(xfft)

        # Apply fork signals to appropriate locations
        yfft[..., posLow] += ((1 - posMix) * forks)
        yfft[..., posHigh] += (posMix * forks)
        yfft.add_(xfft * isigmoid(self.signalDecay))

        return yfft
    
    def dampen(self, stop:bool=False):
        """Decay the ringing by one internal decay step. Optionally stop the ringing.

        Args:
                stop (bool, optional): If True, stop the previous ringing entirely. Defaults to False.
        """
        # If stopping, fully decaying
        if stop:
            self.forkVals = self.forkVals * 0
        # Regular 1/phi() decay
        else:
            self.forkVals = self.forkVals * isigmoid(self.forkDecay)
        

    def view(self, samples:int=DEFAULT_FFT_SAMPLES) -> t.Tensor:
        """Get the current ringing signal.

        Args:
                samples (int, optional): The amount of samples to construct the signal with. Defaults to DEFAULT_FFT_SAMPLES.

        Returns:
                t.Tensor: The ringing signal.
        """
        # Generate metadata needed to create the output signal
        assert samples >= 1
        positions = isigmoid(self.forkPos) * (samples - 1)
        posLow = positions.type(t.int64)
        posHigh = (posLow + 1).clamp_max(samples - 1)
        posMix = positions - posLow
        xfft = t.zeros((samples), dtype=self.forkVals.dtype)

        # Generate the output signal
        yfft = self.__createOutputSignal__(forks=self.forkVals, xfft=xfft, posLow=posLow, posHigh=posHigh, posMix=posMix)

        # Generate the output signal in the time domain according to the sample size
        return ifft(yfft, n=samples, dim=-1)

    def forward(self, x:t.Tensor, stopTime:bool=False, regBatchInput:bool=True) -> t.Tensor:
        """The default forward call for the ringing module.

        Args:
                x (t.Tensor): The signal deconstruct and add to the bank of forks.
                stopTime (bool, optional): If False, store the signal decay. Defaults to False.
                regBatchInput (bool, optional): If True, use a mean as the way to input the signal, False sums. Defaults to True.

        Returns:
                t.Tensor: The ringing signal from the input signal using the banked forks.
        """
        # Gather parameters needed to have some light attention to the tunes coming in
        xfft = fft(x, dim=-1)
        xsamples = x.size()[-1]
        positions = isigmoid(self.forkPos) * (xsamples - 1)

        # Extract the target parameters from the signal. In doing this, signal decay is avoided
        #   only when applying to the forks. In all other parts of this function (parts not contributing
        #   to the xvals->forkvals relationship), decay should be applied and represented/stored.
        posLow = positions.type(t.int64)
        posHigh = (posLow + 1).clamp_max(xsamples - 1)
        posMix = positions - posLow # [1, 0] -> [HIGH, 1-LOW]
        xvals = ((1 - posMix) * xfft[..., posLow]) + (posMix * xfft[..., posHigh])

        # To account for the collapse of the potentially massive amount of signals 
        #   coming in, the paramter regBatchInput (from the definition of the
        #   method) if True averages out all of the signals per sample. Otherwise, all of the signals
        #   added to the output with regularization left up to later implementation.
        if len(xvals.size()) > 1:
            # Turn the samples to the most significant dimension
            xvals.transpose_(-1, 0)

            # Both of the batch handling functions shift left
            if regBatchInput:
                for _ in range(len(xvals.size()) - 1):
                    xvals = t.mean(xvals, dim=1)
            else:
                for _ in range(len(xvals.size()) - 1):
                    xvals = t.sum(xvals, dim=1)

        # Add the input signals to the enclosed signals, remember, xvals doesn't decay
        #   here, the recurrent fork values do.
        forkVals = ((self.forkVals * isigmoid(self.forkDecay)) + xvals)
        if not stopTime:
            self.forkVals = forkVals
        
        # Create the output signal
        yfft = self.__createOutputSignal__(forks=forkVals, xfft=xfft, posLow=posLow, posHigh=posHigh, posMix=posMix)

        # Return constructed signal
        return ifft(yfft, n=xsamples, dim=-1)
