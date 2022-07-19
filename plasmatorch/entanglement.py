from .defaults import *
from .distributions import *
from .activations import *
from .conversions import *
from .math import *
from .losses import *

from enum import Flag



class EntangleOutputMode(int, Flag):
    """
    The output mode used in the Entangle() function.
    """

    # Output the superpositions between the signals (with knowledge graphs)
    SUPERPOSITION:int = 0b1 << 0

    # Output the collapsed, fully elaborated, signals at the end of the function
    COLLAPSE:int = 0b1 << 1

    # Output both of the contained modes in superposition collapse order respectively.
    BOTH:int = SUPERPOSITION | COLLAPSE



class Entangle(nn.Module):
    """
    Entangles n signals together to form a higher complexity signal.
    """
    def __init__(self, inputSignals:int, curveChannels:int = DEFAULT_SPACE_PRIME, \
        samples:int = DEFAULT_FFT_SAMPLES, useKnowledgeMask:bool = True, \
        outputMode:EntangleOutputMode = EntangleOutputMode.BOTH, dtype:t.dtype = DEFAULT_DTYPE):
        """Create a new Entangle object, specifying functionality before runtime.

        Args:
                inputSignals (int): The amount of signals to entangle together.
                curveChannels (int, optional): The amount of dimensions in the curve/knot. Defaults to DEFAULT_SPACE_PRIME.
                useKnowledgeMask (bool, optional): Use a knowledge mask on a superposition of the signals. Defaults to True.
                dtype (t.dtype, optional): Specify the data type of the module. Defaults to DEFAULT_DTYPE.
        """
        super(Entangle, self).__init__()

        # Store data about the signals going into/out of the module
        self.signalCount:int = inputSignals
        self.curveChannels:int = curveChannels
        self.samples:int = samples
        self.outputMode:EntangleOutputMode = outputMode

        # Hold the entanglement parameters
        self.entangleActivation:nn.ModuleList = nn.ModuleList([LinearGauss(1, dtype=dtype) for _ in range(inputSignals)])
        self.entanglePolarization:nn.Parameter = nn.Parameter(toComplex(t.zeros(
            (inputSignals), dtype=dtype
        )))

        # If requested, use a knowledge mask at the end of the forward() call
        self.knowledgeMask:nn.Parameter = None
        if useKnowledgeMask:
            # This should broadcast an identity matrix over the knowledge mask for collapsing
            iEye:t.Tensor = toComplex(t.eye(samples, dtype=dtype, requires_grad=False))
            self.knowledgeMask = nn.Parameter(
                toComplex(t.zeros((inputSignals, curveChannels, samples, samples), dtype=dtype)) \
                + iEye)
    
    def forward(self, x:t.Tensor) -> Tuple[t.Tensor]:
        """Computes the forward pass of the module.

        Args:
                x (t.Tensor): A tensor of size [..., SIGNALS, CURVES, SAMPLES] that
                    represents the continuous signals specified in the __init__() function.

        Returns:
                Tuple[t.Tensor]: A tensor of size [..., SIGNALS, CURVES, SAMPLES] for the first term
                    that has been entangled, and a tensor of size [..., SIGNALS, CURVES, SAMPLES, SAMPLES] for
                    the second term. The terms represent the collapsed signal and the superpositions respectively.
        """

        # Define some constants
        SAMPLE_POS:int = -1
        CURVE_POS:int = -2
        COUNT_POS:int = -3
        MASK_POS:List[int] = [-1, -2]

        # Check to make sure that x is of compatible shape
        inputSize:t.Size = x.size()
        inputSizeLen:int = len(inputSize)
        if inputSizeLen == 3: 
            x = x.unsqueeze(0)
        assert inputSizeLen == 4
        assert inputSize[SAMPLE_POS] == self.samples
        assert inputSize[CURVE_POS] == self.curveChannels
        assert inputSize[COUNT_POS] == self.signalCount
        isComplex:bool = t.is_complex(x)

        # Find out what the signals are made of
        signals:t.Tensor = fft(x, n=self.samples, dim=SAMPLE_POS)

        # Store where the signals are going
        y:t.Tensor = t.zeros_like(x)
        s:t.Tensor = t.zeros((inputSize[0], self.signalCount, self.curveChannels, self.samples, self.samples), \
            dtype=self.knowledgeMask.dtype)
        for idx in range(self.signalCount):
            signal = signals[:,idx]
            polarization:t.Tensor = self.entanglePolarization[idx]

            for jdx in range(self.signalCount):
                # See how similar each signal is
                subsig = signals[:,jdx]
                corr:t.Tensor = correlation(x=signal, y=subsig, dim=SAMPLE_POS, isbasis=True).mean(dim=SAMPLE_POS)

                # Create a superposition through a tensor product
                superpos:t.Tensor = superposition(a=signal, b=subsig)

                # Apply knowledge to the superposition of the subsignals if requested
                if self.knowledgeMask is not None:
                    superpos = superpos * nsoftunit(self.knowledgeMask[jdx], dims=MASK_POS)

                # Save superposition for output if needed
                if (int(self.outputMode) & int(EntangleOutputMode.SUPERPOSITION)) != 0:
                    s[:,idx].add_(ifft(superpos, n=2*[self.samples], dim=MASK_POS))

                # No need to collapse
                if (int(self.outputMode) & int(EntangleOutputMode.COLLAPSE)) == 0:
                    continue

                # Act on correlation for collapse
                entangleMix:t.Tensor = self.entangleActivation[idx].forward(corr).unsqueeze(-1)
                classicalMix:t.Tensor = 1 - entangleMix

                # Collapse
                collapseSignal:t.Tensor = collapse(superpos, polarization.abs())
                collapseSmear:t.Tensor = ifft(collapseSignal, n=self.samples, dim=SAMPLE_POS)
                if not isComplex:
                    collapseSmear:t.Tensor = realfold(collapseSmear, phase=polarization.angle())

                # Put into output for signals
                y[:,idx] = y[:,idx] + ((entangleMix * collapseSmear) + (classicalMix * x[:,idx]))
        
        # Regularize
        if (int(self.outputMode) & int(EntangleOutputMode.COLLAPSE)) != 0:
            y.div_(self.signalCount)
        if (int(self.outputMode) & int(EntangleOutputMode.SUPERPOSITION)) != 0:
            s.div_(self.signalCount)

        # Return
        if self.outputMode == EntangleOutputMode.COLLAPSE:
            return y, None
        if self.outputMode == EntangleOutputMode.SUPERPOSITION:
            return None, s
        return y, s



@ts
def collapse(x:t.Tensor, polarization:t.Tensor) -> t.Tensor:
    """Collapse a signal vector according to how it ROUGHLY works in reality. A three phase signal
    is constructed and summed together at the end of two transposed means and an eigenvalue computation.
    This should fully transfer the information of the system to the collapsed signal assuming infinitely
    sampled polarizations.

    Args:
        x (t.Tensor): The signal to collapse, sized (*, n, n).
        polarization (t.Tensor): The polarization of the collapse for the signal.

    Returns:
        t.Tensor: The collapsed signal of size (*, n, n) (assuming pointwise or less polarization).
    """
    # Brief argument check
    assert x.size(-1) == x.size(-2)
    
    # Get the means of the matrix on each view according to transposition on the final
    #   dimensions (*..., n, n)
    suma = t.mean(x, dim=-1)
    sumb = t.mean(x, dim=-2)

    # Gets the eigenvalues of the matrix for the third and final phase of the collapse
    eigv = t.linalg.eigvals(x)

    # Perform a three phase collapse
    iter = 2. * pi() / 3.

    # The eigvals will be the one vals as they represent the solved roots of the
    #   input matrix
    rote = eigv * isin(polarization)
    rota = suma * isin(polarization + iter)
    rotb = sumb * isin(polarization - iter)

    # Combine the three phases and return
    return rote + rota + rotb


@ts
def superposition(a:t.Tensor, b:t.Tensor) -> t.Tensor:
    """Create a superposition of the two input signals with a complex softunit on the result from the matmul.

    Args:
        a (t.Tensor): The first tensor to create a superposition with.
        b (t.Tensor): The second tensor to create a superposition with.

    Returns:
        t.Tensor: The superpositioned tensor.
    """
    # Do the matrix multiplication required 
    rawSuper:t.Tensor = a.unsqueeze(-1) @ b.unsqueeze(-2)
    return nsoftunit(x=rawSuper, dims=[-1,-2])


@ts
def entangle(a:t.Tensor, b:t.Tensor, mask:t.Tensor, polarization:t.Tensor) -> t.Tensor:
    """Create a superposition of signals `a` and `b` with a knowledge mask and polarization
    to reduce the output signals to the same dimensionality as the input signals.

    Args:
        a (t.Tensor): The first tensor of size (*, n) to create an entanglement with.
        b (t.Tensor): The second tensor of size (*, m) to create an entanglement with.
        mask (t.Tensor): The mask of size (*, n, m) to apply knowledge through.
        polarization (t.Tensor): The polarization of the signals.

    Returns:
        t.Tensor: The entangled signals.
    """
    # Assert the arguments are size compatible with each other
    assert len(mask.size()) >= 2
    assert mask.size(-2) == a.size(-1)
    assert mask.size(-1) == b.size(-1)

    # Create superposition and cover in a mask
    super:t.Tensor = superposition(a, b)
    entSuper:t.Tensor = super * nsoftunit(mask, dims=[-1, -2])

    # Collapse the masked superposition using polarization in the collapse() method
    return collapse(entSuper, polarization)
