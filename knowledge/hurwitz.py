from ..defaults import *
from .zeta import *
from .routing import KnowledgeFilter
from ..math import nsoftmax
from ..conversions import nantonum


class HurwitzFilter(KnowledgeFilter):
    def __init__(self, corrSamples:int=DEFAULT_FFT_SAMPLES, ioSamples:int=DEFAULT_FFT_SAMPLES, cdtype:t.dtype=DEFAULT_COMPLEX_DTYPE):
        super(HurwitzFilter, self).__init__(corrSamples=corrSamples, inputSamples=ioSamples, outputSamples=ioSamples, cdtype=cdtype)

        # Store parameters to remap the input values to one another prior to the evaluation of the
        # hurwitz zeta function.
        self.remap:nn.Parameter = nn.Parameter(toComplex(t.eye(ioSamples, dtype=self.cdtype)))

    def __forward__(self, a:t.Tensor, b:t.Tensor) -> t.Tensor:
        # Find the basis vectors of the signal
        afft = tfft.fft(a, n=self.inputSamples, dim=-1)
        bfft = tfft.fft(b, n=self.inputSamples, dim=-1)

        # Remap the input vectors before the evaluation of the Hurwitz-Zeta function
        softmap = nsoftmax(self.remap, dims=[-1, -2])
        ar = afft @ softmap
        br = bfft @ softmap.transpose(-1, -2)

        # Run through the Hurwitz-Zeta function, get rid of invalid values
        hurAb = nantonum(hzetae(s=afft, a=br)).unsqueeze(-1)
        hurBa = nantonum(hzetae(s=bfft, a=ar)).unsqueeze(-1)

        # Create superposition style output through a tensor product
        return hurAb @ hurBa.transpose(-1, -2)
