from ..defaults import *
from .zeta import *
from .routing import KnowledgeFilter
from ..conversions import nantonum, toComplex


class LerchFilter(KnowledgeFilter):
    def __init__(self, corrSamples:int=DEFAULT_FFT_SAMPLES, ioSamples:int=DEFAULT_FFT_SAMPLES, cdtype:t.dtype=DEFAULT_COMPLEX_DTYPE):
        super(LerchFilter, self).__init__(corrSamples=corrSamples, inputSamples=ioSamples, outputSamples=ioSamples, cdtype=cdtype)

        # Store the internal lambda values and plug into the lambda section of the Lerch-Zeta function
        self.lambdas:nn.Parameter = nn.Parameter(toComplex(t.randn((2, ioSamples), dtype=self.cdtype)))

    def __forward__(self, a:t.Tensor, b:t.Tensor) -> t.Tensor:
        # Evaluate the Lerch-Zeta function at all of the values provided
        lab = nantonum(lerche(lam=self.lambdas[0], s=a, a=b))
        lba = nantonum(lerche(lam=self.lambdas[1], s=b, a=a))

        # Create a tensor product out of the resultant vectors and return the
        # superposition of the signals.
        return lab.unsqueeze(-1) @ lba.unsqueeze(-1).transpose(-1, -2)
