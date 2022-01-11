from .zeta import *
from .routing import *
from ..conversions import *

import torch as t
import torch.nn as nn
import torch.fft as tfft

class LerchFilter(KnowledgeFilter):
    def __init__(self, corrSamples:int=DEFAULT_FFT_SAMPLES, cdtype:t.dtype=DEFAULT_COMPLEX_DTYPE):
        super(LerchFilter, self).__init__(corrSamples=corrSamples, cdtype=cdtype)

        # Store the internal lambda values and plug into the lambda section of the Lerch-Zeta function
        self.lambdas:nn.Parameter = nn.Parameter(toComplex(t.randn((2, self.corrSamples), dtype=cdtype)))

    def forward(self, a:t.Tensor, b:t.Tensor) -> t.Tensor:
        # Do not get basis vectors, but make sure that the signals coming in are of the
        # correct sample count
        if a.size(-1) == self.corrSamples:
            wa:t.Tensor = a
        else:
            wa:t.Tensor = resampleSmear(a, samples=self.corrSamples, dim=-1)
        if b.size(-1) == self.corrSamples:
            wb:t.Tensor = b
        else:
            wb:t.Tensor = resampleSmear(b, samples=self.corrSamples, dim=-1)

        # Evaluate the Lerch-Zeta function at all of the values provided
        lab = nantonum(lerche(lam=self.lambdas[0], s=wa, a=wb))
        lba = nantonum(lerche(lam=self.lambdas[1], s=wb, a=wa))

        # Create a tensor product out of the resultant vectors and return the
        # superposition of the signals.
        return lab.unsqueeze(-1) @ lba.unsqueeze(-1).transpose(-1, -2)
