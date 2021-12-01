from .filter import *
from ..math import *
from ..losses import *
from ..distributions import *
from ..defaults import *
from ..conversions import *
from ..activations import *

import torch as t
import torch.nn as nn
from torch.jit import script as ts

class KnowledgeRouter(KnowledgeFilter):
    def __init__(self, maxk:int=3, corrSize:int=DEFAULT_FFT_SAMPLES, cdtype:t.dtype=DEFAULT_COMPLEX_DTYPE):
        super(KnowledgeRouter, self).__init__(corrSize=corrSize, cdtype=cdtype)

        # Store all of the filters that the router can call to
        self.subfilters:nn.ModuleList = nn.ModuleList()
        self.maxk:int = 3

    def addFilter(self, x:KnowledgeFilter):
        assert x is KnowledgeFilter
        self.subfilters.append(x)

    def forward(self, a:t.Tensor, b:t.Tensor) -> t.Tensor:
        # Find the basis vectors of the input signals
        samples:int = max(a.size(-1), b.size(-1))
        afft = t.fft.fft(a, n=samples, dim=-1)
        bfft = t.fft.fft(b, n=samples, dim=-1)
        afftflat = t.flatten(afft, start_dim=0, end_dim=-2)
        bfftflat = t.flatten(bfft, start_dim=0, end_dim=-2)
        
        # Create the storage for the correlations
        dummy:t.Tensor = (afft.detach() * bfft.detach().conj())
        assert len(dummy.size()) >= 2
        dflat:t.Tensor = dummy.flatten(start_dim=0, end_dim=-2)
        icorrs:t.Tensor = t.zeros((len(self.subfilters), dummy[...,0,0].numel()), dtype=dummy.dtype)

        # Evaluate the correlation for all contained knowledge filters
        for idx, kfilter in enumerate(self.subfilters):
            icorrs[idx] = kfilter.forward(a=afftflat, b=bfftflat)

        # Grab the top correlation indices
        _, topcorrs = t.topk(icorrs.abs(), k=self.maxk, dim=0)
        topcorrs.transpose_(0, 1)

        # Run each signal through each set of knowledge filters and geometric mean together
        result:t.Tensor = t.zeros_like(dflat)
        for sdx in range(topcorrs.size(0)):
            for kdx in topcorrs[sdx]:
                kfilter = self.subfilters[kdx]
                result[sdx].add_(kfilter.forward(a=afftflat[sdx], b=bfftflat[sdx]))
            result[sdx].div_(len(topcorrs[sdx]))

        # Unflatten the resultant signals back to the relevant size
        return nn.Unflatten(0, dummy.size()[:-2])