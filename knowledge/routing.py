from ..defaults import *
from ..activations import *
from ..conversions import *
from ..math import *
from ..losses import *
from ..distributions import *

import torch as t
import torch.nn as nn
import torch.fft as tfft
from abc import ABC, abstractmethod


class KnowledgeFilter(nn.Module, ABC):
    """
    An abstract class used for creating encapsulated bits of knowledge to be called by
    other KnowledgeFilters or structures looking to call knowledge from plasmatorch.
    """
    @abstractmethod
    def __init__(self, corrSamples:int=DEFAULT_FFT_SAMPLES, cdtype:t.dtype=DEFAULT_COMPLEX_DTYPE):
        """The abstract constructor for a knowledge filter.

        Args:
            corrSamples (int, optional): The amount of samples to describe each curve. Defaults to DEFAULT_FFT_SAMPLES.
            cdtype (t.dtype, optional): The default datatype for the complex correlation parameter. Defaults to DEFAULT_COMPLEX_DTYPE.
        """
        super(KnowledgeFilter, self).__init__()
        self.corrToken:nn.Parameter = nn.Parameter(toComplex(t.zeros((2, corrSamples), dtype=cdtype)), requires_grad=True)
        self.routers:nn.ModuleList = nn.ModuleList()
        self.corrSamples:int = corrSamples
        self.cdtype:t.dtype = cdtype

    def implicitCorrelation(self, a:t.Tensor, b:t.Tensor, isbasis:bool=False) -> t.Tensor:
        """Calculate the stored correlation of the input signal with the tokenized
        basis vectors. This is used to predict what could be inside of the function before
        evaluating said function.

        Args:
            a (t.Tensor): A basis vector (optionally a signal) used for calculation.
            b (t.Tensor): Another basis vector (optionally a signal) used for calculation.
            isbasis (bool, optional): If False, the vectors coming in are preFFT'd. Defaults to False.

        Returns:
            t.Tensor: The average correlation accross the samples, curves, and vectors.
        """
        # Put the self correlation into an easy to process bounds
        selfcorr = isigmoid(self.corrToken)

        # Find the respective correlation from the token with the input signals
        acorr:t.Tensor = correlation(x=a, y=selfcorr[0], dim=-1, isbasis=isbasis).mean(dim=-1).mean(dim=-1)
        bcorr:t.Tensor = correlation(x=b, y=selfcorr[1], dim=-1, isbasis=isbasis).mean(dim=-1).mean(dim=-1)

        # Find the mean of the mean correlations
        return (acorr + bcorr) / 2.

    @abstractmethod
    def forward(self, a:t.Tensor, b:t.Tensor) -> t.Tensor:
        """Runs two tensors through comparative knowledge.

        Args:
            a (t.Tensor): The first set of basis vectors defining an interacting signal.
            b (t.Tensor): The second set of basis vectors defining another interacting signal.

        Returns:
            t.Tensor: The comparative knowledge graph output signal.
        """
        pass


class KnowledgeRouter(KnowledgeFilter):
    """
    A KnowledgeFilter type class that is used to call other knowledge filter type classes.
    Due to the way that the signal traversal works, this should be a borderline completely unified
    tree traversal method due to the continuous nature. Every single layer of traversal is evaluated
    in parallel, and ever computation is chronologically independent. Every depth will also do a layered
    set of amplitudes from the previous signal, making the potential for things like the harmonic series
    just fall out.
    """
    def __init__(self, maxk:int=3, corrSamples:int=DEFAULT_FFT_SAMPLES, cdtype:t.dtype=DEFAULT_COMPLEX_DTYPE):
        """The constructor for a KnowledgeRouter, defining performance parameters 

        Args:
            maxk (int, optional): The starting amount of maximum signals to evaluate. Defaults to 3.
            corrSamples (int, optional): The amount of samples to describe each curve. Defaults to DEFAULT_FFT_SAMPLES.
            cdtype (t.dtype, optional): The default datatype for the complex correlation parameter. Defaults to DEFAULT_COMPLEX_DTYPE.
        """
        super(KnowledgeRouter, self).__init__(corrSamples=corrSamples, cdtype=cdtype)

        # Store all of the filters that the router can call to
        self.correlationMask:nn.Parameter = nn.Parameter(toComplex(torch.zeros((corrSamples, corrSamples), dtype=cdtype)))
        self.correlationPolarization:nn.Parameter = nn.Parameter(t.zeros((1), dtype=self.correlationMask.real.dtype))
        self.subfilters:nn.ModuleList = nn.ModuleList()
        self.maxk:int = maxk

    def addFilter(self, x:KnowledgeFilter):
        """Adds a knowledge filter to the router, circularly linking the router in the filter.

        Args:
            x (KnowledgeFilter): The filter to add to the router.
        """
        assert isinstance(x, KnowledgeFilter)
        self.subfilters.append(x)
        x.routers.append(self)

    def forward(self, a:t.Tensor, b:t.Tensor) -> t.Tensor:
        # Find the basis vectors of the input signals
        samples:int = self.correlationMask.size(-1)
        afft:t.Tensor = tfft.fft(a, n=samples, dim=-1)
        bfft:t.Tensor = tfft.fft(b, n=samples, dim=-1)

        # Entangle the signals with the `nsoftmax` function and a matmul, then sum out orthogonally
        superposition:t.Tensor = (afft.unsqueeze(-1) @ bfft.unsqueeze(-1).transpose(-1,-2)) * nsoftmax(self.correlationMask, dims=[-1,-2])
        afft = superposition.sum(dim=-1)
        bfft = superposition.sum(dim=-2)

        # Flatten the batches to be able to easily index the contained filters
        # Need to flatten at -3 due to inclusive `end_dim` argument
        afftflat = t.flatten(afft, start_dim=0, end_dim=-3)
        aflat = t.flatten(a, start_dim=0, end_dim=-3)
        bfftflat = t.flatten(bfft, start_dim=0, end_dim=-3)
        bflat = t.flatten(b, start_dim=0, end_dim=-3)
        
        # Create the storage for the correlations
        icorrs:t.Tensor = t.zeros((len(self.subfilters), aflat[...,0].numel()), dtype=aflat.dtype)

        # Evaluate the correlation for all contained knowledge filters
        for idx, kfilter in enumerate(self.subfilters):
            icorrs[idx] = kfilter.implicitCorrelation(a=afftflat, b=bfftflat, isbasis=True)

        # Grab the top correlation indices, choosing the modules to be evaluated
        _, topcorrs = t.topk(icorrs.abs(), k=self.maxk, dim=0, largest=True)
        topcorrs.transpose_(0, 1)

        # Run each signal through each set of knowledge filters and geometric mean together
        result:t.Tensor = t.zeros_like(aflat)
        for sdx in range(topcorrs.size(0)):
            for kdx in topcorrs[sdx]:
                # Pull the filter
                kfilter = self.subfilters[kdx]

                # Add the resultant signal to the current filter (router) result signal
                result[sdx].add_(kfilter.forward(a=aflat[sdx], b=bflat[sdx]))
            # Divide by the number of indices used to collect all of the signals
            result[sdx].div_(topcorrs[sdx].numel())

        # Unflatten the resultant signals back to the relevant size
        return nn.Unflatten(0, aflat.size()[:-2])(result)
