from ..defaults import *
from ..activations import *
from ..conversions import *
from ..math import *
from ..losses import *

import torch as t
import torch.nn as nn
from torch.jit import script as ts
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
    
    def implicitCorrelation(self, a:t.Tensor, b:t.Tensor, isbasis:bool=False) -> t.Tensor:
        """Calculate the stored correlation of the input signal with the tokenized basis
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
