from .routing import *
from ..defaults import *
from ..activations import *
from ..conversions import *
from ..math import *
from ..losses import *

import torch as t
import torch.nn as nn


class PipelineFilter(KnowledgeFilter):
    def __init__(self, corrSamples:int=DEFAULT_FFT_SAMPLES, cdtype:t.dtype=DEFAULT_COMPLEX_DTYPE):
        super (KnowledgeFilter, self).__init__(corrSamples=corrSamples, cdtype=cdtype)

        # To handle importing sets of Scaffold Filters, a scalar divisor of Phi is recommended
        # for each layer of filter. The idea behind this is the preservation of presented data.
        # If the prior layer's basis frequencies are regressed as not being resonant with later
        # filter frequencies, all of the data should come out and be recoverable by the main agent.
        
