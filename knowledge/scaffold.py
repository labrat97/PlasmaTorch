from .routing import *
from ..defaults import *
from ..activations import *
from ..conversions import *
from ..math import *
from ..losses import *

import torch as t
import torch.nn as nn


class ScaffoldFilter(KnowledgeFilter):
    def __init__(self, parentModules:nn.ModuleList, corrSamples:int=DEFAULT_FFT_SAMPLES, cdtype:t.dtype=DEFAULT_COMPLEX_DTYPE):
        super(KnowledgeFilter, self).__init__(corrSamples=corrSamples, cdtype=cdtype)
        
        # The current plan for this filter is to freeze all of the weights coming in,
        # then feed the `a` and `b` vectors in the forward function to the QKV tensors
        # on the layers of attention being imported.

