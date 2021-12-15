from .routing import *
from ..defaults import *
from ..activations import *
from ..conversions import *
from ..math import *
from ..losses import *

import torch as t
import torch.nn as nn


class SuperpositionFilter(KnowledgeFilter):
    def __init__(self, reprojSamples:int=DEFAULT_FFT_SAMPLES, corrSamples:int=DEFAULT_FFT_SAMPLES, cdtype:t.dtype=DEFAULT_COMPLEX_DTYPE):
        super(KnowledgeFilter, self).__init__(corrSamples=corrSamples, cdtype=cdtype)  

