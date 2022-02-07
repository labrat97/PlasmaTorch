from .routing import *
from ..defaults import *
from ..activations import *
from ..conversions import *
from ..math import *
from ..losses import *

import torch as t
import torch.nn as nn


class ScaffoldFilter(KnowledgeFilter):
    def __init__(self, cid:str, ipns:bool, corrSamples:int=DEFAULT_FFT_SAMPLES, 
        inputSamples:int=DEFAULT_FFT_SAMPLES, outputSamples:int=DEFAULT_FFT_SAMPLES, 
        cdtype:t.dtype=DEFAULT_COMPLEX_DTYPE):

        # Build the superclass
        super(KnowledgeFilter, self).__init__(corrSamples=corrSamples, 
            inputSamples=inputSamples, outputSamples=outputSamples, cdtype=cdtype)
        
        # Iterate through the CID from IPNS in IPFS, turning it into a byte array
        #   then a nn.Parameter for the module (so that it may be serialized).
        preparam = bytearray.fromhex(cid)
        assert len(preparam) == int(32)
        self.cid:nn.Parameter = nn.Parameter(t.zeros((32), dtype=t.uint8, device='cpu'), requires_grad=False)
        for idx in range(32):
            self.cid[idx].add_(preparam[idx])
    
    def __hashStr(self):
        bytelist = self.cid.tolist()
        return bytearray(bytelist).hex()
