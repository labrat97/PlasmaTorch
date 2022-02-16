from .routing import *
from ..defaults import *
from ..activations import *
from ..conversions import *
from ..math import *
from ..losses import *

import torch as t
import torch.nn as nn

import cid as CID
import subprocess as proc

import os
from os import path


class ScaffoldFilter(KnowledgeFilter):
    def __init__(self, multihash:str, ipns:bool, corrSamples:int=DEFAULT_FFT_SAMPLES, 
        inputSamples:int=DEFAULT_FFT_SAMPLES, outputSamples:int=DEFAULT_FFT_SAMPLES, 
        cdtype:t.dtype=DEFAULT_COMPLEX_DTYPE, ipfsMount:str=path.sep+'ipfs', 
        ipnsMount:str=path.sep+'ipns', fastdev:str=None,
        weightName:str='weight', biasName:str='bias', invbiasName:str='invbias'):

        # Build the superclass
        super(KnowledgeFilter, self).__init__(corrSamples=corrSamples, 
            inputSamples=inputSamples, outputSamples=outputSamples, cdtype=cdtype)
        
        # Iterate through the CID from IPFS, turning it into a byte array
        #   then a nn.Parameter for the module (so that it may be serialized).
        assert CID.is_cid(multihash)
        mcid = CID.from_string(multihash)
        mhashBytes = mcid.multihash
        self.multihash:nn.Parameter = nn.Parameter(t.zeros((len(mhashBytes)), dtype=t.uint8, device='cpu'), requires_grad=False)
        for idx in range(len(mhashBytes)):
            self.multihash[idx].add_(mhashBytes[idx])
        
        # Read the requested file from the ipfs file system
        if ipns:
            self.mount:nn.Parameter = nn.Parameter(t.Tensor(path.join(ipnsMount, multihash).encode()).type(t.uint8))
        else:
            self.mount:nn.Parameter = nn.Parameter(t.Tensor(path.join(ipfsMount, multihash).encode()).type(t.uint8))

        # Open the file into the CPU so that it may be potentially swapped
        with open(bytes(self.mount).decode(), 'br') as heartfile:
            self.heart:nn.Module = t.jit.load(heartfile, map_location='cpu')
        
        # Set the device to be used later for fast computation
        if fastdev is None:
            if t.cuda.is_available():
                fastdev = 'cuda'
            else:
                fastdev = 'cpu'
        self.fastdev:nn.Parameter = nn.Parameter(t.Tensor(fastdev.encode()).type(t.uint8))

        # Set up the configuration flags for the model inference
        self.freezeHeart:bool = not ipns
        self.lstmType:bool = isinstance(self.heart, nn.LSTM)
        self.linearType:bool = isinstance(self.heart, nn.Linear)
        self.weightName:str = weightName
        self.biasName:str = biasName
        self.invbiasName:str = invbiasName
        self.invertedLinear:bool = self.linearType

        # If this is a linear module, check for an inverted bias
        if self.linearType:
            self.invertedLinear = False
            for name, _ in self.named_parameters(recurse=True):
                if name == self.invbiasName:
                    self.invertedLinear = True
                    break
        
    
    def cid(self):
        # Encode the contained parameter into something understandable by IPFS
        bytelist = self.cid.tolist()
        return CID.from_bytes(bytearray(bytelist))
    
    def multihash(self):
        # Get the raw bytes of the encoding of the cid
        return self.cid().multihash

    def __forward__(self, a:t.Tensor, b:t.Tensor) -> t.Tensor:
        #
