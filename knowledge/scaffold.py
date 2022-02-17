from .routing import *
from ..defaults import *
from ..activations import *
from ..conversions import *
from ..math import *
from ..losses import *

from .ipfs import IPFile

import torch as t
import torch.nn as nn

import cid as CID

from os import path


class ScaffoldFilter(KnowledgeFilter):
    def __init__(self, multihash:str, ipns:bool, corrSamples:int=DEFAULT_FFT_SAMPLES, 
        inputSamples:int=DEFAULT_FFT_SAMPLES, outputSamples:int=DEFAULT_FFT_SAMPLES, 
        cdtype:t.dtype=DEFAULT_COMPLEX_DTYPE, ipfsCommand:str='ipfs', fastdev:str=None,
        weightName:str='weight', biasName:str='bias', invbiasName:str='invbias'):

        # Build the superclass
        super(KnowledgeFilter, self).__init__(corrSamples=corrSamples, 
            inputSamples=inputSamples, outputSamples=outputSamples, cdtype=cdtype)
        
        # Iterate through the CID from IPFS, turning it into a byte array
        #   then a nn.Parameter for the module (so that it may be serialized).
        assert CID.is_cid(multihash)
        self.multihash:nn.Parameter = nn.Parameter(strToTensor(multihash), requires_grad=False)

        # Open the file into the CPU so that it may be potentially swapped
        with IPFile(multihash, ipns=ipns, command=ipfsCommand) as heartfile:
            self.heart:nn.Module = t.jit.load(heartfile, map_location='cpu')
        
        # Set the device to be used later for fast computation
        if fastdev is None:
            if t.cuda.is_available():
                fastdev = 'cuda'
            else:
                fastdev = 'cpu'
        self.fastdev:nn.Parameter = nn.Parameter(strToTensor(fastdev), requires_grad=False)

        # Set up the configuration flags for the model inference
        self.freezeHeart:nn.Parameter = nn.Parameter(t.tensor([not ipns], device='cpu'), requires_grad=False)
        self.lstmType:nn.Parameter = nn.Parameter(t.tensor([isinstance(self.heart, nn.LSTM)], device='cpu'), requires_grad=False)
        self.linearType:nn.Parameter = nn.Parameter(t.tensor([isinstance(self.heart, nn.Linear)], device='cpu'), requires_grad=False)
        self.weightName:nn.Parameter = nn.Parameter(strToTensor(weightName), requires_grad=False)
        self.biasName:nn.Parameter = nn.Parameter(strToTensor(biasName), requires_grad=False)
        self.invbiasName:nn.Parameter = nn.Parameter(strToTensor(invbiasName), requires_grad=False)
        self.invertedLinear:nn.Parameter = nn.Parameter(t.tensor([self.linearType], device='cpu'), requires_grad=False)

        # If this is a linear module, check for an inverted bias
        if self.linearType[0]:
            self.invertedLinear[0] = False
            for name, _ in self.named_parameters(recurse=True):
                if name == self.invbiasName:
                    self.invertedLinear[0] = True
                    break
    
    def cid(self):
        # Encode the contained parameter into something understandable by IPFS
        bytelist = self.multihash.tolist()
        return CID.from_bytes(bytearray(bytelist))
    
    def multihash(self):
        # Get the raw bytes of the encoding of the cid
        return self.cid().multihash

    def __forward__(self, a:t.Tensor, b:t.Tensor) -> t.Tensor:
        #
