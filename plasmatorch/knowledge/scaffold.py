from ..defaults import *
from .routing import *
from ..activations import *
from ..conversions import *
from ..math import *
from ..losses import *
from ..entanglement import superposition

from ipyfs import IPFile
import cid as CID


class ScaffoldFilter(KnowledgeCollider):
    def __init__(self, multihash:str, ipns:bool, corrSamples:int=DEFAULT_FFT_SAMPLES, 
        inputSamples:int=DEFAULT_FFT_SAMPLES, outputSamples:int=DEFAULT_FFT_SAMPLES, 
        cdtype:t.dtype=DEFAULT_COMPLEX_DTYPE, ipfsCommand:str='ipfs', fastdev:str=None,
        weightName:str='weight', biasName:str='bias', invbiasName:str='invbias'):

        # Build the superclass
        super(KnowledgeCollider, self).__init__(corrSamples=corrSamples, 
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
        # Remember: The choice to use the CPU is specifically for swap
        self.freezeHeart:nn.Parameter = nn.Parameter(t.tensor([not ipns], device='cpu'), requires_grad=False)
        self.lstmType:nn.Parameter = nn.Parameter(t.tensor([isinstance(self.heart, nn.LSTM)], device='cpu'), requires_grad=False)
        self.linearType:nn.Parameter = nn.Parameter(t.tensor([isinstance(self.heart, nn.Linear)], device='cpu'), requires_grad=False)
        self.weightName:nn.Parameter = nn.Parameter(strToTensor(weightName), requires_grad=False)
        self.biasName:nn.Parameter = nn.Parameter(strToTensor(biasName), requires_grad=False)
        self.invbiasName:nn.Parameter = nn.Parameter(strToTensor(invbiasName), requires_grad=False)
        self.invertedLinear:nn.Parameter = nn.Parameter(t.tensor(self.linearType, device='cpu'), requires_grad=False)

        # Set the appropriate format function, optionally seek params
        self.__specForward = None
        assert not (self.linearType[0] and self.lstmType[0])
        if self.linearType[0]:
            self.__formFunc = self.__linearFormat

            # If this is a linear module, check for an inverted bias
            self.invertedLinear[0] = False
            for name, _ in self.named_parameters(recurse=True):
                if name == self.invbiasName:
                    self.invertedLinear[0] = True
                    break
        elif self.lstmType[0]:
            self.__formFunc = self.__rnnFormat
        else:
            self.__formFunc = None

                
    # TODO: Figure out a module close like method
    
    def cid(self):
        # Encode the contained parameter into something understandable by IPFS
        bytelist = self.multihash.tolist()
        return CID.from_bytes(bytearray(bytelist))
    
    def multihash(self):
        # Get the raw bytes of the encoding of the cid
        return self.cid().multihash

    def __linearFormat(self, a:t.Tensor, b:t.Tensor) -> Tuple[t.Tensor, nn.Module]:
        return a+b
    
    def __rnnFormat(self, a:t.Tensor, b:t.Tensor) -> Tuple[t.Tensor, nn.Module]:
        return a+b

    def __forward__(self, a:t.Tensor, b:t.Tensor) -> t.Tensor:
        # Create a superposition of the incoming signals to be applied to the weights
        abSuper:t.Tensor = superposition(a, b)

        # Apply the superposition to both the normal and transposed versions

