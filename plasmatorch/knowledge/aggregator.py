from ..defaults import *
from .routing import KnowledgeCollider
import cid

AGGREGATOR_ID_LENGTH = 32

class WeightedAggregator(nn.Module):
    def __init__(self, samplesA:int=DEFAULT_FFT_SAMPLES, dimA:int=-1, samplesB:int=DEFAULT_FFT_SAMPLES, dimB:int=-1):
        super(WeightedAggregator, self).__init__()

        # Create an ID token to grab the appropriate lens for the aggregator
        # Using the cpu as the device is meant to keep the parameter in the more bountiful
        #   system memory
        self.lensID:nn.Parameter = nn.Parameter(t.randn((32), dtype=t.uint8, device='cpu'), requires_grad=False)
        self.samples:nn.Parameter = nn.Parameter(t.tensor([samplesA, samplesB], dtype=t.int64, device='cpu'), requires_grad=False)

    def __idstr__(self) -> str:
        # Convert the bytes stored in the class to a cid
        rawcid = bytes(self.lensID.cpu())
        pycid = cid.from_bytes(rawcid)

        # Return the cid as a string
        return pycid.multihash

    def __addSelf__(self, filter:KnowledgeCollider) -> Tuple[nn.Parameter, nn.Parameter]:
        # Get the key for the aggregator
        idstr = self.__idstr__()

        # Make sure both lenses exist in the selected KnowledgeFilter
        try:
            lensA = filter.aggregateLenses[0][idstr]
        except KeyError:
            lensA = nn.Parameter(t.zeros((self.samples[0]), dtype=t.float32, device='cpu'))
            filter.aggregateLenses[0].update({idstr: lensA})
        try:
            lensB = filter.aggregateLenses[1][idstr]
        except KeyError:
            lensB = nn.Parameter(t.zeros((self.samples[1]), dtype=t.float32, device='cpu'))
            filter.aggregateLenses[1].update({idstr: lensB})
        
        # Show the calling function what the lenses are
        return (lensA, lensB)
    
    def blank(self, filter:KnowledgeCollider) -> Tuple[t.Tensor, t.Tensor]:
        # Set up the output lenses
        lensA, lensB = self.__addSelf__(filter)

        

    def forward(self, a:t.Tensor, b:t.Tensor, filter:KnowledgeCollider) -> Tuple[t.Tensor, t.Tensor]:
        # Set up the output lenses
        lensA, lensB = self.__addSelf__(filter)

        # Call the associated filter
        superposition = filter.forward(a=a, b=b)

        # Resample the superposition
