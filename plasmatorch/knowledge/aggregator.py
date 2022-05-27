from ..defaults import *
from ..conversions import toComplex
from .routing import KnowledgeCollider


# TODO: Fuck, this isn't needed, integrate functionality into the router; break out method
# TODO: FUCK, THIS IS NEEDED. FUCK THE ROUTER, IT ROUTES AND IS TYPE COMPATIBLE WITH IT'S FILTERS
class Aggregator(nn.Module):
    def __init__(self, lensSlots:int=AGGREGATE_LENSES, outputSamples:int=DEFAULT_FFT_SAMPLES, 
    colliders:List[KnowledgeCollider]=None, cdtype:t.dtype=DEFAULT_COMPLEX_DTYPE):
        # Do not use the attentive resample option available in the aggregator as it is essentially 
        #   just a lensing system.
        super(Aggregator, self).__init__()

        # Used for building complex types
        typeDummy:t.Tensor = toComplex(t.zeros((1), dtype=cdtype))

        # Hold the amount of output samples to be used internally by the module in the nn module parameters
        self.outputSamples:nn.Parameter = nn.Parameter(t.tensor([lensSlots], dtype=t.int64), requires_grad=False)

        # Hold the amount of lenses used in the module in the nn module parameters
        self.lensSlots:nn.Parameter = nn.Parameter(t.tensor([lensSlots], dtype=t.int64), requires_grad=False)
        
        # Each lens to be interpolated between during the signal evaluation
        # The first lens is used for the input signal in a GREISS_SAMPLES size batch,
        #   the second lens is for the post filter evaluated patch also in a GREISS_SAMPLES size batch. 
        self.lensBasis:nn.Parameter = nn.Parameter(t.randn((2, lensSlots, GREISS_SAMPLES), dtype=typeDummy.dtype))
        
        # Because each signal is resignalled to be in to having GREISS_SAMPLES as a size,
        #   the lens selectors select the respective lenses by viewing the entirety and evaluating
        #   to a single value.
        self.lensSelectors:nn.Parameter = nn.Parameter(t.randn((2, GREISS_SAMPLES, 1), dtype=typeDummy.dtype))

        # The starting set of collisions to run the feeding signals through
        self.collisions:nn.ModuleList = nn.ModuleList(colliders)
        self.collision


    def __addCollisionSubstep__(self, collider:KnowledgeCollider):
        self.collisions.append(collider)

    def addCollision(self, collider:KnowledgeCollider, duplicates:t.uint64=0):
        # Type and value check the arguments
        assert collider is KnowledgeCollider

        # Add the collider 
        self.__addCollisionSubstep__(collider)

        # Duplicate the collider the specified number of times
        for _ in range(duplicates):
            self.__addCollisionSubstep__(collider)


    def forward(self, a:t.Tensor, b:t.Tensor) -> t.Tensor:
        # 
