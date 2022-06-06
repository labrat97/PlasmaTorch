from ..defaults import *
from ..activations import *
from ..conversions import toComplex
from ..sizing import resignal, weightedResample
from ..lens import lens
from ..entanglement import entangle
from .routing import KnowledgeCollider



class Aggregator(nn.Module):
    """An Aggregator is a class used to shove very large signals (likely 196884 samples) into smaller
    collisions from KnowledgeCollider classes. These collisions result in a mask that the signal is then
    lensed into and out of so that an entanglement can ensue minus the 196884^2 complex floats that would
    need to be stored in memory.
    """
    def __init__(self, lensSlots:int=AGGREGATE_LENSES, outputSamples:int=-1, 
    colliders:List[KnowledgeCollider]=None, selectorSide:int=SUPERSINGULAR_PRIMES_LH[3], cdtype:t.dtype=DEFAULT_COMPLEX_DTYPE):
        """Initialize the class so that it may function in a full neural system.

        Args:
            lensSlots (int, optional): The amount of lenses to interpolate between internally. Defaults to AGGREGATE_LENSES.
            outputSamples (int, optional): The amount of samples to output each signal to, negative being no resize. Defaults to -1.
            colliders (List[KnowledgeCollider], optional): A list of KnowledgeColliders to use for forward() evaluation. Defaults to None.
            selectorSide (int, optional): The amount of neurons used after the convolutional encoding from the key basis signal. Defaults to SUPERSINGULAR_PRIMES_LH[3].
            cdtype (t.dtype, optional): The default complex type. Defaults to DEFAULT_COMPLEX_DTYPE.
        """
        # Do not use the attentive resample option available in the aggregator as it is essentially 
        #   just a lensing system.
        super(Aggregator, self).__init__()

        # Used for building complex types
        typeDummy:t.Tensor = toComplex(t.zeros((1), dtype=cdtype))

        # Hold the amount of output samples to be used internally by the module in the nn module parameters
        self.outputSamples:nn.Parameter = nn.Parameter(t.tensor([outputSamples], dtype=t.int64), requires_grad=False)

        # Hold the amount of lenses used in the module in the nn module parameters
        self.lensSlots:nn.Parameter = nn.Parameter(t.tensor([lensSlots], dtype=t.int64), requires_grad=False)
        
        # Each lens to be interpolated between during the signal evaluation
        # The first lens is used for the input signal in a GREISS_SAMPLES size batch,
        #   the second lens is for the post filter evaluated patch also in a GREISS_SAMPLES size batch. 
        self.lensBasis:nn.Parameter = nn.Parameter(t.randn((2, lensSlots, GREISS_SAMPLES), dtype=typeDummy.dtype))
        
        # Because each signal is resignalled to be in to having GREISS_SAMPLES as a size,
        #   the lens selectors select the respective lenses by viewing the entirety and evaluating
        #   to a single value.
        self.lensSelectorProj:nn.Parameter = nn.Parameter(t.randn((2, GREISS_SAMPLES, selectorSide), dtype=typeDummy.dtype))
        # The final convolution needs to be real valued to properly interpolate the lenses
        self.lensSelectorConv:nn.Parameter = nn.Parameter(t.randn((2, selectorSide, 1), dtype=typeDummy.real.dtype))
        # The final polarization needs to ALSO be real valued to properly mix the signal collapse
        self.lensPolarizer:nn.Parameter = nn.Parameter(t.randn((2, selectorSide, 2), dtype=typeDummy.dtype))

        # The starting set of KnowledgeColliders to run the feeding signals through
        self.colliders:nn.ModuleList = nn.ModuleList(colliders)


    def __colliderCaster__(self, collider) -> KnowledgeCollider:
        # Type check
        assert collider is KnowledgeCollider
        
        # Cast-ish for linting and suggestions
        return collider


    def addCollider(self, collider:KnowledgeCollider):
        """Adds a KnowledgeCollider to the internal list of colliders for later evaluation.

        Args:
            collider (KnowledgeCollider): The collider to add to the internal set for later aggregation.
        """
        # Type and value check the arguments
        self.__colliderCaster__(collider)

        # Add the collider, duplicates will just increase the signal gain
        self.colliders.append(collider)


    def __keyToSelection__(self, collider:KnowledgeCollider) -> t.Tensor:
        # Turn the key basis vector into something that can be maximally remapped through Greiss algebra
        greissKey:t.Tensor = tfft.ifft(itanh(collider.keyBasis), n=GREISS_SAMPLES, norm='ortho', dim=-1)

        # Matmul the Greiss key into the latent type used to select lenses
        return tfft.ifft(
            greissKey @ self.lensSelectorProj, n=self.lensSelectorProj.size(-1), 
            norm='ortho', dim=-1)

    def __keyToIdx__(self, selection:t.Tensor) -> t.Tensor:
        # Evaluate the final lens selection through a convolution and an activation,
        #   binding the value between (0.0, 1.0)
        # Use realfold() to collapse into a real value for proper interpolation
        return nnf.sigmoid(realfold(selection) @ self.lensSelectorConv).squeeze(-1)


    def __keyToPolarization__(self, selection:t.Tensor) -> t.Tensor:
        # Evaluate the final lens polarization, used for signal collapse, through convolution
        #   and an activation binding the resultant value between (0.0, 1.0)
        # No squeezing needed as two polarizations come out for each signal individually
        return nnf.sigmoid(selection @ self.lensPolarizer)


    def forward(self, a:t.Tensor, b:t.Tensor, callColliders:bool=False) -> Tuple[t.Tensor]:
        """The default forward call of the module.

        Args:
            a (t.Tensor): The first signal to send through for entanglement.
            b (t.Tensor): The second signal to send through for entanglement.
            callColliders (bool, optional): If True, call the stored colliders with the provided signals to update the internal collisions. Defaults to False.

        Returns:
            Tuple[t.Tensor]: Resultant signal (a, b).
        """
        # Running data for caching the outputs of the internal colliders
        collCount = len(self.colliders)
        ldxArr = [None] * collCount # Lens indices
        polArr = [None] * collCount # Polarizations
        results:Tuple[t.Tensor] = (t.zeros_like(toComplex(a)), t.zeros_like(toComplex(b)))

        # Iterate through the colliders, gathering collisions to get lens interpolation
        #   values for aggregation
        for idx, collModule in enumerate(self.colliders):
            # Do a light casting to a KnowledgeCollider
            collider:KnowledgeCollider = self.__colliderCaster__(collModule)

            # If requested, call the stored colliders
            if callColliders:
                _ = collider.forward(a, b)
            
            # Get the associated implicit lens selection signal through the lens
            #   projection system
            lproj:t.Tensor = self.__keyToSelection__(collider)
            
            # Convert the implicit projection to the interpolated lens index and
            #   entanglement polarization
            ldxArr[idx] = self.__keyToIdx__(lproj)
            polArr[idx] = self.__keyToPolarization__(lproj)

        # Stack all of the lens indexes into a set of interpolatable indicies,
        #   then align to the corners by binding into (-1.0, 1.0)
        ldx:t.Tensor = (2.0 * t.stack(ldxArr, dim=-1)) - 1.0

        # Choose the lens to use, per collision, smoothly through resampling. The
        #   dimensions should be layed out [IO, collision, GREISS_SAMPLES]
        lbSample:t.Tensor = weightedResample(self.lensBasis, ldx, dim=1, ortho=False)
        lInput:t.Tensor = realfold(tfft.ifft(lbSample[0], n=GREISS_SAMPLES, dim=-1, norm='ortho')) # Input -> [collision, GREISS_SAMPLES]
        lOutput:t.Tensor = realfold(tfft.ifft(lbSample[1], n=GREISS_SAMPLES, dim=-1, norm='ortho')) # Output -> [collision, GREISS_SAMPLES]
        lBalancer:t.Tensor = realfold(tfft.ifft(
            nnf.softmax(lbSample[1], dim=0), 
            n=GREISS_SAMPLES, dim=-1, norm='ortho')) # Adds to one, balancer -> [collision, GREISS_SAMPLES]

        # Stack the collisions into the same set of batches found in the lens parameters
        collisions:List[t.Tensor] = [self.__colliderCaster__(collider).lastCollision for collider in self.colliders]

        # Lens the signals into the input collision as an entanglement
        for idx, collision in enumerate(collisions):
            # Apply input lens and resignal
            la:t.Tensor = resignal(
                lens(a, lens=lInput[idx], dim=-1), 
                samples=collision.size(-2), dim=-1)
            lb:t.Tensor = resignal(
                lens(b, lens=lInput[idx], dim=-1),
                samples=collision.size(-1), dim=-1)

            # Entangle the signals with two polarizations to get a proper seperable
            #   collapse signal.
            # Move the two signals to the -2 dim to keep the -1 dim idea consistent.
            entanglement:t.Tensor = entangle(a=la, b=lb, mask=collision, polarization=polArr[idx])

            # Use the output lens with the collapsed output signals from the entanglement.
            # Because of the polarization signal, the final dim is the collapsed signals
            #   from the entanglement. The actual dim to lens is the dim that holds BOTH output
            #   signals or -1.
            # Take the signal that defines the output of the system compared to the
            #   rest of the systems and multiply it to the output to keep the total output
            #   summing to one per sample.
            observation:t.Tensor = lBalancer[idx].unsqueeze(-1) * lens(entanglement, lens=lOutput[idx], dim=-1)

            # Seperate the observation into the appriopriate bins to be returned as the results
            results[0].add_(observation[..., 0]) # Signal a
            results[1].add_(observation[..., 1]) # Signal b
        
        # Return the results as signal a and signal b respectively
        return results
