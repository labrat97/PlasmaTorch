from ..defaults import *
from ..conversions import *
from ..sizing import *
from ..activations import *
from ..losses import correlation
from ..lens import lens

from abc import ABC, abstractmethod
import time



class KnowledgeFilter(nn.Module, ABC):
    """
    An abstract class used for creating encapsulated bits of knowledge to be applied
    to signals in a per value way.
    """
    @abstractmethod
    def __init__(self, keySamples:int=DEFAULT_FFT_SAMPLES, inputSamples:int=DEFAULT_FFT_SAMPLES, outputSamples:int=DEFAULT_FFT_SAMPLES, attentiveResample:bool=True, cdtype:t.dtype=DEFAULT_COMPLEX_DTYPE):
        """The abstract constructor for a knowledge filter.

        Args:
            keySamples (int, optional): The amount of samples used to describe each curve's position. Defaults to DEFAULT_FFT_SAMPLES.
            inputSamples (int, optional): The amount of samples to signal the input with. Defaults to DEFAULT_FFT_SAMPLES.
            outputSamples (int, optional): The amount of samples to signal the output with. Defaults to DEFAULT_FFT_SAMPLES.
            attentiveResample (bool, optional): Use a weighted resampling system to find a better view of the input. Defaults to True.
            cdtype (t.dtype, optional): The default datatype for the complex correlation parameter. Defaults to DEFAULT_COMPLEX_DTYPE.
        """
        super(KnowledgeFilter, self).__init__()

        # Create parameters to be modified and stored during runtime
        self.keyBasis:nn.Parameter = nn.Parameter(toComplex(t.zeros((keySamples), dtype=cdtype)), requires_grad=True)
        self.keyCarry:nn.Parameter = nn.Parameter(t.zeros_like(self.keyBasis), requires_grad=True)
        self.corrBias:nn.Parameter = nn.Parameter(t.zeros_like(self.keyBasis), requires_grad=True)
        self.callCount:nn.Parameter = nn.Parameter(t.zeros((1), dtype=t.int64), requires_grad=False)

        # Some runtime data to make the filter optimize the signals coming in
        self.keySamples:int = self.keyBasis.size(-1)
        self.cdtype:t.dtype = self.keyBasis.dtype
        self.inputSamples:int = inputSamples
        if inputSamples == 0:
            self.inputSamples = self.keySamples
        self.outputSamples:int = outputSamples
        if outputSamples == 0:
            self.outputSamples = self.inputSamples
        self.lastForwardExec:nn.Parameter = nn.Parameter(-t.ones((2), dtype=t.float), requires_grad=False)

        # Resample the input with some grid resampled attention if requested
        # This is a linear parameter set which is nice
        self.resampleWeight:nn.Parameter = None
        if attentiveResample:
            self.resampleWeight = nn.Parameter(toComplex(t.zeros((inputSamples), dtype=cdtype)))

        # Cache the last result of the class to carry more data forward into methods optionally
        # This should not be serialized so no Parameter
        self.lastFilter:t.Tensor = None


    def implicitCorrelation(self, x:t.Tensor, isbasis:bool=True) -> t.Tensor:
        """Calculate the stored correlation of the input signal with the tokenized
        basis vectors. This is used to predict what could be inside of the function before
        evaluating said function.

        Args:
            x (t.Tensor): A basis vector (optionally a signal) used for calculation.
            isbasis (bool, optional): If False, the vectors coming in are preFFT'd. Defaults to True.

        Returns:
            t.Tensor: The average correlation accross the samples, curves, and vectors.
        """
        # Calculate the self correlation based on the internal positioning keyBasis
        selfCarry = csigmoid(self.keyCarry)
        selfCorr = csigmoid((selfCarry * self.keyBasis) + self.corrBias)

        # Assert the input signal is signal compatible with the correlation signal
        if x.size(-1) != self.keySamples:
            x = resignal(x, samples=self.keySamples, dim=-1)
        
        return correlation(x=x, y=selfCorr, dim=-1, isbasis=isbasis).mean(dim=-1).mean(dim=-1)
    
    
    def keySignal(self, samples:int=-1) -> t.Tensor:
        """Get the key basis vector encoded as a signal.
        
        Args:
            samples (t.Tensor, optional): The amount of samples to encode the basis vector as. If negative, keep basis sizing. Defaults to -1.

        Returns:
            t.Tensor: The key signal.
        """
        return tfft.ifft(self.keyBasis, n=samples, dim=-1)


    @abstractmethod
    def __forward__(self, x:t.Tensor) -> t.Tensor:
        """Runs a tensor through speculative knowledge after being preformatted.

        Args:
            x (t.Tensor): The set of vectors defining a querying signal.

        Returns:
            t.Tensor: The speculative, queried, knowledge graph output signal.
        """
        pass


    def forward(self, x:t.Tensor) -> t.Tensor:
        # Time the execution of this function and store later
        beginExec:float = time.time()

        # Resample the input vector to match the internal expected sample count
        if self.inputSamples > 0:
            if self.resampleWeight is None:
                # Use a basic Fourier Transform method to uniformly preserve the input signal
                if x.size(-1) != self.inputSamples:
                    wx:t.Tensor = resignal(x, samples=self.inputSamples, dim=-1)
                else:
                    wx:t.Tensor = toComplex(x)
            else:
                # Use a grid resample with bilinear filtering, and centered, with UV coords
                # Make input values bounded appropriately
                actWeight:t.Tensor = softunit(self.resampleWeight, dim=-1)
                # Turn the signal into a continuous signal (essentially a fully differentiable lens)
                sampleWeight:t.Tensor = realfold(ifft(actWeight, n=actWeight.size(-1), dim=-1))
                # Apply the lens to the input values
                wx:t.Tensor = lens(x, lens=sampleWeight, dim=-1)
        else:
            # No valid input sample setting was provided, ensure complex representation
            wx:t.Tensor = toComplex(x)

        # Call the internal __forward__() method so that this method may wrap safely.
        # Timing the execution time is critical here as the total sample count is likely
        #   to reach 196884 elements which can occupy over 4GB of system memory during evaluation.
        #   It also takes a significantly longer amount of time to evaluate that big of a computation.
        beginForward:float = time.time()
        result:t.Tensor = self.__forward__(x=wx)
        self.lastForward = time.time() - beginForward

        # Resample the output tensors to match the internal expected sample count
        if self.outputSamples > 0 and result.size(-1) != self.outputSamples:
            result = resignal(result, samples=self.outputSamples, dim=-1)

        # Log the execution time of this function for later evaluation.
        self.lastExec = time.time() - beginExec

        # Save a reference to the data last returned by this filter
        self.lastFilter = result

        return result


    def __call__(self, x:t.Tensor) -> t.Tensor:
        """Wrap the forward() method.

        Args:
            x (t.Tensor): The signal to run through the filter.

        Returns:
            t.Tensor: The resultant signal.
        """
        return self.forward(x)



class KnowledgeCollider(nn.Module, ABC):
    """
    An abstract class used for creating encapsulated bits of knowledge interaction to be called by
    other KnowledgeColliders or structures looking to collide knowledge from plasmatorch.
    """
    @abstractmethod
    def __init__(self, keySamples:int=DEFAULT_FFT_SAMPLES, inputSamples:int=DEFAULT_FFT_SAMPLES, outputSamples:int=DEFAULT_FFT_SAMPLES, attentiveResample:bool=True, cdtype:t.dtype=DEFAULT_COMPLEX_DTYPE):
        """The abstract constructor for a knowledge collider.

        Args:
            keySamples (int, optional): The amount of samples to describe each curve. Defaults to DEFAULT_FFT_SAMPLES.
            inputSamples (int, optional): The amount of samples to signal the input with. Defaults to DEFAULT_FFT_SAMPLES.
            outputSamples (int, optional): The amount of samples to signal the output with. Defaults to DEFAULT_FFT_SAMPLES.
            attentiveResample (bool, optional): Use a weighted resampling system to find a better view of the input. Defaults to True.
            cdtype (t.dtype, optional): The default datatype for the complex correlation parameter. Defaults to DEFAULT_COMPLEX_DTYPE.
        """
        super(KnowledgeCollider, self).__init__()
        
        # Create parameters to be modified and stored
        self.keyBasis:nn.Parameter = nn.Parameter(toComplex(t.zeros((2, keySamples), dtype=cdtype)), requires_grad=True)
        self.keyCarry:nn.Parameter = nn.Parameter(t.zeros_like(self.keyBasis), requires_grad=True)
        self.corrBias:nn.Parameter = nn.Parameter(t.zeros_like(self.keyBasis), requires_grad=True)
        self.callCount:nn.Parameter = nn.Parameter(t.zeros((1), dtype=t.int64), requires_grad=False)

        # Some runtime data to make the filter optimize the signals coming in
        self.keySamples:int = self.keyBasis.size(-1)
        self.cdtype:t.dtype = self.keyBasis.dtype
        self.inputSamples:int = inputSamples
        if inputSamples == 0:
            self.inputSamples = self.keySamples
        self.outputSamples:int = outputSamples
        if outputSamples == 0:
            self.outputSamples = self.inputSamples
        self.lastForwardExec:nn.Parameter = nn.Parameter(-t.ones((2), dtype=t.float), requires_grad=False)

        # Resample the input with some grid resampled attention if requested
        # This is a linear parameter set which is nice
        self.resampleWeight:nn.Parameter = None
        if attentiveResample:
            self.resampleWeight = nn.Parameter(toComplex(t.zeros((2, inputSamples), dtype=cdtype)))

        # Cache the last result of the class to carry more data forward into methods optionally
        # This should not be saved when serialized, so no Parameter
        self.lastCollision:t.Tensor = None


    def implicitCorrelation(self, a:t.Tensor, b:t.Tensor, isbasis:bool=True) -> t.Tensor:
        """Calculate the stored correlation of the input signal with the tokenized
        basis vectors. This is used to predict what could be inside of the function before
        evaluating said function.

        Args:
            a (t.Tensor): A basis vector (optionally a signal) used for calculation.
            b (t.Tensor): Another basis vector (optionally a signal) used for calculation.
            isbasis (bool, optional): If False, the vectors coming in are preFFT'd. Defaults to True.

        Returns:
            t.Tensor: The average correlation accross the samples, curves, and vectors.
        """
        # Put the self correlation into an easy to process bounds, processing from the key
        selfCarry = csigmoid(self.keyCarry)
        selfCorr = csigmoid((selfCarry * self.keyBasis) + self.corrBias)

        # Resample the input vectors if needed
        if a.size(-1) != self.keySamples:
            a = resignal(x=a, samples=self.keySamples, dim=-1)
        if b.size(-1) != self.keySamples:
            b = resignal(x=b, samples=self.keySamples, dim=-1)

        # Find the respective correlation from the token with the input signals
        acorr:t.Tensor = correlation(x=a, y=selfCorr[0], dim=-1, isbasis=isbasis).mean(dim=-1).mean(dim=-1)
        bcorr:t.Tensor = correlation(x=b, y=selfCorr[1], dim=-1, isbasis=isbasis).mean(dim=-1).mean(dim=-1)

        # Find the mean of the mean correlations
        return (acorr + bcorr) / 2.

    
    def keySignal(self, samples:int=-1) -> t.Tensor:
        """Get the key basis vector encoded as a signal.

        Args:
            samples (t.Tensor, optional): The amount of samples to encode the basis vector as. If negative, keep basis sizing. Defaults to -1.

        Returns:
            t.Tensor: The key signal.
        """
        return tfft.ifft(self.keyBasis, n=samples, dim=-1)


    @abstractmethod
    def __forward__(self, a:t.Tensor, b:t.Tensor) -> t.Tensor:
        """Runs two tensors through comparative knowledge after being preformatted.

        Args:
            a (t.Tensor): The first set of vectors defining an interacting signal.
            b (t.Tensor): The second set of vectors defining another interacting signal.

        Returns:
            t.Tensor: The comparative knowledge graph output signal.
        """
        pass


    def forward(self, a:t.Tensor, b:t.Tensor) -> t.Tensor:
        # Time the execution of this function and store later
        beginExec:float = time.time()

        # Resample the input vectors to match the internal expected sample count
        if self.inputSamples > 0:
            if self.resampler is None:
                # Use a basic Fourier Transform method to uniformly preserve the input signal
                if self.inputSamples > 0 and a.size(-1) != self.inputSamples:
                    wa:t.Tensor = resignal(a, samples=self.inputSamples, dim=-1)
                else:
                    wa:t.Tensor = toComplex(a)
                if self.inputSamples > 0 and b.size(-1) != self.inputSamples:
                    wb:t.Tensor = resignal(b, samples=self.inputSamples, dim=-1)
                else:
                    wb:t.Tensor = toComplex(b)
            else:
                # Use a grid resample with bilinear filtering and centered, UV coords
                # Make input values bounded appropriately
                actWeightA:t.Tensor = softunit(self.resampleWeight[0], dim=-1)
                actWeightB:t.Tensor = softunit(self.resampleWeight[1], dim=-1)

                # Turn the signal into a continuous signal (essentially a fully differentiable lens)
                sampleWeightA:t.Tensor = realfold(ifft(actWeightA, n=actWeightA.size(-1), dim=-1))
                sampleWeightB:t.Tensor = realfold(ifft(actWeightB, n=actWeightB.size(-1), dim=-1))

                # Apply lens to the input values
                wa:t.Tensor = lens(a, lens=sampleWeightA, dim=-1)
                wb:t.Tensor = lens(b, lens=sampleWeightB, dim=-1)
        else:
            wa:t.Tensor = toComplex(a)
            wb:t.Tensor = toComplex(b)
        
        # Call the internal __forward__() method so that this method may wrap safely.
        # Timing the execution time is critical here as the total sample count is likely
        #   to reach 196884 elements which can occupy over 4GB of system memory during evaluation.
        #   It also takes a significantly longer amount of time to evaluate that big of a computation.
        beginForward:float = time.time()
        result:t.Tensor = self.__forward__(a=wa, b=wb)
        self.lastForward = time.time() - beginForward

        # Resample the output matrices to match the internal expected sample count
        if self.outputSamples > 0 and result.size(-1) != self.outputSamples:
            result = resignal(result, samples=self.outputSamples, dim=-1)
        if self.outputSamples > 0 and result.size(-2) != self.outputSamples:
            result = resignal(result, samples=self.outputSamples, dim=-2)

        # Log the execution time of this function for later evaluation.
        self.lastExec = time.time() - beginExec

        # Save a reference to the data last returned by this filter
        self.lastCollision = result

        return result


    def __call__(self, a:t.Tensor, b:t.Tensor) -> t.Tensor:
        """Wrap the forward() method of the collider.

        Args:
            a (t.Tensor): The first signal to collide.
            b (t.Tensor): The second signal to collide.

        Returns:
            t.Tensor: The collided signals.
        """
        return self.forward(a, b)



class KnowledgeRouter(KnowledgeCollider):
    """
    A KnowledgeCollider type class that is used to call other knowledge collider type classes.
    Due to the way that the signal traversal works, this should be a borderline completely unified
    tree traversal method due to the continuous nature. Every single layer of traversal is evaluated
    in parallel, and ever computation is chronologically independent. Every depth will also do a layered
    set of amplitudes from the previous signal, making the potential for things like the harmonic series
    just fall out.
    """
    def __init__(self, maxk:int=3, keySamples:int=DEFAULT_FFT_SAMPLES, outputSamples:int=DEFAULT_FFT_SAMPLES, cdtype:t.dtype=DEFAULT_COMPLEX_DTYPE):
        """The constructor for a KnowledgeRouter, defining performance parameters 

        Args:
            maxk (int, optional): The starting amount of maximum signals to evaluate. Defaults to 3.
            keySamples (int, optional): The amount of samples to describe each curve. Defaults to DEFAULT_FFT_SAMPLES.
            cdtype (t.dtype, optional): The default datatype for the complex correlation parameter. Defaults to DEFAULT_COMPLEX_DTYPE.
        """
        super(KnowledgeRouter, self).__init__(keySamples=keySamples, inputSamples=-1, outputSamples=outputSamples, cdtype=cdtype)

        # Store all of the colliders that the router can call to
        self.correlationMask:nn.Parameter = nn.Parameter(toComplex(t.zeros((self.keySamples, self.keySamples), dtype=self.cdtype)))
        self.correlationPolarization:nn.Parameter = nn.Parameter(t.zeros((1), dtype=self.correlationMask.real.dtype))
        self.subcolliders:nn.ModuleList = nn.ModuleList()
        self.callCounts:nn.ParameterList = nn.ParameterList()
        self.maxk:int = maxk


    def addCollider(self, x:KnowledgeCollider):
        """Adds a knowledge collider to the router, circularly linking the router in the collider.

        Args:
            x (KnowledgeCollider): The collider to add to the router.
        """
        # Add a KnowledgeCollider that can be routed through later
        assert isinstance(x, KnowledgeCollider)
        self.subcolliders.append(x)
        x.routers.append(self)
        
        # Add a count for the system to evaluate later logrithmically
        self.callCounts.append(nn.Parameter(t.ones((1), dtype=t.float64), requires_grad=False))


    def delCollider(self, idx:int) -> Tuple[KnowledgeCollider, t.Tensor]:
        # Remove the knowledge collider and save for later
        collider:KnowledgeCollider = self.subcolliders[idx]
        self.subcolliders = self.subcolliders[:idx].extend(self.subcolliders[idx+1:])
        collider.routers = collider.routers[:idx].extend(collider.routers[idx+1:])

        # Also grab the call counts because those need to go
        count:t.Tensor = self.callCounts[idx]
        self.callCounts = self.callCounts[:idx].extend(self.callCounts[idx+1:])

        return (collider, count)


    def __forward__(self, a:t.Tensor, b:t.Tensor) -> t.Tensor:
        # Find the basis vectors of the input signals
        afft = tfft.fft(a, n=self.keySamples, dim=-1)
        bfft = tfft.fft(b, n=self.keySamples, dim=-1)

        # Entangle the signals with the `nsoftmax` function and a matmul, then sum out orthogonally
        superposition:t.Tensor = (afft.unsqueeze(-1) @ bfft.unsqueeze(-1).transpose(-1,-2)) * nsoftunit(self.correlationMask, dims=[-1,-2])
        afft = superposition.sum(dim=-1)
        bfft = superposition.sum(dim=-2)

        # Flatten the batches to be able to easily index the contained colliders.
        # In this router, the signals are being ignored in their individual channels
        #   and flattened into the rest of the batches. This is done to maximize
        #   the reusability of the knowledge contained.
        afftflat = t.flatten(afft, start_dim=0, end_dim=-2)
        aflat = t.flatten(a, start_dim=0, end_dim=-2)
        bfftflat = t.flatten(bfft, start_dim=0, end_dim=-2)
        bflat = t.flatten(b, start_dim=0, end_dim=-2)
        
        # Create the storage for the correlations
        icorrs:t.Tensor = t.zeros((len(self.subcolliders), aflat[...,0].numel()), dtype=aflat.dtype)

        # Evaluate the correlation for all contained knowledge colliders
        for idx, kcollider in enumerate(self.subcolliders):
            icorrs[idx] = kcollider.implicitCorrelation(a=afftflat, b=bfftflat, isbasis=True)

        # Grab the top correlation indices, choosing the modules to be evaluated
        _, topcorrs = t.topk(icorrs.abs(), k=self.maxk, dim=0, largest=True)
        topcorrs.transpose_(0, 1)

        # Run each signal through each set of knowledge colliders and geometric mean together
        result:t.Tensor = t.zeros_like(aflat)
        for sdx in range(topcorrs.size(0)):
            for kdx in topcorrs[sdx]:
                # Pull the collider
                kcollider = self.subcolliders[kdx]

                # Add the resultant signal to the current collider (router) result signal
                result[sdx].add_(kcollider.forward(a=aflat[sdx], b=bflat[sdx]))
            # Divide by the number of indices used to collect all of the signals
            result[sdx].div_(topcorrs[sdx].numel())

        # Unflatten the resultant signals back to the relevant size
        return result.unflatten(dim=0, sizes=aflat.size()[:-1])
