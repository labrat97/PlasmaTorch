from ..defaults import *
from ..zeta import *
from .routing import KnowledgeCollider
from ..math import *
from ..conversions import nantonum



class HurwitzFilter(KnowledgeCollider):
    """Creates a KnowledgeCollider that runs two signals through the Hurwitz-Zeta function. Using this
    allows for some really irrational expressions of reality if training properly ensues.
    """
    def __init__(self, keySamples:int=DEFAULT_FFT_SAMPLES, ioSamples:int=DEFAULT_FFT_SAMPLES, cdtype:t.dtype=DEFAULT_COMPLEX_DTYPE):
        """Initialize the Hurwitz-Zeta functionality of the class.

        Args:
            keySamples (int, optional): The amount of samples to use for the key basis parameter. Defaults to DEFAULT_FFT_SAMPLES.
            ioSamples (int, optional): The amount of samples to use for input and output to the collider. Defaults to DEFAULT_FFT_SAMPLES.
            cdtype (t.dtype, optional): The default complex type to use in the class. Defaults to DEFAULT_COMPLEX_DTYPE.
        """
        super(HurwitzFilter, self).__init__(keySamples=keySamples, inputSamples=ioSamples, outputSamples=ioSamples, cdtype=cdtype)

        # Store parameters to remap the input values to one another prior to the evaluation of the
        # hurwitz zeta function.
        self.remap:nn.Parameter = nn.Parameter(toComplex(t.eye(ioSamples, dtype=self.cdtype)))


    def __forward__(self, a:t.Tensor, b:t.Tensor) -> t.Tensor:
        # Find the basis vectors of the signal
        afft = fft(a, n=self.inputSamples, dim=-1)
        bfft = fft(b, n=self.inputSamples, dim=-1)

        # Remap the input vectors before the evaluation of the Hurwitz-Zeta function
        softmap = nsoftmax(self.remap, dims=[-1, -2])
        ar = afft @ softmap
        br = bfft @ softmap.transpose(-1, -2)

        # Run through the Hurwitz-Zeta function, get rid of invalid values
        hurAb = nantonum(hzetae(s=afft, a=br)).unsqueeze(-1)
        hurBa = nantonum(hzetae(s=bfft, a=ar)).unsqueeze(-1)

        # Create superposition style output through a tensor product
        return hurAb @ hurBa.transpose(-1, -2)
