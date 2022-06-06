from ..defaults import *
from ..zeta import *
from .routing import KnowledgeCollider
from ..conversions import nantonum, toComplex


class LerchFilter(KnowledgeCollider):
    """Creates a KnowledgeCollider that runs two signals through the Lerch-Zeta function. Using this
    allows for some really irrational expressions of reality if training properly ensues.
    """
    def __init__(self, keySamples:int=DEFAULT_FFT_SAMPLES, ioSamples:int=DEFAULT_FFT_SAMPLES, cdtype:t.dtype=DEFAULT_COMPLEX_DTYPE):
        """Initialize the Lerch-Zeta functionality of the class.

        Args:
            keySamples (int, optional): The amount of samples to use for the key basis parameter. Defaults to DEFAULT_FFT_SAMPLES.
            ioSamples (int, optional): The amount of samples to use for input and output to the collider. Defaults to DEFAULT_FFT_SAMPLES.
            cdtype (t.dtype, optional): The default complex type to use in the class. Defaults to DEFAULT_COMPLEX_DTYPE.
        """
        super(LerchFilter, self).__init__(keySamples=keySamples, inputSamples=ioSamples, outputSamples=ioSamples, cdtype=cdtype)

        # Store the internal lambda values and plug into the lambda section of the Lerch-Zeta function
        self.lambdas:nn.Parameter = nn.Parameter(toComplex(t.randn((2, ioSamples), dtype=self.cdtype)))

    def __forward__(self, a:t.Tensor, b:t.Tensor) -> t.Tensor:
        # Evaluate the Lerch-Zeta function at all of the values provided
        lab = nantonum(lerche(lam=self.lambdas[0], s=a, a=b))
        lba = nantonum(lerche(lam=self.lambdas[1], s=b, a=a))

        # Create a tensor product out of the resultant vectors and return the
        # superposition of the signals.
        return lab.unsqueeze(-1) @ lba.unsqueeze(-1).transpose(-1, -2)
