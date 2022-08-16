from .defaults import *
from .math import phi, tau
from .conversions import toComplex



@ts
def irregularGauss(x:t.Tensor, mean:t.Tensor, lowStd:t.Tensor, highStd:t.Tensor, reg:bool=False) -> t.Tensor:
    """Generates an piecewise Gaussian curve according to the provided parameters.

    Args:
            x (t.Tensor): The sampling value for the curve with indefinite size.
            mean (t.Tensor): The means that generate the peaks of the function which
                has a shape that is broadcastable upon x.
            lowStd (t.Tensor): The standard deviation to use when the function is below
                the defined mean. The size must be broadcastable upon x.
            highStd (t.Tensor): The standard deviation to use when the function is
                above the defined mean. The size must be broadcastable upon x.
            reg (bool): Apply the regularization needed for the cdf to approach 1. Defaults to False.

    Returns:
            t.Tensor: A sampled set of values with the same size as the input.
    """
    # Breif argument checking
    assert not lowStd.is_complex()
    assert not highStd.is_complex()

    # Constants for evaluation
    PHI:t.Tensor = phi()
    TAU:t.Tensor = tau()

    # Grab the correct side of the curve
    belowMean:t.Tensor = t.le(x, mean).to(t.uint8)
    std:t.Tensor = (belowMean * lowStd) + ((1 - belowMean) * highStd)

    # Calculate the gaussian curve
    top:t.Tensor = (x - mean).abs()

    # Never hits 0
    bottom:t.Tensor = ((1. / PHI) * t.log(1 + t.exp(PHI * std)))
    if bottom.dtype == t.float16 or bottom.dtype == t.complex32:
        bottom.clamp_(min=1e-4, max=1e4)
    else:
        bottom.clamp_(min=1e-18, max=1e18)
    
    # Calculate the normal distribution
    factor:t.Tensor = top / bottom
    result:t.Tensor = t.exp(-0.5 * t.pow(factor, 2.))
    if not reg:
        return result
    
    # Regulate the output so that the cdf approaches 0 at inf
    regulator = 1. / (bottom * t.sqrt(TAU))
    return result * regulator

class LinearGauss(nn.Module):
    """
    A linearly tuned irregular gaussian function to be used as an activation layer of sorts.
    """
    def __init__(self, channels:int, dtype:t.dtype = DEFAULT_DTYPE):
        """Builds a new LinearGauss structure.

        Args:
                channels (int): The amount of linear gausses to build together. Must be broadcastable to
                    the provided set of channels.
                dtype (t.dtype): The type of the parameters used to calculate the gaussian curves.
        """
        super(LinearGauss, self).__init__()

        self.channels:int = channels

        self.mean:nn.Parameter = nn.Parameter(t.zeros((self.channels), dtype=dtype))
        self.lowStd:nn.Parameter = nn.Parameter(t.zeros((self.channels), dtype=dtype))
        self.highStd:nn.Parameter = nn.Parameter(t.zeros((self.channels), dtype=dtype))
        
        self.isComplex:bool = t.is_complex(self.mean)

    def forward(self, x: t.Tensor) -> t.Tensor:
        """The default forward call of the module.

        Args:
            x (t.Tensor): The signal to evaluate.

        Returns:
            t.Tensor: The `irregularGauss()` sampled input signal.
        """
        # Handle the evaluation of a complex number in a non-complex system`
        inputComplex:bool = t.is_complex(x)

        # Move channels if needed
        if self.channels > 1:
            x = x.transpose(-2, -1)

        if inputComplex and not self.isComplex:
            real:t.Tensor = irregularGauss(x=x.real, mean=self.mean, lowStd=self.lowStd, highStd=self.highStd)
            imag:t.Tensor = irregularGauss(x=x.imag, mean=self.mean, lowStd=self.lowStd, highStd=self.highStd)
            
            # Move channels if needed for reconstruction
            if self.channels > 1:
                real = real.transpose(-1, -2)
                imag = imag.transpose(-1, -2)

            return t.view_as_complex(t.stack((real, imag), dim=-1))
        
        # Handle evaluation in a complex system
        if self.isComplex:
            if not inputComplex:
                x = toComplex(x)
            real:t.Tensor = irregularGauss(x=x.real, mean=self.mean.real, lowStd=self.lowStd.real, highStd=self.highStd.real)
            imag:t.Tensor = irregularGauss(x=x.imag, mean=self.mean.imag, lowStd=self.lowStd.imag, highStd=self.highStd.imag)

            # Move channels if needed for the reconstruction
            if self.channels > 1:
                real = real.transpose(-1, -2)
                imag = imag.transpose(-1, -2)

            return t.view_as_complex(t.stack((real, imag), dim=-1))
        

        # Calculate most default result
        result = irregularGauss(x=x, mean=self.mean, lowStd=self.lowStd, highStd=self.highStd)
        
        # Move channels if needed for return
        if self.channels > 1:
            return result.transpose(-1, -2)
        return result
