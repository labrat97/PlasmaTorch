from .__defimp__ import *
from .conversions import toComplex



@ts
def linspace(start:Union[float, complex, t.Tensor], end:Union[float, complex, t.Tensor], steps:int, device:t.device=DEFAULT_FAST_DEV) -> t.Tensor:
    """Wrap `t.linspace()` so that when there is a case where `steps` == 1, the average of
    `start` and `end` come out instead of just `start`.

    Args:
        start (Union[float, complex, t.Tensor]): The starting value of the linear space. Must be a singular value.
        end (Union[float, complex, t.Tensor]): The ending value of the linear space. Must be a singular value.
        steps (int): The amount of steps to return from the linear space.
        device (str, optional): The device to render the linear space on. Defaults to DEFAULT_FAST_DEV.

    Returns:
        t.Tensor: The rendered linear space.
    """
    # Convert the start tensor to a singular number
    ws:Union[float, complex] = 0.
    if isinstance(start, t.Tensor):
        # Check that the tensor is a single value
        assert start.numel() == 1

        # Cast
        if start.is_complex():
            ws = complex(start.flatten()[0])
        else:
            ws = float(start.flatten()[0])
    else:
        # Cast typing down for jit
        ws = start
    
    # Convert the end tensor to a singular number
    we:Union[float, complex] = 0.
    if isinstance(end, t.Tensor):
        # Check that the tensor is a single value
        assert end.numel() == 1

        # Cast
        if end.is_complex():
            we = complex(end.flatten()[0])
        else:
            we = float(end.flatten()[0])
    else:
        # Cast typing down for jit
        we = end

    # The quick return case where steps is equal to one
    if steps == 1:
        # This is done to avoid a multitude of TorchScript bugs
        # If either of the unions are complex, a tensor must be created
        #   of them individually or an error occures in aten. If I add the
        #   complex numbers at all outside of matrix form it results in an
        #   error
        if isinstance(ws, complex) or isinstance(we, complex):
            return (t.tensor(complex(ws), device=device).unsqueeze(0) 
                + t.tensor(complex(we), device=device).unsqueeze(0)) / 2.
        
        # I feel like this should have worked for the previous statement
        #   without having to cast
        return t.tensor([(float(ws) + float(we)) / 2.], device=device)

    # Error correction and default behavior when the above case doesn't test true
    #   is passed directly to `t.linspace()`
    return t.linspace(start=ws, end=we, steps=steps, device=device)



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
            reg (bool, optional): Apply the regularization needed for the cdf to approach 1. Defaults to False.

    Returns:
            t.Tensor: A sampled set of values with the same size as the input.
    """
    # Breif argument checking
    assert not lowStd.is_complex()
    assert not highStd.is_complex()

    # Constants for evaluation
    PHI:t.Tensor = phi(device=x.device)
    TAU:t.Tensor = tau(device=x.device)

    # Grab the correct side of the curve
    belowMean:t.Tensor = t.le(x, mean).type(t.uint8, non_blocking=True)
    std:t.Tensor = (belowMean * lowStd) + ((1 - belowMean) * highStd)

    # Calculate the gaussian curve
    top:t.Tensor = (x - mean).abs()

    # Never hits 0
    bottom:t.Tensor = ((1. / PHI) * t.log(1 + t.exp(PHI * std)))
    if bottom.dtype == t.float16 or bottom.dtype == t.complex32:
        bottom.clamp_(min=1e-4, max=1e4)
    else:
        bottom.clamp_(min=1e-12, max=1e12)
    
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
    def __init__(self, channels:int, dtype:t.dtype = DEFAULT_DTYPE, device:t.device=DEFAULT_FAST_DEV):
        """Builds a new LinearGauss structure.

        Args:
                channels (int): The amount of linear gausses to build together. Must be broadcastable to
                    the provided set of channels.
                dtype (t.dtype): The type of the parameters used to calculate the gaussian curves.
                device (t.device): The device to use for the module. Defaults to DEFAULT_FAST_DEV.
        """
        super(LinearGauss, self).__init__()

        self.channels:int = channels

        self.mean:nn.Parameter = nn.Parameter(t.zeros((self.channels), dtype=dtype, device=device))
        self.lowStd:nn.Parameter = nn.Parameter(t.zeros((self.channels), dtype=dtype, device=device))
        self.highStd:nn.Parameter = nn.Parameter(t.zeros((self.channels), dtype=dtype, device=device))
        
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
