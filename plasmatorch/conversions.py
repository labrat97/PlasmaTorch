from .defaults import *
from .math import xbias


class Smear(nn.Module):
    """
    Turns every single point of a tensor into a linear spread to "smear" the signal.
    """
    def __init__(self, samples:int=DEFAULT_FFT_SAMPLES, lowerScalar:float=1./16, 
        upperScalar:float=1./16, dtype:t.dtype=DEFAULT_DTYPE):
        """Initializes the Smear module.

        Args:
            samples (int, optional): The amount of samples to create the smear with. Defaults to DEFAULT_FFT_SAMPLES.
            lowerScalar (float, optional): The lower scalar to the input points to create the spread with. Defaults to 1./16.
            upperScalar (float, optional): The upper scalar to the input points to create the spread with. Defaults to 1./16.
            dtype (t.dtype, optional): The datatype to create the smears with. Defaults to DEFAULT_DTYPE.
        """
        super(Smear, self).__init__()

        # Store the parameters of the system
        self.smearBias:nn.Parameter = nn.Parameter(t.zeros(1, dtype=dtype))
        self.smearWindow:nn.Parameter = nn.Parameter(t.tensor([-lowerScalar, upperScalar]).type(dtype))

        # Cache a bias generation for later modification
        self.__iter__ = nn.Parameter(xbias(n=samples, bias=0), requires_grad=False)
    
    def forward(self, x:t.Tensor) -> t.Tensor:
        """The default forward call of the module.

        Args:
            x (t.Tensor): The set of points to smear.

        Returns:
            t.Tensor: The smeared input points.
        """
        # Bias the points smeared
        xBias:t.Tensor = x + self.smearBias
        if self.__iter__.size(-1) <= 1:
            return xBias

        # Pull the smear multipliers
        lowerSmear:t.Tensor = self.smearWindow[0]
        upperSmear:t.Tensor = self.smearWindow[1]
      
        # Calculate the iteration modifiers
        xRange:t.Tensor = (upperSmear - lowerSmear) * xBias
        xLow:t.Tensor = ((1 + lowerSmear) * xBias)

        # Modify the built in iteration cache
        return (xRange * self.__iter__) + xLow

@ts
def nantonum(x:t.Tensor, nan:Union[float, None]=None, posinf:Union[float, None]=None, neginf:Union[float, None]=None) -> t.Tensor:
    """Performs the pytorch method `nan_to_num()` on a potentially complex number.

    Args:
        x (t.Tensor): The tensor to convert the poorly bounded values in.
        nan (Union[float, None], optional): The replacement value for a NaN. Defaults to None.
        posinf (Union[float, None], optional): The replacement value for a positive infinity. Defaults to None.
        neginf (Union[float, None], optional): The replacement value for a negative infinity. Defaults to None.

    Returns:
        t.Tensor: The input tensor, `x`, with poorly bounded NaNs and Infs replaced.
    """
    # Already implemented
    if not x.is_complex(): return t.nan_to_num(x, nan=nan, posinf=posinf, neginf=neginf)

    # Do it on a per element basis
    real = x.real.nan_to_num(nan=nan, posinf=posinf, neginf=neginf)
    imag = x.imag.nan_to_num(nan=nan, posinf=posinf, neginf=neginf)

    # Create the stablized output and return
    return t.view_as_complex(t.stack((real, imag), dim=-1))

@ts
def toComplex(x:t.Tensor) -> t.Tensor:
    """Converts a tensor to be of a complex value.

    Args:
        x (t.Tensor): The tensor to turn into a complex tensor.

    Returns:
        t.Tensor: `x`, but for sure as a complex datatype.
    """
    # Already done
    if x.is_complex(): return x
    
    # Turn into a complex number
    complexProto = t.stack((x, t.zeros_like(x)), dim=-1)
    return t.view_as_complex(complexProto)

@ts
def strToTensor(x:str) -> t.Tensor:
    """Convert a string into a pytorch native tensor.

    Args:
        x (str): The string to turn into a tensor.

    Returns:
        t.Tensor: The input string as a tensor (int32).
    """
    # Prepare memory for construction
    rawstr = t.zeros((len(x)), dtype=t.int32, device='cpu')

    # Copy string
    for idx, char in enumerate(x):
        rawstr[idx] = ord(char)
    
    return rawstr

@ts
def tensorToStr(x:t.Tensor, dim:int=-1) -> List[str]:
    """Converts a tensor to a list of python strings.

    Args:
        x (t.Tensor): The tensor to parse.
        dim (int, optional): The dimension that serializes the string. Defaults to -1.

    Returns:
        List[str]: The list of strings translated from the parsed tensor.
    """
    # Put the string dimension in the appropriate place for conversion
    wx = x.transpose(dim, -1)

    # Make sure it can be represented in python natively
    if len(x.size()) == 1:
        wx = x.unsqueeze(0)
    else:
        wx = x.flatten(end_dim=-2)

    # Prepare python traced output
    pystr:List[str] = []

    # Copy the string out of the tensor into Python's format
    for idx in range(wx.size(0)):
        # Isolate
        target:t.Tensor = wx[idx]
        build:str = ''

        # Copy element by element
        for jdx in range(target.size(0)):
            build += chr(target[jdx])
        
        # Add the string to the output list
        pystr.append(build)

    # Return all embedded strings in a list
    return pystr


class RealObserver(nn.Module):
    """
    Folds a potentially complex signal into a real value when called.
    """
    def __init__(self, units:int = 1, dtype:t.dtype = DEFAULT_DTYPE):
        """Initialize the real observation system.

        Args:
            units (int, optional): The amount of reals to observe for each element. Defaults to 1.
            dtype (t.dtype, optional): The datatype for the polarization parameter. Defaults to DEFAULT_DTYPE.
        """
        super(RealObserver, self).__init__()

        # Create the polarization parameter and type check
        self.polarization:nn.Parameter = nn.Parameter(t.zeros((2, units), dtype=dtype))
        assert self.polarization.is_complex() == False
    
    def forward(self, x:t.Tensor) -> t.Tensor:
        """The default forward call of the module.

        Args:
            x (t.Tensor): The set of elements to convert to real values.

        Returns:
            t.Tensor: The real value, observed, tensor.
        """
        # Type forcing
        wx:t.Tensor = toComplex(x)

        # Apply the polarization between the complex signal domains
        return (t.cos(self.polarization[0]) * wx.real) \
            + (t.sin(self.polarization[1]) * wx.imag)


class ComplexObserver(nn.Module):
    """
    Folds a real signal into a complex signal when called.
    """
    def __init__(self, units:int = 1, dtype:t.dtype = DEFAULT_DTYPE):
        """Initialize the complex observation system.

        Args:
            units (int, optional): The amount of complex numbers to observe for each element. Defaults to 1.
            dtype (t.dtype, optional): The datatype for the polarization parameter. Defaults to DEFAULT_DTYPE.
        """
        super(ComplexObserver, self).__init__()

        # Create the polarization parameter then type check
        self.polarization:nn.Parameter = nn.Parameter(t.zeros((2, units), dtype=dtype))
        assert self.polarization.is_complex() == False

    def forward(self, x:t.Tensor) -> t.Tensor:
        """The default forward call of the module.

        Args:
            x (t.Tensor): The set of elements to convert to complex values.

        Returns:
            t.Tensor: The complex value, observed, tensor.
        """
        # Type checking
        assert x.is_complex() == False

        # Apply polarization to pull into complex plane
        xReal:t.Tensor = t.cos(self.polarization[0]) * x
        xImag:t.Tensor = t.sin(self.polarization[1]) * x
        
        # Resize and turn complex
        xReal.unsqueeze_(-1)
        xImag.unsqueeze_(-1)
        return t.view_as_complex(t.cat((xReal, xImag), dim=-1))
