from .defaults import *


# Turn a pointwise signal into a smearwise one
class Smear(nn.Module):
  def __init__(self, samples:int = DEFAULT_FFT_SAMPLES, lowerScalar:float = 1./16, 
    upperScalar:float = 1./16, dtype:t.dtype = DEFAULT_DTYPE):
    super(Smear, self).__init__()

    self.samples:int = samples
    self.smearBias:nn.Parameter = nn.Parameter(t.zeros(1, dtype=dtype))
    self.smearWindow:nn.Parameter = nn.Parameter(t.tensor([-lowerScalar, upperScalar]).type(dtype))

    self.__iter = t.Tensor(
      [builder / (self.samples-1) for builder in range(self.samples)]
    ).type(dtype).detach()
  
  def forward(self, x:t.Tensor) -> t.Tensor:
    xBias:t.Tensor = x + self.smearBias
    if self.samples <= 1:
      return xBias

    lowerSmear:t.Tensor = self.smearWindow[0]
    upperSmear:t.Tensor = self.smearWindow[1]
    xRange:t.Tensor = (upperSmear - lowerSmear) * xBias
    xLow:t.Tensor = ((1 + lowerSmear) * xBias)

    return (xRange * self.__iter) + xLow

@ts
def nantonum(x:t.Tensor, nan:Union[float, None]=None, posinf:Union[float, None]=None, neginf:Union[float, None]=None) -> t.Tensor:
  # Already implemented
  if not x.is_complex(): return t.nan_to_num(x, nan=nan, posinf=posinf, neginf=neginf)

  # Do it on a per element basis
  real = x.real.nan_to_num(nan=nan, posinf=posinf, neginf=neginf)
  imag = x.imag.nan_to_num(nan=nan, posinf=posinf, neginf=neginf)
  
  # Create the stablized output and return
  return t.view_as_complex(t.stack((real, imag), dim=-1))

@ts
def toComplex(x:t.Tensor) -> t.Tensor:
  # Already done
  if x.is_complex(): return x
  
  # Turn into a complex number
  complexProto = t.stack((x, t.zeros_like(x)), dim=-1)
  return t.view_as_complex(complexProto)

@ts
def strToTensor(x:str) -> t.Tensor:
  # Prepare memory for construction
  rawstr = t.zeros((len(x)), dtype=t.int32, device='cpu')

  # Copy string
  for idx, char in enumerate(x):
    rawstr[idx] = ord(char)
  
  return rawstr

@ts
def tensorToStr(x:t.Tensor, dim:int=-1) -> List[str]:
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
    def __init__(self, units:int = 1, dtype:t.dtype = DEFAULT_DTYPE):
        super(RealObserver, self).__init__()

        # Create the polarization parameter and type check
        self.polarization:nn.Parameter = nn.Parameter(t.zeros((units), dtype=dtype))
        assert self.polarization.is_complex() == False
    
    def forward(self, x:t.Tensor) -> t.Tensor:
        # Type checking
        assert x.is_complex()

        # Apply the polarization between the complex signal domains
        return (t.cos(self.polarization) * x.real) \
            + (t.sin(self.polarization) * x.imag)


class ComplexObserver(nn.Module):
    def __init__(self, units:int = 1, dtype:t.dtype = DEFAULT_DTYPE):
        super(ComplexObserver, self).__init__()

        # Create the polarization parameter then type check
        self.polarization:nn.Parameter = nn.Parameter(t.zeros((units), dtype=dtype))
        assert self.polarization.is_complex() == False

    def forward(self, x:t.Tensor) -> t.Tensor:
        # Type checking
        assert x.is_complex() == False

        # Apply polarization to pull into complex plane
        xReal:t.Tensor = t.cos(self.polarization) * x
        xImag:t.Tensor = t.sin(self.polarization) * x
        
        # Resize and turn complex
        xReal.unsqueeze_(-1)
        xImag.unsqueeze_(-1)
        return t.view_as_complex(t.cat((xReal, xImag), dim=-1))
