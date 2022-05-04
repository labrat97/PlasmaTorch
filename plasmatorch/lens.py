from .defaults import *
from .sizing import *
from .math import nantonum
from .knowledge import *

from enum import Enum

class PolarLensPosition(Enum):
    """Defines the discrete direction of observation through a lens that has only two
    directions to be observed through.
    """
    NS = 0
    SN = 1

class InterferringLensPosition(Enum):
    """Defines the discrete direction of observation through a lens that can be viewed
    in a discrete cartesean setting.
    """
    NS = 0
    SN = 1
    WE = 2
    EW = 3

class Lens(KnowledgeFilter):
    """Defines a lens that can be viewed through two discrete directions (hence the
    KnowledgeFilter inheritence). Using this allows a signal to be processed as a monster
    grouping temporarily, then compressed through a lens into the size and distortion image
    needed for the application. If the lens is viewed in the opposite direction, the distortion
    is reversed. It should be noted that this distortion can be incredibly lossy.
    """
    def __init__(self, samples:int=DEFAULT_SIGNAL_LENS_SAMPLES, padding:int=DEFAULT_SIGNAL_LENS_PADDING,
        corrSamples:int=DEFAULT_FFT_SAMPLES, cdtype:t.dtype=DEFAULT_COMPLEX_DTYPE):

        super(Lens, self).__init__(corrSamples=corrSamples, inputSamples=samples, outputSamples=samples, 
            attentiveResample=False, cdtype=cdtype)

        # Store the parameters for the lens basis vector, to be used later in an irfft call
        self.lensBasis:nn.Parameter = nn.Parameter(t.zeros((samples), dtype=self.cdtype))

        # Store the direction of the lens evaluation, defined by the above enum
        self.lensDir:nn.Parameter = nn.Parameter(PolarLensPosition.NS + t.zeros((1), dtype=t.int8), requires_grad=False)

        # Store how much padding to add to the signal (expanded) circularly
        self.signalPadding:nn.Parameter = nn.Parameter(abs(padding) + t.zeros((1), dtype=t.int64), requires_grad=False)

    def setDirection(self, dir:PolarLensPosition):
        # Translate to what the tensor can understand
        self.lensDir[0] = int(dir)

    def __forward__(self, x:t.Tensor) -> t.Tensor:
        # Create the lens as a signal
        lensIntrinsics:t.Tensor = tfft.irfft(self.lensBasis, n=self.lensBasis.size(-1), dim=-1, norm='ortho')
        lensSquish:t.Tensor = (lensIntrinsics + 1.) / 2.

        # Clip the lens into a circular padding aligning to the corners
        lensCast:t.Tensor = lensSquish.to(t.int64, non_blocking=True)
        lensClip:t.Tensor = (lensIntrinsics.abs() > 1.).to(t.int64, non_blocking=True)
        lensDir:t.Tensor = lensIntrinsics.sign()

        # Apply the clipping
        clippedIntrinsics:t.Tensor = ((lensSquish - (lensCast * lensClip * lensDir)) * 2.) + 1.

        # Add padding to the input signal to allow viewing outside of the maximal signal representation
        xpad:t.Tensor = paddim(x=x, lowpad=self.signalPadding, highpad=self.signalPadding, dim=-1, mode='circular')
        
        # Modify the lens intrinsics to be bound within the unpadded signal in the padded signal
        lensScalar:float = x.size(-1) / (x.size(-1) + (2. * self.signalPadding[0]))
        padIntrinsics:t.Tensor = lensScalar * clippedIntrinsics

        # Apply the lens through a weighted resample
        return weightedResample(xpad, padIntrinsics, dim=-1)
