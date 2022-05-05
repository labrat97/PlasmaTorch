from .defaults import *
from .sizing import *
from .knowledge import *

from enum import Enum

class PolarLensPosition(Enum):
    """Defines the discrete direction of observation through a lens that has only two
    directions to be observed through.
    """
    NS = 0
    SN = 1

class PolarLens(KnowledgeFilter):
    """Defines a lens that can be viewed through two discrete directions (hence the
    KnowledgeFilter inheritence). Using this allows a signal to be processed as a monster
    grouping temporarily, then compressed through a lens into the size and distortion image
    needed for the application. If the lens is viewed in the opposite direction, the distortion
    is reversed. It should be noted that this distortion can be incredibly lossy.
    """
    def __init__(self, samples:int=DEFAULT_SIGNAL_LENS_SAMPLES, padding:int=DEFAULT_SIGNAL_LENS_PADDING,
        corrSamples:int=DEFAULT_FFT_SAMPLES, cdtype:t.dtype=DEFAULT_COMPLEX_DTYPE):

        super(PolarLens, self).__init__(corrSamples=corrSamples, inputSamples=-1, outputSamples=samples, 
            attentiveResample=False, cdtype=cdtype)

        # Store the parameters for the lens basis vector, to be used later in an irfft call
        self.lensBasis:nn.Parameter = nn.Parameter(t.zeros((GREISS_SAMPLES), dtype=self.cdtype))

        # Store the direction of the lens evaluation, defined by the above enum
        self.lensDir:nn.Parameter = nn.Parameter((2 * int(PolarLensPosition.NS)) - 1 + t.zeros((1), dtype=t.int8), requires_grad=False)

        # Store how much padding to add to the signal before the lensing, circularly
        self.signalPadding:nn.Parameter = nn.Parameter(abs(padding) + t.zeros((1), dtype=t.int64), requires_grad=False)

    def setDirection(self, dir:PolarLensPosition):
        # Translate to what the tensor can understand in the equation
        self.lensDir[0] = (2 * int(dir)) - 1

    def __forward__(self, x:t.Tensor) -> t.Tensor:
        # Create the lens as a signal
        lensIntrinsics:t.Tensor = self.lensDir * tfft.irfft(self.lensBasis, n=self.lensBasis.size(-1), dim=-1, norm='ortho')
        lensSquish:t.Tensor = (lensIntrinsics + 1.) / 2.

        # Clip the lens into a circular padding aligning to the corners
        lensCast:t.Tensor = lensSquish.to(t.int64, non_blocking=True)
        lensClip:t.Tensor = (lensIntrinsics.abs() > 1.).to(t.int64, non_blocking=True)
        lensSign:t.Tensor = lensIntrinsics.sign()

        # Apply the clipping
        clippedIntrinsics:t.Tensor = ((lensSquish - (lensCast * lensClip * lensSign)) * 2.) + 1.

        # Add padding to the input signal to allow viewing outside of the maximal signal representation
        xpad:t.Tensor = paddim(x=x, lowpad=self.signalPadding, highpad=self.signalPadding, dim=-1, mode='circular')
        
        # Modify the lens intrinsics to be bound within the unpadded signal in the padded signal
        lensScalar:float = x.size(-1) / (x.size(-1) + (2. * self.signalPadding[0]))
        padIntrinsics:t.Tensor = lensScalar * clippedIntrinsics

        # Apply the lens through a weighted resample
        return weightedResample(xpad, padIntrinsics, dim=-1)


class InterferringLensPosition(Enum):
    """Defines the discrete direction of observation through a lens that can be viewed
    in a discrete cartesean setting.
    """
    NSWE = 0b0 | (0b0 << 1)
    SNWE = 0b1 | (0b0 << 1)
    NSEW = 0b0 | (0b1 << 1)
    SNEW = 0b1 | (0b1 << 1)

class InterferringLens(KnowledgeCollider):
    """Defines a lens that can be viewed through four discrete directions (hence the
    KnowledgeCollider inheritence). Using this allows a coupling of signals to be processed as a monster
    grouping temporarily, then compressed through a prism like lens into the size and distortion
    needed for the application. Viewing the lens through an opposite direction
    """
    def __init__(self, samples:int=DEFAULT_SIGNAL_LENS_SAMPLES, padding:int=DEFAULT_SIGNAL_LENS_PADDING,
        corrSamples:int=DEFAULT_FFT_SAMPLES, cdtype:t.dtype=DEFAULT_COMPLEX_DTYPE):

        super(InterferringLens, self).__init__(corrSamples=corrSamples, inputSamples=-1, outputSamples=samples, 
            attentiveResample=False, cdtype=cdtype)

        # Hold two PolarLenses internally and route accordingly
        self.nsLens:PolarLens = PolarLens(samples=samples, padding=padding, corrSamples=corrSamples, cdtype=self.cdtype)
        self.weLens:PolarLens = PolarLens(samples=samples, padding=padding, corrSamples=corrSamples, cdtype=self.cdtype)

    def setDirection(self, dir:InterferringLensPosition):
        # Split out the enum into its associated directions for PolarLenses
        nsDir:int = (int(dir) >> 0) & 0b1
        weDir:int = (int(dir) >> 1) & 0b1

        # Forward the direction updates
        self.nsLens.setDirection(nsDir)
        self.weLens.setDirection(weDir)

    def getDirection(self) -> InterferringLensPosition:
        # Get the individual directions
        nsDir:t.int8 = self.nsLens.lensDir[0]
        weDir:t.int8 = self.weLens.lensDir[0]

        # Bitwise or everything together
        return InterferringLensPosition(int(nsDir | (weDir << 1)))

    def __forward__(self, a:t.Tensor, b:t.Tensor) -> t.Tensor:
        # Run each channel through an individually defined lens
        la:t.Tensor = self.nsLens.forward(x=a)
        lb:t.Tensor = self.weLens.forward(x=b)

        # The sample count should be synchronus at this point as the forward() call
        #   should resignal the output of the lenses into the samples parameter defined in __init__()
        return la.unsqueeze(-1) @ lb.unsqueeze(-2)
