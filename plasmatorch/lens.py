from .defaults import *
from .sizing import *



@ts
def lens(x:t.Tensor, lens:t.Tensor, padding:int=DEFAULT_SIGNAL_LENS_PADDING, dim:int=-1) -> t.Tensor:
    """Perform a weighted resample with circular padding and extra padding for the interpolation
    around the edges.

    Args:
        x (t.Tensor): The signal to put through the lens.
        lens (t.Tensor): The lens signal.
        padding (int, optional): The amount of padding around the signal to add circularly before sampling. Defaults to DEFAULT_SIGNAL_LENS_PADDING.
        dim (int, optional): The dimension to perform the operation on. Defaults to -1.

    Returns:
        t.Tensor: The signal after refraction through the lens.
    """
    # Cast the lens to having something of circular padding with aligned corners
    lensSquish:t.Tensor = (lens + 1.) / 2.
    lensCast:t.Tensor = lensSquish.to(t.int64, non_blocking=True)
    lensClip:t.Tensor = (lens.abs() > 1).to(t.int64, non_blocking=True)
    lensSign:t.Tensor = lens.sign()

    # Apply the clipping
    clippedIntrinsics:t.Tensor = ((lensSquish - (lensCast * lensClip * lensSign)) * 2.) - 1.

    # Add padding to the input signal to allow viewing outside of the maximum signal representation
    xpad:t.Tensor = paddim(x=x, lowpad=padding, highpad=padding, dim=dim, mode='circular')

    # Modify the lens intrinsics to be bound within the unpadded signal in the padded signal
    lensScalar:float = x.size(dim) / (x.size(dim) + (2. * padding))
    padIntrinsics:t.Tensor = lensScalar * clippedIntrinsics

    # Apply the lens through a weighted resample
    return weightedResample(xpad, padIntrinsics, dim=dim)
