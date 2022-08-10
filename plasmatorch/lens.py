from .defaults import *
from .sizing import *



@ts
def lens(x:t.Tensor, lens:t.Tensor, dim:int=-1) -> t.Tensor:
    """Perform a weighted resample with circular padding and extra padding for the interpolation
    around the edges.

    Args:
        x (t.Tensor): The signal to put through the lens.
        lens (t.Tensor): The lens signal.
        dim (int, optional): The dimension to perform the operation on. Defaults to -1.

    Returns:
        t.Tensor: The signal after refraction through the lens.
    """
    # Apply the lens through a weighted resample
    return weightedResample(x, lens, dim=dim, ortho=True, ringCoords=True, padding='reflection')
