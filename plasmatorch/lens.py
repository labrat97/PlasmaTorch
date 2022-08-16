from .math import tau
from .defaults import *
from .sizing import weightedResample
from .distributions import irregularGauss



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
    # Quick argument checking
    assert not lens.is_complex()

    # Constants for evaluation
    TAU:t.Tensor = tau()
    ONE:t.Tensor = t.ones(1)
    ZERO:t.Tensor = t.zeros(1)
    DAMPED_SPACE:t.Tensor = t.linspace(start=-TAU, end=TAU, steps=2*x.size(dim)).unsqueeze(0)

    # Create the edges of the signal by flipping a dimension and multiplying by a decay of the signal length + 1
    transX:t.Tensor = x.transpose(dim, -1)
    flippedX:t.Tensor = transX.flip(-1)
    gaussSpread:t.Tensor = irregularGauss(x=DAMPED_SPACE, mean=ZERO, lowStd=ONE, highStd=ONE, reg=False)
    sides:t.Tensor = t.stack((flippedX * gaussSpread[0, :x.size(dim)], flippedX * gaussSpread[0, x.size(dim):]), dim=0)
    assert t.all(sides[0][..., 0].abs() <= 1e-4)
    assert t.all(sides[1][..., -1].abs() <= 1e-4)
    sides.transpose_(dim+1, -1)
    lensSpace:t.Tensor = t.cat([sides[0], x, sides[1]], dim=dim)

    # Apply the lens through a weighted resample
    return weightedResample(lensSpace, lens/3., dim=dim, ortho=True, ringCoords=True, padding='zeros')
