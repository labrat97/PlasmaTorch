from .__defimp__ import *
from .sizing import weightedResample
from .distributions import linspace, irregularGauss



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
    TAU:t.Tensor = tau(device=x.device)
    ONE:t.Tensor = t.ones(1, device=x.device)
    ZERO:t.Tensor = t.zeros(1, device=x.device)
    DAMPED_SPACE:t.Tensor = linspace(start=-TAU, end=TAU, steps=2*x.size(dim), device=x.device).unsqueeze(0)

    # Create the edges of the signal by flipping a dimension and multiplying by a decay of the signal length
    wx:t.Tensor = x.transpose(dim, -1).flip(-1)
    gaussSpread:t.Tensor = irregularGauss(x=DAMPED_SPACE, mean=ZERO, lowStd=ONE, highStd=ONE, reg=False)
    sides:t.Tensor = t.stack((wx * gaussSpread[..., :x.size(dim)], wx * gaussSpread[..., x.size(dim):]), dim=0)
    sides.transpose_(dim+1, -1)
    lensSpace:t.Tensor = t.cat([sides[0], x, sides[1]], dim=dim)
    sizeGain:float = x.size(dim) / float(lensSpace.size(dim))

    # Create a new ortholut for the lens system
    orthoscalar:float = float(x.size(dim) - 1) / x.size(dim)
    ortholut:t.Tensor = linspace(start=-orthoscalar*sizeGain, end=orthoscalar*sizeGain, steps=lens.size(-1), device=x.device)
    ortholens:t.Tensor = ((lens * sizeGain) + ortholut)

    # Apply the lens through a weighted resample
    return weightedResample(lensSpace, ortholens, dim=dim, ortho=False, ringCoords=True, padding='zeros')
