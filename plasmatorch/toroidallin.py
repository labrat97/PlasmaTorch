from .__defimp__ import *
from .entanglement import superposition
from .math import hmean



@ts
def toroidalLinear(a:t.Tensor, b:t.Tensor, weight:t.Tensor, bias:t.Tensor, invbias:t.Tensor) -> Tuple[t.Tensor, t.Tensor]:
    """Superposes two signals on top of each other, masks them, then sums them out through both
    possible transposed views with a bias for each view. This ends up behaving a lot more like
    a matrix with all opposing edges connected, which is a toroid.

    Args:
        a (t.Tensor): The first signal to perform a toroidally mapped linear operation on.
        b (t.Tensor): The second signal to perform a toroidally mapped linear operation on.
        weight (t.Tensor): The mask to multiply unit-wise (after a transpose) to the superposition.
        bias (t.Tensor): The bias for resultant signal a.
        invbias (t.Tensor): The bias for resultant signal b.

    Returns:
        Tuple[t.Tensor, t.Tensor]: Resultant signal (a, b).
    """
    # Initial size checking
    assert a.size()[:-1] == b.size()[:-1]
    assert b.size()[:-1] == weight.size()[:-2]
    assert a.size(-1) == weight.size(-1)
    assert b.size(-1) == weight.size(-2)
    assert weight.size()[:-2] == bias.size()[:-1]
    assert bias.size()[:-1] == invbias.size()[:-1]
    assert bias.size(-1) == weight.size(-1)
    assert invbias.size(-1) == weight.size(-2)

    # Call superposition call for out-of-order execution capability
    abSuper = superposition(a, b)
    
    # Apply the superposition to the input weight elementwise
    unbiased:t.Tensor = abSuper * weight.transpose(-1, -2)

    # Apply the biases to the superposition
    ya:t.Tensor = hmean(unbiased, dim=-1) + bias
    yb:t.Tensor = hmean(unbiased, dim=-2) + invbias

    # Return the ya and yb signal vectors respective to the input a and b signal vectors
    return (ya, yb)
