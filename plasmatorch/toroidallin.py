from sizing import resignal
from .defaults import *
from .entanglement import superposition



@ts
def toroidalLinear(a:t.Tensor, b:t.Tensor, weight:t.Tensor, bias:t.Tensor, invbias:t.Tensor) -> Tuple[t.Tensor, t.Tensor]:
    """Superposes two signals on top of each other, masks them, then sums them out through both
    possible transposed views with a bias for each view. This ends up behaving a lot more like
    a matrix with all opposing edges connected, which is a toroid.

    Args:
        a (t.Tensor): The first signal to perform a toroidally mapped linear operation on.
        b (t.Tensor): The second signal to perform a toroidally mapped linear operation on.
        weight (t.Tensor): The mask to multiply unit-wise to the superposition.
        bias (t.Tensor): The bias for resultant signal a.
        invbias (t.Tensor): The bias for resultant signal b.

    Returns:
        Tuple[t.Tensor, t.Tensor]: Resultant signal (a, b).
    """
    # Initial size checking
    assert a.size()[:-1] == b.size()[:-1]
    assert b.size()[:-1] == weight.size()[:-2]
    assert weight.size()[:-2] == bias.size()[:-1]
    assert bias.size()[:-1] == invbias.size()[:-1]

    # Call superposition call for out-of-order execution capability
    abSuper = superposition(a, b) # Note: Normalized out of a superposition
    
    # Figure out if the weight should be transposed before being applied to vectors a and b
    # The default option is to transpose as that is how the torch library handles nn.Linear
    #   which this function is mostly parasitic from.
    weightTranspose:bool = True
    if abSuper.size(-2) == weight.size(-2):
        assert abSuper.size(-1) == weight.size(-1)
        if abSuper.size(-1) != abSuper.size(-2):
            weightTranspose = False
    elif abSuper.size(-1) == weight.size(-2):
        assert abSuper.size(-2) == weight.size(-1)
    else:
        assert abSuper.size()[-2:] == weight.size()[-2:]
    
    # Figure out how the biases align to the weight on the inside.
    # The standard here is to do a sum on dim=-2 after the matmul of a Linear module.
    #   I honestly don't know if that should be denoted as transposed, but I'll call it
    #   normal here.
    biasTranspose:bool = False
    if bias.size(-1) == abSuper.size(-1):
        assert invbias.size(-1) == abSuper.size(-2)
        if bias.size(-1) != invbias.size(-1):
            biasTranspose = True
    elif bias.size(-1) == abSuper.size(-2):
        assert invbias.size(-1) == abSuper.size(-1)
    else:
        assert [bias.size(-1), invbias.size(-1)] == abSuper.size()[-2:]
    
    # Apply the superposition to the input weight elementwise
    wweight = weight
    if weightTranspose:
        wweight = weight.transpose(-1, -2)
    unbiased:t.Tensor = abSuper * wweight

    # Apply the biases to the superposition
    if biasTranspose:
        unbiased = unbiased.transpose(-1, -2)
    ya:t.Tensor = unbiased.sum(-2) + bias
    yb:t.Tensor = unbiased.sum(-1) + invbias

    # Return the ya and yb signal vectors respective to the input a and b signal vectors
    return (ya, yb)
