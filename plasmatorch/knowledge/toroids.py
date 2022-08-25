from ..defaults import *
from ..toroidallin import *

class ToroidalLinear(nn.Linear):
    """
    The module version of `toroidalLinear()`.
    """
    def __init__(self, in_features:int, out_features:int, device:str=None, dtype:t.dtype=None):
        """Initialize the module.

        Args:
            in_features (int): The amount of features to take in for reprojection.
            out_features (int): The amount of features that should come out of the system.
            device (str, optional): The device to use for memory storage/computation. Defaults to None.
            dtype (t.dtype, optional): The datatype to use for the parameters. Defaults to None.
        """
        super(ToroidalLinear, self).__init__(in_features=in_features, out_features=out_features, 
            bias=True, device=device, dtype=dtype)
        
        # Add the one extra parameter to evaluate toroidally
        self.invbias:nn.Parameter = nn.Parameter(t.zeros((in_features), device=device))

    def forward(self, a:t.Tensor, b:t.Tensor=None) -> Tuple[t.Tensor, t.Tensor]:
        """The default forward call of the module.

        Args:
            a (t.Tensor): The first tensor to throw into the toroidal linear system.
            b (t.Tensor, optional): The second tensor to throw into the toroidal linear system.
            If None, remap the first signal to be the appropriate dimensions. Defaults to None.

        Returns:
            Tuple[t.Tensor, t.Tensor]: Resultant signals (a, b) from the internal `toroidalLinear()` call.
        """
        # Make sure b is present for evaluation
        if b is None:
            b = resignal(a, self.weight.size(-1), dim=-1)
        
        # Call the main torchscript function previously defined in this file
        return toroidalLinear(a=a, b=b, weight=self.weight, bias=self.bias, invbias=self.invbias)
