import unittest

import torch as t
from plasmatorch import *

from random import randint



class LensTest(unittest.TestCase):
    def __testBase__(self, posgen:Callable[[t.Tensor, int], t.Tensor], test:Callable[[t.Tensor, t.Tensor, t.Tensor, int], str]):
        # Generate the tensors for testing
        SIZELEN:int = randint(2, 4)
        SIZE:List[int] = [randint(1, SUPERSINGULAR_PRIMES_LH[2]) for _ in range(SIZELEN)]
        x:t.Tensor = t.randn(SIZE, dtype=DEFAULT_DTYPE)
        xc:t.Tensor = t.randn(SIZE, dtype=DEFAULT_COMPLEX_DTYPE)
        posWeights:List[t.Tensor] = [posgen(x, idx) for idx in range(SIZELEN)]

        # Randomize batch dimension
        for idx in range(1, len(posWeights)):
            addDim:bool = bool(randint(0, 1))
            if addDim and (posWeights[idx].dim() <= 1): posWeights[idx] = t.stack([posWeights[idx]] * SIZE[0], dim=0)
        for weight in posWeights: assert not weight.is_complex()

        # Iterate through each possible dim to make sure that the weighted resample is 
        #   performing appropriately. Specifically check the sizing here.
        for idx, posWeight in enumerate(posWeights):
            # Run each dim of the testing tensors through the function
            wx:t.Tensor = lens(x, lens=posWeight, dim=idx)
            wxc:t.Tensor = lens(xc, lens=posWeight, dim=idx)

            # Run the perscribed test
            runResults:List[str] = [test(wx, x, posWeight, idx),
            test(wxc, xc, posWeight, idx)]
            for idx, result in enumerate(runResults):
                self.assertTrue(result == None, msg=f'[{idx}]->{result}')


    def __sizingPos__(x:t.Tensor, dim:int) -> t.Tensor:
        return t.randn(randint(1, x.size(dim)*2), dtype=DEFAULT_DTYPE)

    def __sizingTest__(wx:t.Tensor, x:t.Tensor, pos:t.Tensor, dim:int) -> str:
        if wx.size(dim) != pos.size(-1):
            return f'{x.size()} @ {pos.size()} -> {wx.size()}'
        return None

    def testSizing(self):
        self.__testBase__(posgen=LensTest.__sizingPos__, test=LensTest.__sizingTest__)
    

    def __passthroughPos__(x:t.Tensor, dim:int) -> t.Tensor:
        return t.zeros(x.size(dim), dtype=DEFAULT_DTYPE)

    def __passthroughTest__(wx:t.Tensor, x:t.Tensor, pos:t.Tensor, dim:int) -> str:
        testPass:bool = t.all((wx - x).abs() <= 1e-4)
        return None if testPass else f'Passthrough failed with an error of:\n{wx - x}'

    def testPassthrough(self):
        self.__testBase__(posgen=LensTest.__passthroughPos__, test=LensTest.__passthroughTest__)


    def __wrapZeroPos__(x:t.Tensor, dim:int) -> t.Tensor:
        result:t.Tensor = t.cat([
            t.linspace(start=-7., end=-4., steps=x.size(dim)+1),
            t.linspace(start=4., end=7., steps=x.size(dim)+1)
            ], dim=-1)
        return result

    def __wrapZeroTest__(wx:t.Tensor, x:t.Tensor, pos:t.Tensor, dim:int) -> str:
        # Find the wrapping stride
        stride:int = wx.size(dim) // 2

        # Seperate out the wraps
        testX:t.Tensor = wx.transpose(dim, -1)

        # Perform consistency tests between the wrap directions
        testPass:bool = t.all(testX.abs() <= 1e-4)
        if not testPass:
            return f'Padding failed (max: {testX.abs().max()}):\n{wx}->{testX}'
        return None

    def testWrapZero(self):
        self.__testBase__(posgen=LensTest.__wrapZeroPos__, test=LensTest.__wrapZeroTest__)


    # TODO: Figure out the position functions below, ring coordinates are really weird
    def __decayMirrorPosRight__(x:t.Tensor, dim:int) -> t.Tensor:
        result = t.zeros(x.size(dim))
        phase:float = 2. - (x.size(dim) / ((3 * x.size(dim)) - 2))
        if x.size(dim) != 1: phase = phase * (((3 * x.size(dim)) - 2) - 1.) / ((3 * x.size(dim)) - 2)
        return result + phase

    def __decayMirrorPosLeft__(x:t.Tensor, dim:int) -> t.Tensor:
        result = t.zeros(x.size(dim))
        phase:float = 2. - (x.size(dim) / ((3 * x.size(dim)) - 2))
        return result - phase

    def __decayMirrorTest__(wx:t.Tensor, x:t.Tensor, pos:t.Tensor, dim:int, lR:bool) -> str:
        # Constants for evaluation
        TAU:t.Tensor = tau()
        ONE:t.Tensor = t.ones(1)
        ZERO:t.Tensor = t.zeros(1)
        DAMPED_SPACE:t.Tensor = t.linspace(start=-TAU, end=0., steps=wx.size(dim))
        if lR: DAMPED_SPACE.add_(TAU)

        # Get the gaussian for the decays
        gaussSpread:t.Tensor = irregularGauss(x=DAMPED_SPACE, mean=ZERO, lowStd=ONE, highStd=ONE, reg=False)
        twx:t.Tensor = wx.transpose(dim, -1)

        # Flip x for the reflection
        testX:t.Tensor = x.transpose(dim, -1).flip(-1) * gaussSpread

        # Test
        testResult:t.Tensor = (twx - testX).abs()
        if not t.all(testResult <= 1e-4):
            return f'Off by max:({testResult.max()}), min:({testResult.min()}), mean:({testResult.mean()})\n\n{twx}\t->\n{testX}\t==\n{twx-testX}'
        return None

    def __decayMirrorTestRight__(wx:t.Tensor, x:t.Tensor, pos:t.Tensor, dim:int) -> str:
        return LensTest.__decayMirrorTest__(wx, x, pos, dim, True)

    def __decayMirrorTestLeft__(wx:t.Tensor, x:t.Tensor, pos:t.Tensor, dim:int) -> str:
        return LensTest.__decayMirrorTest__(wx, x, pos, dim, False)

    def testDecayMirrorRight(self):
        self.__testBase__(posgen=LensTest.__decayMirrorPosRight__, test=LensTest.__decayMirrorTestRight__)
    
    def testDecayMirrorLeft(self):
        self.__testBase__(posgen=LensTest.__decayMirrorPosLeft__, test=LensTest.__decayMirrorTestLeft__)
