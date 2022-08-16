import unittest

import torch as t
from plasmatorch import *

from random import randint



class LensTest(unittest.TestCase):
    def __testBase__(self, posgen:Callable[[t.Tensor, int], t.Tensor], test:Callable[[t.Tensor, t.Tensor, t.Tensor, int], str]):
        # Generate the tensors for testing
        SIZELEN:int = randint(2, 4)
        SIZE:List[int] = [randint(1, SUPERSINGULAR_PRIMES_LH[7]) for _ in range(SIZELEN)]
        x:t.Tensor = t.randn(SIZE, dtype=DEFAULT_DTYPE)
        xc:t.Tensor = t.randn(SIZE, dtype=DEFAULT_COMPLEX_DTYPE)
        posWeights:List[t.Tensor] = [posgen(x, idx) for idx in range(SIZELEN)]

        # Randomize batch dimension
        for idx in range(1, len(posWeights)):
            addDim:bool = bool(randint(0, 1))
            if addDim: posWeights[idx] = t.stack([posWeights[idx]] * SIZE[0], dim=0)
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
        self.__testBase__(posgen=LensTest.__passthroughPos__, test=LensTest.__sizingTest__)


    def __wrapZeroPos__(x:t.Tensor, dim:int) -> t.Tensor:
        return t.tensor([-4, -3, 3, 4], dtype=DEFAULT_DTYPE) + 1e-4

    def __wrapZeroTest__(wx:t.Tensor, x:t.Tensor, pos:t.Tensor, dim:int) -> str:
        # Find the wrapping stride
        stride:int = wx.size(dim) // 2

        # Seperate out the wraps
        testX:t.Tensor = wx.transpose(dim, 0)
        wxp:t.Tensor = testX[:stride].transpose(dim, 0)
        wxn:t.Tensor = testX[stride:].transpose(dim, 0)

        # Perform consistency tests between the wrap directions
        testPass:bool = t.all(wxp.abs() <= 1e-4)
        if not testPass:
            return f'High padding failed (max: {wxp.abs().max()}):\n{wxp}'
        testPass:bool = t.all(wxn.abs() <= 1e-4)
        if not testPass:
            return f'Low padding failed (max: {wxn.abs().max()}):\n{wxn}'
        return None

    def testWrapZero(self):
        self.__testBase__(posgen=LensTest.__wrapZeroPos__, test=LensTest.__wrapZeroTest__)


    # TODO: Test decay mirroring
