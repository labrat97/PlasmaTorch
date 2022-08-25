import unittest

import torch as t
from plasmatorch import *

from random import randint



class ToroidalLinearTest(unittest.TestCase):
    def __testBase__(self, valgen:Callable[[List[int], t.dtype], t.Tensor], test:Callable[[Tuple[t.Tensor], List[t.Tensor]], str]):
        # Generate random sizing parameters
        SIZELEN:int = randint(2, 4)
        SIZEA:List[int] = [randint(1, SUPERSINGULAR_PRIMES_LH[7]) for _ in range(SIZELEN)]
        SIZEB:List[int] = SIZEA[:-1] + [randint(1, SUPERSINGULAR_PRIMES_LH[7])]
        SIZEW:List[int] = SIZEB + [SIZEA[-1]]

        # Generate the tensors to run the tests with
        a:t.Tensor = valgen(SIZEA, DEFAULT_DTYPE)
        ac:t.Tensor = valgen(SIZEA, DEFAULT_COMPLEX_DTYPE)
        b:t.Tensor = valgen(SIZEB, DEFAULT_DTYPE)
        bc:t.Tensor = valgen(SIZEB, DEFAULT_COMPLEX_DTYPE)
        w:t.Tensor = valgen(SIZEW, DEFAULT_DTYPE)
        wc:t.Tensor = valgen(SIZEW, DEFAULT_COMPLEX_DTYPE)
        bias:t.Tensor = valgen(SIZEA, DEFAULT_DTYPE)
        biasc:t.Tensor = valgen(SIZEA, DEFAULT_COMPLEX_DTYPE)
        invbias:t.Tensor = valgen(SIZEB, DEFAULT_DTYPE)
        invbiasc:t.Tensor = valgen(SIZEB, DEFAULT_COMPLEX_DTYPE)

        # Perform the computation
        regularToroid:Tuple[t.Tensor] = toroidalLinear(a=a, b=b, weight=w, bias=bias, invbias=invbias)
        complexToroid:Tuple[t.Tensor] = toroidalLinear(a=ac, b=bc, weight=wc, bias=biasc, invbias=invbiasc)

        # Check the results of the tests
        regularResult:str = test(regularToroid, [a, b, w, bias, invbias])
        complexResult:str = test(complexToroid, [ac, bc, wc, biasc, invbiasc])
        self.assertTrue(regularResult is None, msg=f'Regular result failure:\n{regularResult}')
        self.assertTrue(complexResult is None, msg=f'Complex result failure:\n{complexResult}')


    def __sizingVals__(size:List[int], dtype:t.dtype) -> t.Tensor:
        return t.randn(size=size, dtype=dtype)

    def __sizingTest__(result:Tuple[t.Tensor], controls:List[t.Tensor]) -> str:
        tasize:t.Size = result[0].size()
        tbsize:t.Size = result[1].size()
        asize:t.Size = controls[0].size()
        bsize:t.Size = controls[1].size()

        if tasize != asize:
            return f'{asize} -!=> {tasize}'
        if tbsize != bsize:
            return f'{bsize} -!=> {tbsize}'
        return None

    def testSizing(self):
        self.__testBase__(valgen=ToroidalLinearTest.__sizingVals__, test=ToroidalLinearTest.__sizingTest__)


    def __valueVals__(size:List[int], dtype:t.dtype) -> t.Tensor:
        return t.randn(size=size, dtype=dtype)

    def __valueTest__(result:Tuple[t.Tensor], controls:List[t.Tensor]) -> str:
        maxval = t.cat(controls[:2], dim=-1).abs().max()

        unbiasedRes:Tuple[t.Tensor] = (result[0] - controls[3], result[1] - controls[4])
        maxres = t.cat(unbiasedRes, dim=-1).abs().max()

        # Can't check the lowest value due to hmean() call
        maxTestRes:bool = (maxres - maxval) <= 1e-4

        if not maxTestRes:
            return f'Maximum range failed, unbiased result has a higher peak amplitude than the input.\n{maxres} > {maxval}'
        return None

    def testValues(self):
        self.__testBase__(valgen=ToroidalLinearTest.__valueVals__, test=ToroidalLinearTest.__valueTest__)
