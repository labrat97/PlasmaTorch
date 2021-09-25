from .test import *

from ..conversions import *

class SmearTest(Test):
    def test(self, compl:bool=DEFAULT_COMPLEX):
        super(SmearTest, self).test()

        # The smear to test
        dtype = DEFAULT_COMPLEX_DTYPE if compl else DEFAULT_DTYPE
        smear = Smear(samples=TEST_FFT_SAMPLES, lowerScalar=1/16, upperScalar=1/16, dtype=dtype)

        # Test smear sizing
        x = torch.zeros((KLYBATCH,1), dtype=dtype)
        sx = smear.forward(x)

        TEST0:bool = sx.size() == (KLYBATCH, TEST_FFT_SAMPLES)
        self.log(msg=f'Smear sizing was {sx.size()}', passed=TEST0)

        # Test smear with zeros
        TEST1:bool = sx == torch.zeros_like(sx)
        self.log(msg=f'Zero smear resulted in zeros', passed=TEST1)

        # Test smear with ones to test the bounds scalars
        y = torch.ones((KLYBATCH, 1))
        sy = smear.forward(y)
        TEST2:bool = sy[:,0] == 1-(1/16)
        TEST3:bool = sy[:,-1] == 1+(1/16)
        self.log(msg=f'One smear resulted in smear domain [{sy[:,0]}, {sy[:,-1]}]', passed=TEST2 and TEST3)

class SmearResampleTest(Test):
    def test(self, compl:bool=DEFAULT_COMPLEX):
        super(SmearResampleTest, self).__init__()

        # The smear to test
        dtype = DEFAULT_COMPLEX_DTYPE if compl else DEFAULT_DTYPE
        smear = Smear(samples=TEST_FFT_SAMPLES, lowerScalar=1/16, upperScalar=1/16, dtype=dtype)

        # Test smear sizing and basis vectors
        x = torch.zeros((KLYBATCH, 1))
        s = smear.forward(x)
        sRand = torch.rand_like(s)
        
        randResize = resampleSmear(sRand, samples=int(TEST_FFT_SAMPLES*2))
        randReturnSize = resampleSmear(randResize, samples=TEST_FFT_SAMPLES)

        TEST0:bool = torch.fft.fft(sRand, n=TEST_FFT_SAMPLES, dim=-1) == torch.fft.fft(randReturnSize, n=TEST_FFT_SAMPLES, dim=-1)
        self.log(msg=f'Basis vector equivalence test.', passed=TEST0)

        # Test the expansion of the size of the smear
        randResizeReturn = resampleSmear(randReturnSize, samples=int(TEST_FFT_SAMPLES*2))

        TEST1:bool = torch.fft.fft(randResize, n=int(TEST_FFT_SAMPLES*2), dim=-1) == torch.fft.fft(randResizeReturn, n=int(TEST_FFT_SAMPLES*2), dim=-1)
        self.log(msg=f'Increased size resample basis vector equvalence test.', passed=TEST1)
