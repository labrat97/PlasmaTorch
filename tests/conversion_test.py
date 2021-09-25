import unittest
import test

import torch
from plasmatorch import *

def getsmear(dtype:torch.dtype):
            # The smear to test
            smear = Smear(samples=test.TEST_FFT_SAMPLES, lowerScalar=1/16, upperScalar=1/16, dtype=dtype)
            x = torch.zeros((test.KLYBATCH,1), dtype=dtype)
            return smear.forward(x), smear

class SmearTest(unittest.TestCase):
    def testSizing(self):
        sx, _ = getsmear(DEFAULT_DTYPE)
        sxc, _ = getsmear(DEFAULT_COMPLEX_DTYPE)
        
        self.assertEqual(sx.size(), torch.Size((test.KLYBATCH, test.TEST_FFT_SAMPLES)), msg='Sizing test (real)')
        self.assertEqual(sxc.size(), torch.Size((test.KLYBATCH, test.TEST_FFT_SAMPLES)), msg='Sizing test (imag)')

    def testValues(self):
        sx, smear = getsmear(DEFAULT_DTYPE)
        sxc, smearc = getsmear(DEFAULT_COMPLEX_DTYPE)

        ZERO_TEST_REAL = torch.all(sx == torch.zeros_like(sx))
        ZERO_TEST_COMPL = torch.all(sxc == torch.zeros_like(sxc))
        self.assertTrue(ZERO_TEST_REAL, msg='Zero test (real)')
        self.assertTrue(ZERO_TEST_COMPL, msg='Zero test (imag)')

        # Test smear with ones to test the bounds scalars
        y = torch.ones((test.KLYBATCH, 1), dtype=DEFAULT_DTYPE)
        yc = torch.ones((test.KLYBATCH, 1), dtype=DEFAULT_COMPLEX_DTYPE)
        sy = smear.forward(y)
        syc = smearc.forward(yc)
        
        LOWER_TEST_REAL = torch.all(sy[:, 0] == torch.ones_like(sy[:, 0], dtype=DEFAULT_DTYPE) * (1-(1/16)))
        LOWER_TEST_COMPL = torch.all(syc[:, 0] == torch.ones_like(syc[:, 0], dtype=DEFAULT_COMPLEX_DTYPE) * (1-(1/16)))
        self.assertTrue(LOWER_TEST_REAL, msg=f'Lower bound test (real: {sy[0, 0]})')
        self.assertTrue(LOWER_TEST_COMPL, msg=f'Lower bounds test (imag: {syc[0, 0]})')
        
        UPPER_TEST_REAL = torch.all(sy[:, -1] == torch.ones_like(sy[:, -1], dtype=DEFAULT_DTYPE) * (1+(1/16)))
        UPPER_TEST_COMPL = torch.all(syc[:, -1] == torch.ones_like(syc[:, -1], dtype=DEFAULT_COMPLEX_DTYPE) * (1+(1/16)))
        self.assertTrue(UPPER_TEST_REAL, msg=f'Upper bounds test (real: {sy[0, -1]})')
        self.assertTrue(UPPER_TEST_COMPL, msg=f'Upper bounds test (imag: {syc[0, -1]})')

class SmearResampleTest(unittest.TestCase):
    def testEquivalence(self):
        EPSILON = 1e-3
        
        sx, _ = getsmear(DEFAULT_DTYPE)
        sxc, _ = getsmear(DEFAULT_COMPLEX_DTYPE)
        sxRand = torch.rand_like(sx)
        sxcRand = torch.rand_like(sxc)

        # Test smear sizing and basis vectors
        randResize = resampleSmear(sxRand, samples=int(test.TEST_FFT_SAMPLES*2))
        randcResize = resampleSmear(sxcRand, samples=int(test.TEST_FFT_SAMPLES*2))
        randReturnSize = resampleSmear(randResize, samples=test.TEST_FFT_SAMPLES)
        randcReturnSize = resampleSmear(randcResize, samples=test.TEST_FFT_SAMPLES)

        sxRandFFT = torch.fft.fft(sxRand, n=test.TEST_FFT_SAMPLES, dim=-1)
        randReturnSizeFFT = torch.fft.fft(randReturnSize, n=test.TEST_FFT_SAMPLES, dim=-1)
        sxcRandFFT = torch.fft.fft(sxcRand, n=test.TEST_FFT_SAMPLES, dim=-1)
        randcReturnSizeFFT = torch.fft.fft(randcReturnSize, n=test.TEST_FFT_SAMPLES, dim=-1)

        FORWARD_BACK_REAL = not torch.all(torch.view_as_real((sxRandFFT - randReturnSizeFFT)) >= EPSILON)
        FORWARD_BACK_COMPLEX = not torch.all(torch.view_as_real((sxcRandFFT - randcReturnSizeFFT)) >= EPSILON)
        self.assertTrue(FORWARD_BACK_REAL, msg='Forward back test (real)')
        self.assertTrue(FORWARD_BACK_COMPLEX, msg='Forward back test (imag)')

        # Test the expansion of the size of the smear
        randResizeReturn = resampleSmear(randReturnSize, samples=int(test.TEST_FFT_SAMPLES*2))
        randcResizeReturn = resampleSmear(randcReturnSize, samples=int(test.TEST_FFT_SAMPLES*2))

        randResizeFFT = torch.fft.fft(randResize, n=int(test.TEST_FFT_SAMPLES*2), dim=-1)
        randResizeReturnFFT = torch.fft.fft(randResizeReturn, n=int(test.TEST_FFT_SAMPLES*2), dim=-1)
        randcResizeFFT = torch.fft.fft(randcResize, n=int(test.TEST_FFT_SAMPLES*2), dim=-1)
        randcResizeReturnFFT = torch.fft.fft(randcResizeReturn, n=int(test.TEST_FFT_SAMPLES*2), dim=-1)

        FORWARD_BACK_FORWARD_REAL = not torch.all(torch.view_as_real(randResizeFFT - randResizeReturnFFT) >= EPSILON)
        FORWARD_BACK_FORWARD_COMPL = not torch.all(torch.view_as_real(randcResizeFFT - randcResizeReturnFFT) >= EPSILON)
        self.assertTrue(FORWARD_BACK_FORWARD_REAL, msg='Forward back forward test (real)')
        self.assertTrue(FORWARD_BACK_FORWARD_COMPL, msg='Forward back forward test (imag)')
