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

        self.assertEqual(sx, torch.zeros_like(sx), msg='Zero test (real)')
        self.assertEqual(sxc, torch.zeros_like(sxc), msg='Zero test (imag)')

        # Test smear with ones to test the bounds scalars
        y = torch.ones((test.KLYBATCH, 1), dtype=DEFAULT_DTYPE)
        yc = torch.ones((test.KLYBATCH, 1), dtype=DEFAULT_COMPLEX_DTYPE)
        sy = smear.forward(y)
        syc = smearc.forward(yc)
        
        self.assertEqual(sy[:, 0], torch.ones_like(sy[:, 0], dtype=DEFAULT_DTYPE) * (1-(1/16)), msg='Lower bound test (real)')
        self.assertEqual(syc[:, 0], torch.ones_like(syc[:, 0], dtype=DEFAULT_COMPLEX_DTYPE) * (1-(1/16)), msg='Lower bounds test (imag)')
        
        self.assertEqual(sy[:, -1], torch.ones_like(sy[:, -1], dtype=DEFAULT_DTYPE) * (1+(1/16)), msg='Upper bounds test (real)')
        self.assertEqual(syc[:, -1], torch.ones_like(syc[:, -1], dtype=DEFAULT_COMPLEX_DTYPE) * (1+(1/16)), msg='Upper bounds test (imag)')

class SmearResampleTest(unittest.TestCase):
    def testEquivalence(self):
        sx, _ = getsmear(DEFAULT_DTYPE)
        sxc, _ = getsmear(DEFAULT_COMPLEX_DTYPE)
        sxRand = torch.rand_like(sx)
        sxcRand = torch.rand_like(sxc)

        # Test smear sizing and basis vectors
        randResize = resampleSmear(sxRand, samples=int(test.TEST_FFT_SAMPLES*2))
        randcResize = resampleSmear(sxcRand, samples=int(test.TEST_FFT_SAMPLES*2))
        randReturnSize = resampleSmear(randResize, samples=test.TEST_FFT_SAMPLES)
        randcReturnSize = resampleSmear(randcResize, samples=test.TEST_FFT_SAMPLES)

        self.assertEqual(torch.fft.fft(sxRand, n=test.TEST_FFT_SAMPLES, dim=-1), torch.fft.fft(randReturnSize, n=test.TEST_FFT_SAMPLES, dim=-1), msg='Forward back test (real)')
        self.assertEqual(torch.fft.fft(sxcRand, n=test.TEST_FFT_SAMPLES, dim=-1), torch.fft.fft(randcReturnSize, n=test.TEST_FFT_SAMPLES, dim=-1), msg='Forward back test (imag)')

        # Test the expansion of the size of the smear
        randResizeReturn = resampleSmear(randReturnSize, samples=int(test.TEST_FFT_SAMPLES*2))
        randcResizeReturn = resampleSmear(randcReturnSize, samples=int(test.TEST_FFT_SAMPLES*2))

        self.assertEqual(torch.fft.fft(randResize, n=int(test.TEST_FFT_SAMPLES*2), dim=-1), torch.fft.fft(randResizeReturn, n=int(test.TEST_FFT_SAMPLES*2), dim=-1), msg='Forward back forward test (real)')
        self.assertEqual(torch.fft.fft(randcResize, n=int(test.TEST_FFT_SAMPLES*2), dim=-1), torch.fft.fft(randcResizeReturn, n=int(test.TEST_FFT_SAMPLES*2), dim=-1), msg='Forward back forward test (imag)')
