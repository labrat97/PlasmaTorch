import unittest
import test
import torch
from plasmatorch import *
from random import randint

class HurwitzZetaTest(unittest.TestCase):
    def testSizing(self):
        # Generate random sizing parameters
        SIZELEN = randint(1, 5)
        SIZESCALAR = randint(5, 10)
        SIZE = torch.Size((torch.randn((SIZELEN)) * SIZESCALAR).type(torch.int64).abs() + 1)
        BLANKS = randint(0, 512)
        SAMPLES = randint(10, 100)

        # Generate the control tensors for later testing
        x = torch.randn(SIZE, dtype=DEFAULT_COMPLEX_DTYPE)
        altx = torch.randn_like(x)

        # Test the non-batch sampled version of the function (just a single final element)
        hxf = hzetae(s=x, a=altx, maxiter=SAMPLES)
        hxb = hzetae(s=altx, a=x, maxiter=SAMPLES)
        
        # Test the batch sampled version of the function while toggling the FFT transcoding between the results
        hxffft = hzetas(s=x, a=altx, blankSamples=BLANKS, samples=SAMPLES, fftformat=True)
        hxff = hzetas(s=x, a=altx, blankSamples=BLANKS, samples=SAMPLES, fftformat=False)
        hxbfft = hzetas(s=altx, a=x, blankSamples=BLANKS, samples=SAMPLES, fftformat=True)
        hxbf = hzetas(s=altx, a=x, blankSamples=BLANKS, samples=SAMPLES, fftformat=False)

        # Size testing of the non-batch sampled version
        self.assertEqual(x.size(), hxf.size())
        self.assertEqual(x.size(), hxb.size())

        # Size testing of the batch sampled version
        self.assertEqual(x.size(), hxffft.size()[:-1])
        self.assertEqual(hxffft.size()[-1], SAMPLES)
        self.assertEqual(hxffft.size(), hxff.size())
        self.assertEqual(hxff.size(), hxbfft.size())
        self.assertEqual(hxbfft.size(), hxbf.size())