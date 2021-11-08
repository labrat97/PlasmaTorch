import unittest
import torch
from plasmatorch import *
from random import randint

class HurwitzTests(unittest.TestCase):
    def testSizing(self):
        # Generate random sizing parameters
        SIZELEN = randint(1, 5)
        SIZESCALAR = randint(5, 10)
        SIZE = torch.Size(torch.randn((SIZELEN), dtype=torch.int64) * SIZESCALAR)
        BLANKS = randint(0, 10)
        SAMPLES = randint(512, 1024)

        # Generate the control tensors for later testing
        x = torch.randn(SIZE, dtype=DEFAULT_COMPLEX_DTYPE)
        single = torch.randn((1))
        altx = torch.randn_like(x)

        # Test the non-batch sampled version of the function (just a single final element)
        hxf = hzeta(s=x, a=altx, maxiter=SAMPLES)
        hxb = hzeta(s=altx, a=x, maxiter=SAMPLES)
        hxfs = hzeta(s=x, a=single, maxiter=SAMPLES)
        hxbs = hzeta(s=single, a=x, maxiter=SAMPLES)
        
        # Test the batch sampled version of the function while toggling the FFT transcoding between the results
        hxffft = hzeta(s=x, a=altx, blankSamples=BLANKS, samples=SAMPLES, fftFormat=True)
        hxff = hzeta(s=x, a=altx, blankSamples=BLANKS, samples=SAMPLES, fftFormat=False)
        hxbfft = hzeta(s=altx, a=x, blankSamples=BLANKS, samples=SAMPLES, fftFormat=True)
        hxbf = hzeta(s=altx, a=x, blankSamples=BLANKS, samples=SAMPLES, fftFormat=False)
        hxfsfft = hzeta(s=x, a=single, blankSamples=BLANKS, samples=SAMPLES, fftFormat=True)
        hxfsf = hzeta(s=x, a=single, blankSamples=BLANKS, samples=SAMPLES, fftFormat=False)
        hxbsfft = hzeta(s=single, a=x, blankSamples=BLANKS, samples=SAMPLES, fftFormat=True)
        hxbsf = hzeta(s=single, a=x, blankSamples=BLANKS, samples=SAMPLES, fftFormat=False)

        # Size testing of the non-batch sampled version
        self.assertEqual(x.size(), hxf.size())
        self.assertEqual(x.size(), hxb.size())
        self.assertEqual(x.size(), hxfs.size())
        self.assertEqual(x.size(), hxbs.size())

        # Size testing of the batch sampled version
        self.assertEqual(x.size(), hxffft.size()[:-1])
        self.assertEqual(hxffft.size()[-1], SAMPLES)
        self.assertEqual(hxffft.size(), hxff.size())
        self.assertEqual(hxff.size(), hxbfft.size())
        self.assertEqual(hxbfft.size(), hxbf.size())
        self.assertEqual(hxbf.size(), hxfsfft.size())
        self.assertEqual(hxfsfft.size(), hxfsf.size())
        self.assertEqual(hxfsf.size(), hxbsfft.size())
        self.assertEqual(hxbsfft.size(), hxbsf.size())
