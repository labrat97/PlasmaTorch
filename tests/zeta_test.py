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
        BLANKS = randint(0, 10)
        SAMPLES = randint(512, 1024)

        # Generate the control tensors for later testing
        x = torch.randn(SIZE, dtype=DEFAULT_COMPLEX_DTYPE)
        single = torch.randn((1))
        altx = torch.randn_like(x)

        # Test the non-batch sampled version of the function (just a single final element)
        hxf = hzetae(s=x, a=altx, maxiter=SAMPLES)
        hxb = hzetae(s=altx, a=x, maxiter=SAMPLES)
        hxfs = hzetae(s=x, a=single, maxiter=SAMPLES)
        hxbs = hzetae(s=single, a=x, maxiter=SAMPLES)
        
        # Test the batch sampled version of the function while toggling the FFT transcoding between the results
        hxffft = hzetas(s=x, a=altx, blankSamples=BLANKS, samples=SAMPLES, fftFormat=True)
        hxff = hzetas(s=x, a=altx, blankSamples=BLANKS, samples=SAMPLES, fftFormat=False)
        hxbfft = hzetas(s=altx, a=x, blankSamples=BLANKS, samples=SAMPLES, fftFormat=True)
        hxbf = hzetas(s=altx, a=x, blankSamples=BLANKS, samples=SAMPLES, fftFormat=False)
        hxfsfft = hzetas(s=x, a=single, blankSamples=BLANKS, samples=SAMPLES, fftFormat=True)
        hxfsf = hzetas(s=x, a=single, blankSamples=BLANKS, samples=SAMPLES, fftFormat=False)
        hxbsfft = hzetas(s=single, a=x, blankSamples=BLANKS, samples=SAMPLES, fftFormat=True)
        hxbsf = hzetas(s=single, a=x, blankSamples=BLANKS, samples=SAMPLES, fftFormat=False)

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
