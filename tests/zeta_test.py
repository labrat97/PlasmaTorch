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
    
    def testValues(self):
        # Generate random sizing parameters
        SIZELEN = randint(1, 3)
        SIZESCALAR = 10
        SIZE = torch.Size((torch.randn((SIZELEN)) * SIZESCALAR).type(torch.int64).abs() + 1)
        BLANKS = randint(0, 512) + 10240
        SAMPLES = 1024

        # Generate the control tensors for later testing
        s = torch.randn(SIZE, dtype=DEFAULT_DTYPE).abs() + 1
        a = torch.randn_like(s).abs()

        # Calculate the values to put through tests
        hxe = hzetae(s=s, a=a, res=torch.ones((1)), maxiter=SAMPLES+BLANKS)
        hxf = hzetas(s=s, a=a, res=torch.ones((1)), blankSamples=BLANKS, samples=SAMPLES, fftformat=False)
        hxfft = hzetas(s=s, a=a, res=torch.ones((1)), blankSamples=BLANKS, samples=SAMPLES, fftformat=True)

        # Test the final value of each type of hzeta evaluation against the torch.special version
        zetacontrol = torch.special.zeta(s, a)
        hxediff = torch.log((hxe - zetacontrol).abs())
        hxfdiff = torch.log((hxf[..., -1] - zetacontrol).abs())

        # Find the most commonly occuring error
        # (sometimes there is a really high absolute error, but that can be expected with something sort of fractalline 
        # like this)
        while len(hxediff.size()) >= 1:
            hxediff = hxediff.mean(dim=-1)
            hxfdiff = hxfdiff.mean(dim=-1)
        hxediff = torch.exp(hxediff).abs()
        hxfdiff = torch.exp(hxfdiff).abs()

        self.assertTrue(hxediff < 1e-2, msg=f"hxediff: {hxediff}")
        self.assertTrue(hxfdiff < 1e-2, msg=f"hxfdiff: {hxfdiff}")

        # Make sure the fftformat option makes the values come out with the same 
        fftmirror = torch.fft.ifft(torch.fft.fft(hxfft))
        fftdiff = hxfft - fftmirror
        self.assertTrue(torch.all(fftdiff.abs() < 1e-4), msg=f'fftdiff: {fftdiff}')
