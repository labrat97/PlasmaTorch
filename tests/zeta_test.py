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
        s = torch.randn(SIZE, dtype=DEFAULT_DTYPE).abs() + 2
        a = torch.randn_like(s).abs() + 1

        # Calculate the values to put through tests
        hxe = hzetae(s=s, a=a, res=torch.ones((1)), maxiter=SAMPLES+BLANKS)
        hxf = hzetas(s=s, a=a, res=torch.ones((1)), blankSamples=BLANKS, samples=SAMPLES, fftformat=False)
        hxfft = hzetas(s=s, a=a, res=torch.ones((1)), blankSamples=BLANKS, samples=SAMPLES, fftformat=True)

        # Test the final value of each type of hzeta evaluation against the torch.special version
        zetacontrol = torch.special.zeta(s, a)
        hxediff = (hxe - zetacontrol)
        hxfdiff = (hxf[..., -1] - zetacontrol)

        # Find the most commonly occuring error
        # (sometimes there is a really high absolute error, but that can be expected with something sort of fractalline 
        # like this)
        flatc = zetacontrol.flatten()
        flate = hxediff.flatten()
        flatf = hxfdiff.flatten()
        flate = torch.log((flate).abs()/flatc.abs()).mean(dim=-1)
        flatf = torch.log((flatf).abs()/flatc.abs()).mean(dim=-1)
        flate = torch.exp(flate)
        flatf = torch.exp(flatf)

        self.assertTrue(flate < 1e-2, msg=f"hxediff: {flate}")
        self.assertTrue(flatf < 1e-2, msg=f"hxfdiff: {flatf}")

        # Make sure the fftformat option makes the values come out with the same 
        fftmirror = torch.fft.ifft(torch.fft.fft(hxfft))
        fftdiff = hxfft - fftmirror
        self.assertTrue(torch.all(fftdiff.abs() < 1e-4), msg=f'fftdiff: {fftdiff}')

class LerchZetaTest(unittest.TestCase):
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
        l = torch.randn_like(x)

        # Test the non-batch sampled version of the function (just a single final element)
        lxf = lerche(lam=l, s=x, a=altx, maxiter=SAMPLES)
        lxb = lerche(lam=l, s=altx, a=x, maxiter=SAMPLES)
        
        # Test the batch sampled version of the function while toggling the FFT transcoding between the results
        lxffft = lerchs(lam=l, s=x, a=altx, blankSamples=BLANKS, samples=SAMPLES, fftformat=True)
        lxff = lerchs(lam=l, s=x, a=altx, blankSamples=BLANKS, samples=SAMPLES, fftformat=False)
        lxbfft = lerchs(lam=l, s=altx, a=x, blankSamples=BLANKS, samples=SAMPLES, fftformat=True)
        lxbf = lerchs(lam=l, s=altx, a=x, blankSamples=BLANKS, samples=SAMPLES, fftformat=False)

        # Size testing of the non-batch sampled version
        self.assertEqual(x.size(), lxf.size())
        self.assertEqual(x.size(), lxb.size())

        # Size testing of the batch sampled version
        self.assertEqual(x.size(), lxffft.size()[:-1])
        self.assertEqual(lxffft.size()[-1], SAMPLES)
        self.assertEqual(lxffft.size(), lxff.size())
        self.assertEqual(lxff.size(), lxbfft.size())
        self.assertEqual(lxbfft.size(), lxbf.size())

    def testValues(self):
        # Generate random sizing parameters
        SIZELEN = randint(1, 3)
        SIZESCALAR = 10
        SIZE = torch.Size((torch.randn((SIZELEN)) * SIZESCALAR).type(torch.int64).abs() + 1)
        BLANKS = randint(0, 512) + 10240
        SAMPLES = 1024

        # Generate the control tensors for later testing
        s = torch.randn(SIZE, dtype=DEFAULT_DTYPE).abs() + 2
        a = torch.randn_like(s).abs() + 1
        l = torch.zeros_like(s)

        # Calculate the values to put through tests
        hxe = hzetae(s=s, a=a, res=torch.ones((1)), maxiter=SAMPLES+BLANKS)
        hxf = hzetas(s=s, a=a, res=torch.ones((1)), blankSamples=BLANKS, samples=SAMPLES, fftformat=False)
        hxfft = hzetas(s=s, a=a, res=torch.ones((1)), blankSamples=BLANKS, samples=SAMPLES, fftformat=True)
        lxe = lerche(lam=l, s=s, a=a, res=torch.ones((1)), maxiter=SAMPLES+BLANKS)
        lxf = lerchs(lam=l, s=s, a=a, res=torch.ones((1)), blankSamples=BLANKS, samples=SAMPLES, fftformat=False)
        lxfft = lerchs(lam=l, s=s, a=a, res=torch.ones((1)), blankSamples=BLANKS, samples=SAMPLES, fftformat=True)

        # Make sure that the values that come out are the same as the hzeta function
        # This happens because the lambda value being set to zero forces the lerch transcedent
        #   to have the same value as the hzeta iteration function.
        self.assertTrue(torch.all((lxe - hxe).abs() < 1e-4))
        self.assertTrue(torch.all((lxf - hxf).abs() < 1e-4))
        self.assertTrue(torch.all((lxfft - hxfft).abs() < 1e-4))
