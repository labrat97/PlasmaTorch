import unittest
import test

import torch
from plasmatorch import *

class IrregularGaussTest(unittest.TestCase):
    SIZE = (97, 11, 13, 128)

    def testSizing(self):
        # Seeding tensors, no complex support
        x = torch.randn(self.SIZE, dtype=DEFAULT_DTYPE)
        xmean = torch.randn_like(x)
        xlow = torch.randn_like(x)
        xhigh = torch.abs(torch.randn_like(x)) + xlow
        xmean1 = torch.randn((1), dtype=DEFAULT_DTYPE)
        xlow1 = torch.randn_like(xmean1)
        xhigh1 = torch.abs(torch.randn_like(xmean1)) + xlow1

        # Calculate
        gauss = irregularGauss(x=x, mean=xmean, lowStd=xlow, highStd=xhigh)
        gauss1 = irregularGauss(x=x, mean=xmean1, lowStd=xlow1, highStd=xhigh1)

        # Test the sizing to make sure that broadcasting is working properly
        self.assertEqual(x.size(), gauss.size(), msg=f'{x.size()} != {gauss.size()}')
        self.assertEqual(x.size(), gauss1.size(), msg=f'{x.size()} != {gauss1.size()}')

    def testValues(self):
        # Seeding tensors, no complex support
        x = torch.randn(self.SIZE, dtype=DEFAULT_DTYPE)
        xmean = torch.randn_like(x)
        xlow = torch.randn_like(x)
        xhigh = torch.randn_like(x) + xlow

        # Calculate
        gauss = irregularGauss(x=x, mean=xmean, lowStd=xlow, highStd=xhigh)

        # Calculate control values
        normallow = torch.distributions.normal.Normal(xmean, torch.exp(xlow))
        normalhigh = torch.distributions.normal.Normal(xmean, torch.exp(xhigh))
        reflow = normallow.cdf(x)
        refhigh = normalhigh.cdf(x)

        # Test the values to make sure a normal guassian is happening on
        # either side of the mean
        xbottom = (x <= xmean)
        xtop = (x >= xmean)
        self.assertTrue(torch.all(
            torch.logical_or(torch.logical_not(xbottom), (gauss - reflow < 0.0001))
        ), msg=f'Lower curve off by approx: {torch.mean(gauss-reflow)}')
        self.assertTrue(torch.all(
            torch.logical_or(torch.logical_not(xtop), (gauss - refhigh < 0.0001))
        ), msg=f'Upper curve off by approx: {torch.mean(gauss-refhigh)}')

class LinearGaussTest(unittest.TestCase):
    SIZE = (128, 11, 13, 97)

    def testSizing(self):
        # Seeding tensors
        x = torch.randn(self.SIZE, dtype=DEFAULT_DTYPE)
        xc = torch.randn(self.SIZE, dtype=DEFAULT_COMPLEX_DTYPE)

        # Calculate the result values
        gauss1 = LinearGauss(1, dtype=DEFAULT_DTYPE)
        gf1 = gauss1.forward(x)
        gauss = LinearGauss(self.SIZE[-2], dtype=DEFAULT_DTYPE)
        gf = gauss.forward(x)
        gaussc1 = LinearGauss(1, dtype=DEFAULT_COMPLEX_DTYPE)
        gfc1 = gaussc1.forward(xc)
        gaussc = LinearGauss(self.SIZE[-2], dtype=DEFAULT_COMPLEX_DTYPE)
        gfc = gaussc.forward(xc)
        cross = gauss.forward(xc)
        crossc = gaussc.forward(x)

        # Verify that the resulting sizes are correct
        self.assertEqual(x.size(), gf.size(), msg=f'{x.size()} != {gf.size()}')
        self.assertEqual(x.size(), gf1.size(), msg=f'{x.size()} != {gf1.size()}')
        self.assertEqual(xc.size(), gfc.size(), msg=f'{xc.size()} != {gfc.size()}')
        self.assertEqual(xc.size(), gfc1.size(), msg=f'{xc.size()} != {gfc1.size()}')
        self.assertEqual(xc.size(), cross.size(), msg=f'{xc.size()} != {cross.size()}')
        self.assertEqual(x.size(), crossc.size(), msg=f'{x.size()} != {crossc.size()}')

    def testValues(self):
        # Seeding tensor
        x = torch.randn(self.SIZE, dtype=DEFAULT_DTYPE)
        xc = torch.randn(self.SIZE, dtype=DEFAULT_COMPLEX_DTYPE)

        # Construct the modules to test along with alternate parameters
        gauss1 = LinearGauss(1, dtype=DEFAULT_DTYPE)
        altMean1 = nn.Parameter(torch.randn_like(gauss1.mean))
        altLow1 = nn.Parameter(torch.randn_like(gauss1.lowStd))
        altHigh1 = nn.Parameter(torch.randn_like(gauss1.highStd))
        gauss = LinearGauss(self.SIZE[-2], dtype=DEFAULT_DTYPE)
        altMean = nn.Parameter(torch.randn_like(gauss.mean))
        altLow = nn.Parameter(torch.randn_like(gauss.lowStd))
        altHigh = nn.Parameter(torch.randn_like(gauss.highStd))

        # Zero formed gaussian as a control variable
        zgauss = irregularGauss(x=x, mean=torch.zeros(1), lowStd=torch.zeros(1), highStd=torch.zeros(1))
        zgaussr = irregularGauss(x=xc.real, mean=torch.zeros(1), lowStd=torch.zeros(1), highStd=torch.zeros(1))
        zgaussi = irregularGauss(x=xc.imag, mean=torch.zeros(1), lowStd=torch.zeros(1), highStd=torch.zeros(1))
        zgaussc = torch.view_as_complex(torch.stack((zgaussr, zgaussi), dim=-1))

        # Test against zero case
        self.assertTrue(torch.all(gauss1.forward(x) == zgauss))
        self.assertTrue(torch.all(gauss.forward(x) == zgauss))
        self.assertTrue(torch.all(gauss1.forward(xc) == zgaussc))
        self.assertTrue(torch.all(gauss.forward(xc) == zgaussc))

        # Move over parameters for next test
        gauss1.mean = altMean1
        gauss1.lowStd = altLow1
        gauss1.highStd = altHigh1
        gauss.mean = altMean
        gauss.lowStd = altLow
        gauss.highStd = altHigh

        # Test to make sure the values are distributing properly
        self.assertTrue(torch.all(gauss1.forward(x) == irregularGauss(x=x, \
            mean=altMean1, lowStd=altLow1, highStd=altHigh1)))
        rgaussr = irregularGauss(x=xc.real, mean=altMean1, lowStd=altLow1, highStd=altHigh1)
        rgaussi = irregularGauss(x=xc.imag, mean=altMean1, lowStd=altLow1, highStd=altHigh1)
        rgaussc = torch.view_as_complex(torch.stack((rgaussr, rgaussi), dim=-1))
        self.assertTrue(torch.all(gauss1.forward(xc) == rgaussc))

        bigGauss = gauss.forward(x)
        bigGaussc = gauss.forward(xc)
        self.assertEqual(bigGauss.size(), bigGaussc.size())
        for idx in range(self.SIZE[-2]):
            testGauss = irregularGauss(x=x[:,:,idx,:], mean=altMean[idx], lowStd=altLow[idx], highStd=altHigh[idx])
            self.assertTrue(torch.all(testGauss == bigGauss[:,:,idx,:]), msg=f'Values not properly distributing to channels in position -2.')

            testGaussr = irregularGauss(x=xc.real[:,:,idx,:], mean=altMean[idx], lowStd=altLow[idx], highStd=altHigh[idx])
            testGaussi = irregularGauss(x=xc.imag[:,:,idx,:], mean=altMean[idx], lowStd=altLow[idx], highStd=altHigh[idx])
            testGaussc = torch.view_as_complex(torch.stack((testGaussr, testGaussi), dim=-1))
            self.assertTrue(torch.all(testGaussc == bigGaussc[:,:,idx,:]))

    def testValuesComplex(self):
        # Seeding tensor
        xc = torch.randn(self.SIZE, dtype=DEFAULT_COMPLEX_DTYPE)

        # Construct the modules to test along with alternate parameters
        gaussc1 = LinearGauss(1, dtype=DEFAULT_COMPLEX_DTYPE)
        altMeanc1 = nn.Parameter(torch.randn_like(gaussc1.mean))
        altLowc1 = nn.Parameter(torch.randn_like(gaussc1.lowStd))
        altHighc1 = nn.Parameter(torch.randn_like(gaussc1.highStd))
        gaussc = LinearGauss(self.SIZE[-2], dtype=DEFAULT_COMPLEX_DTYPE)
        altMeanc = nn.Parameter(torch.randn_like(gaussc.mean))
        altLowc = nn.Parameter(torch.randn_like(gaussc.lowStd))
        altHighc = nn.Parameter(torch.randn_like(gaussc.highStd))

        # Zero formed gaussian as a control variable
        zc1 = torch.zeros((1), dtype=DEFAULT_COMPLEX_DTYPE)
        zgauss = irregularGauss(x=xc.real, mean=zc1.real, lowStd=zc1.real, highStd=zc1.real)
        zgaussi = irregularGauss(x=xc.imag, mean=zc1.imag, lowStd=zc1.imag, highStd=zc1.imag)
        zgaussz = irregularGauss(x=torch.zeros_like(xc.real), mean=zc1.imag, lowStd=zc1.imag, highStd=zc1.imag)
        zgaussc = torch.view_as_complex(torch.stack((zgauss, zgaussi), dim=-1))
        zgausscz = torch.view_as_complex(torch.stack((zgauss, zgaussz), dim=-1))

        # Test against the zero case
        self.assertTrue(torch.all(zgaussc == gaussc1.forward(xc)))
        self.assertTrue(torch.all(zgausscz == gaussc1.forward(xc.real)))
        self.assertTrue(torch.all(zgaussc == gaussc.forward(xc)))
        self.assertTrue(torch.all(zgausscz == gaussc.forward(xc.real)))

        # Move over parameters for next test
        gaussc1.mean = altMeanc1
        gaussc1.lowStd = altLowc1
        gaussc1.highStd = altHighc1
        gaussc.mean = altMeanc
        gaussc.lowStd = altLowc
        gaussc.highStd = altHighc

        # Test to make sure that the values are distributing properly
        realResult1r = irregularGauss(x=xc.real, mean=altMeanc1.real, lowStd=altLowc1.real, highStd=altHighc1.real)
        realResult1i = irregularGauss(x=torch.zeros_like(xc.imag), mean=altMeanc1.imag, lowStd=altLowc1.imag, highStd=altHighc1.imag)
        realResult1c = torch.view_as_complex(torch.stack((realResult1r, realResult1i), dim=-1))
        self.assertTrue(torch.all(gaussc1.forward(xc.real) == realResult1c))

        result1r = realResult1r
        result1i = irregularGauss(x=xc.imag, mean=altMeanc1.imag, lowStd=altLowc1.imag, highStd=altHighc1.imag)
        result1c = torch.view_as_complex(torch.stack((result1r, result1i), dim=-1))
        self.assertTrue(torch.all(gaussc1.forward(xc) == result1c))
