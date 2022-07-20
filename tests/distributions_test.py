import unittest

import torch as t
from plasmatorch import *



class IrregularGaussTest(unittest.TestCase):
    SIZE = (11, 11, 13, 128)

    def testSizing(self):
        # Seeding tensors, no complex support
        x = t.randn(self.SIZE, dtype=DEFAULT_DTYPE)
        xmean = t.randn_like(x)
        xlow = t.randn_like(x)
        xhigh = t.abs(t.randn_like(x)) + xlow
        xmean1 = t.randn((1), dtype=DEFAULT_DTYPE)
        xlow1 = t.randn_like(xmean1)
        xhigh1 = t.abs(t.randn_like(xmean1)) + xlow1

        # Calculate
        gauss = irregularGauss(x=x, mean=xmean, lowStd=xlow, highStd=xhigh)
        gauss1 = irregularGauss(x=x, mean=xmean1, lowStd=xlow1, highStd=xhigh1)

        # Test the sizing to make sure that broadcasting is working properly
        self.assertEqual(x.size(), gauss.size(), msg=f'{x.size()} != {gauss.size()}')
        self.assertEqual(x.size(), gauss1.size(), msg=f'{x.size()} != {gauss1.size()}')

    def testValues(self):
        # Seeding tensors, no complex support
        x = t.randn(self.SIZE, dtype=DEFAULT_DTYPE)
        xmean = t.randn_like(x)
        xlow = t.randn_like(x)
        xhigh = t.randn_like(x).abs() + xlow

        # Calculate
        gauss = irregularGauss(x=x, mean=xmean, lowStd=xlow, highStd=xhigh, reg=False)
        gaussreg = irregularGauss(x=x, mean=xmean, lowStd=xlow, highStd=xhigh, reg=True)

        # Calculate control values
        softp = t.nn.Softplus(beta=1.618033988749895, threshold=100)
        reglow = 1. / (softp(xlow) * t.sqrt(t.ones_like(xlow) * 2 * 3.14159265358979))
        reflow = t.exp(-0.5 * t.pow((x - xmean) / softp(xlow), 2.))
        reghigh = 1. / (softp(xhigh) * t.sqrt(t.ones_like(xhigh) * 2 * 3.14159265358979))
        refhigh = t.exp(-0.5 * t.pow((x - xmean) / softp(xhigh), 2.))

        # Test the values to make sure a normal guassian is happening on
        # either side of the mean
        xbottom = (x <= xmean)
        xtop = (x >= xmean)
        self.assertTrue(t.all(
            t.logical_or(t.logical_not(xbottom), (gauss - reflow < 1e-4))
        ), msg=f'Lower curve off by approx: {t.mean(gauss-reflow)}')
        self.assertTrue(t.all(
            t.logical_or(t.logical_not(xtop), (gauss - refhigh < 1e-4))
        ), msg=f'Upper curve off by approx: {t.mean(gauss-refhigh)}')
        self.assertTrue(t.all(
            t.logical_or(t.logical_not(xbottom), (gaussreg - (reglow * reflow) < 1e-4))
        ), msg=f'Lower curve off by approx: {t.mean(gaussreg-(reglow*reflow))}')
        self.assertTrue(t.all(
            t.logical_or(t.logical_not(xtop), (gaussreg - (reghigh * refhigh) < 1e-4))
        ), msg=f'Upper curve off by approx: {t.mean(gaussreg-(reghigh*refhigh))}')



class LinearGaussTest(unittest.TestCase):
    SIZE = (128, 11, 13, 97)

    def testSizing(self):
        # Seeding tensors
        x = t.randn(self.SIZE, dtype=DEFAULT_DTYPE)
        xc = t.randn(self.SIZE, dtype=DEFAULT_COMPLEX_DTYPE)

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
        x = t.randn(self.SIZE, dtype=DEFAULT_DTYPE)
        xc = t.randn(self.SIZE, dtype=DEFAULT_COMPLEX_DTYPE)

        # Construct the modules to test along with alternate parameters
        gauss1 = LinearGauss(1, dtype=DEFAULT_DTYPE)
        altMean1 = nn.Parameter(t.randn_like(gauss1.mean))
        altLow1 = nn.Parameter(t.randn_like(gauss1.lowStd))
        altHigh1 = nn.Parameter(t.randn_like(gauss1.highStd))
        gauss = LinearGauss(self.SIZE[-2], dtype=DEFAULT_DTYPE)
        altMean = nn.Parameter(t.randn_like(gauss.mean))
        altLow = nn.Parameter(t.randn_like(gauss.lowStd))
        altHigh = nn.Parameter(t.randn_like(gauss.highStd))

        # Zero formed gaussian as a control variable
        zgauss = irregularGauss(x=x, mean=t.zeros(1), lowStd=t.zeros(1), highStd=t.zeros(1))
        zgaussr = irregularGauss(x=xc.real, mean=t.zeros(1), lowStd=t.zeros(1), highStd=t.zeros(1))
        zgaussi = irregularGauss(x=xc.imag, mean=t.zeros(1), lowStd=t.zeros(1), highStd=t.zeros(1))
        zgaussc = t.view_as_complex(t.stack((zgaussr, zgaussi), dim=-1))

        # Test against zero case
        self.assertTrue(t.all(gauss1.forward(x) - zgauss < 1e-4))
        self.assertTrue(t.all(gauss.forward(x) - zgauss < 1e-4))
        self.assertTrue(t.all(gauss1.forward(xc).abs() - zgaussc.abs() < 1e-4))
        self.assertTrue(t.all(gauss.forward(xc).abs() - zgaussc.abs() < 1e-4))

        # Move over parameters for next test
        gauss1.mean = altMean1
        gauss1.lowStd = altLow1
        gauss1.highStd = altHigh1
        gauss.mean = altMean
        gauss.lowStd = altLow
        gauss.highStd = altHigh

        # Test to make sure the values are distributing properly
        self.assertTrue(t.all(gauss1.forward(x) - irregularGauss(x=x, \
            mean=altMean1, lowStd=altLow1, highStd=altHigh1) < 1e-4))
        rgaussr = irregularGauss(x=xc.real, mean=altMean1, lowStd=altLow1, highStd=altHigh1)
        rgaussi = irregularGauss(x=xc.imag, mean=altMean1, lowStd=altLow1, highStd=altHigh1)
        rgaussc = t.view_as_complex(t.stack((rgaussr, rgaussi), dim=-1))
        self.assertTrue(t.all(gauss1.forward(xc).abs() - rgaussc.abs() < 1e-4))

        bigGauss = gauss.forward(x)
        bigGaussc = gauss.forward(xc)
        self.assertEqual(bigGauss.size(), bigGaussc.size())
        for idx in range(self.SIZE[-2]):
            testGauss = irregularGauss(x=x[:,:,idx,:], mean=altMean[idx], lowStd=altLow[idx], highStd=altHigh[idx])
            self.assertTrue(t.all(t.abs(testGauss - bigGauss[:,:,idx,:]) < 1e-4), msg=f'Values not properly distributing to channels in position -2.')

            testGaussr = irregularGauss(x=xc.real[:,:,idx,:], mean=altMean[idx], lowStd=altLow[idx], highStd=altHigh[idx])
            testGaussi = irregularGauss(x=xc.imag[:,:,idx,:], mean=altMean[idx], lowStd=altLow[idx], highStd=altHigh[idx])
            testGaussc = t.view_as_complex(t.stack((testGaussr, testGaussi), dim=-1))
            self.assertTrue(t.all(t.abs(testGaussc - bigGaussc[:,:,idx,:]) < 1e-4))

    def testValuesComplex(self):
        # Seeding tensor
        xc = t.randn(self.SIZE, dtype=DEFAULT_COMPLEX_DTYPE)

        # Construct the modules to test along with alternate parameters
        gaussc1 = LinearGauss(1, dtype=DEFAULT_COMPLEX_DTYPE)
        altMeanc1 = nn.Parameter(t.randn_like(gaussc1.mean))
        altLowc1 = nn.Parameter(t.randn_like(gaussc1.lowStd))
        altHighc1 = nn.Parameter(t.randn_like(gaussc1.highStd))
        gaussc = LinearGauss(self.SIZE[-2], dtype=DEFAULT_COMPLEX_DTYPE)
        altMeanc = nn.Parameter(t.randn_like(gaussc.mean))
        altLowc = nn.Parameter(t.randn_like(gaussc.lowStd))
        altHighc = nn.Parameter(t.randn_like(gaussc.highStd))

        # Zero formed gaussian as a control variable
        zc1 = t.zeros((1), dtype=DEFAULT_COMPLEX_DTYPE)
        zgauss = irregularGauss(x=xc.real, mean=zc1.real, lowStd=zc1.real, highStd=zc1.real)
        zgaussi = irregularGauss(x=xc.imag, mean=zc1.imag, lowStd=zc1.imag, highStd=zc1.imag)
        zgaussz = irregularGauss(x=t.zeros_like(xc.real), mean=zc1.imag, lowStd=zc1.imag, highStd=zc1.imag)
        zgaussc = t.view_as_complex(t.stack((zgauss, zgaussi), dim=-1))
        zgausscz = t.view_as_complex(t.stack((zgauss, zgaussz), dim=-1))

        # Test against the zero case
        self.assertTrue(t.all((zgaussc - gaussc1.forward(xc)).abs() < 1e-4))
        self.assertTrue(t.all((zgausscz - gaussc1.forward(xc.real)).abs() < 1e-4))
        self.assertTrue(t.all((zgaussc - gaussc.forward(xc)).abs() < 1e-4))
        self.assertTrue(t.all((zgausscz - gaussc.forward(xc.real)).abs() < 1e-4))

        # Move over parameters for next test
        gaussc1.mean = altMeanc1
        gaussc1.lowStd = altLowc1
        gaussc1.highStd = altHighc1
        gaussc.mean = altMeanc
        gaussc.lowStd = altLowc
        gaussc.highStd = altHighc

        # Test to make sure that the values are distributing properly
        realResult1r = irregularGauss(x=xc.real, mean=altMeanc1.real, lowStd=altLowc1.real, highStd=altHighc1.real)
        realResult1i = irregularGauss(x=t.zeros_like(xc.imag), mean=altMeanc1.imag, lowStd=altLowc1.imag, highStd=altHighc1.imag)
        realResult1c = t.view_as_complex(t.stack((realResult1r, realResult1i), dim=-1))
        self.assertTrue(t.all((gaussc1.forward(xc.real) - realResult1c).abs() < 1e-4))

        result1r = realResult1r
        result1i = irregularGauss(x=xc.imag, mean=altMeanc1.imag, lowStd=altLowc1.imag, highStd=altHighc1.imag)
        result1c = t.view_as_complex(t.stack((result1r, result1i), dim=-1))
        self.assertTrue(t.all((gaussc1.forward(xc) - result1c).abs() < 1e-4))
