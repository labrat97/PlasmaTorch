import unittest

import torch as t
from plasmatorch import *

from random import random, randint



class LinspaceTest(unittest.TestCase):
    def testSizingTyping(self):
        # Create the testing numbers
        a:float = random()
        b:float = abs(random()) + a
        steps:int = randint(2, 196884)

        # Create testing tensors
        lfn = linspace(start=a, end=b, steps=steps)
        lf1 = linspace(start=a, end=b, steps=1)
        lcn = linspace(start=complex(a), end=complex(b), steps=steps)
        lc1 = linspace(start=complex(a), end=complex(b), steps=1)
        ltfn = linspace(start=t.tensor(a, dtype=DEFAULT_DTYPE), end=t.tensor(b, dtype=DEFAULT_DTYPE), steps=steps)
        ltf1 = linspace(start=t.tensor(a, dtype=DEFAULT_DTYPE), end=t.tensor(b, dtype=DEFAULT_DTYPE), steps=1)
        ltcn = linspace(start=t.tensor(a, dtype=DEFAULT_COMPLEX_DTYPE), end=t.tensor(b, dtype=DEFAULT_COMPLEX_DTYPE), steps=steps)
        ltc1 = linspace(start=t.tensor(a, dtype=DEFAULT_COMPLEX_DTYPE), end=t.tensor(b, dtype=DEFAULT_COMPLEX_DTYPE), steps=1)

        # Test the types of the tensors
        self.assertFalse(lfn.is_complex())
        self.assertFalse(lf1.is_complex())
        self.assertTrue(lcn.is_complex())
        self.assertTrue(lc1.is_complex())
        self.assertFalse(ltfn.is_complex())
        self.assertFalse(ltf1.is_complex())
        self.assertTrue(ltcn.is_complex())
        self.assertTrue(ltc1.is_complex())

        # Test the sizes of the tensors
        self.assertEqual(list(lfn.size()), [steps])
        self.assertEqual(list(lcn.size()), [steps])
        self.assertEqual(list(ltfn.size()), [steps])
        self.assertEqual(list(ltcn.size()), [steps])
        self.assertEqual(list(lf1.size()), [1])
        self.assertEqual(list(lc1.size()), [1])
        self.assertEqual(list(ltf1.size()), [1])
        self.assertEqual(list(ltc1.size()), [1])

    def testConsistency(self):
        # Create the testing numbers
        a:float = random()
        ac:complex = a + (random() * 1j)
        b:float = abs(random()) + a
        bc:complex = b + (random() * 1j)
        steps:int = randint(2, 196884)

        # Create testing tensors
        lfn = linspace(start=a, end=b, steps=steps)
        lcn = linspace(start=ac, end=bc, steps=steps)
        ltfn = linspace(start=t.tensor(a, dtype=DEFAULT_DTYPE), end=t.tensor(b, dtype=DEFAULT_DTYPE), steps=steps)
        ltcn = linspace(start=t.tensor(ac, dtype=DEFAULT_COMPLEX_DTYPE), end=t.tensor(bc, dtype=DEFAULT_COMPLEX_DTYPE), steps=steps)

        # Create the control tensors
        cfn = t.linspace(start=a, end=b, steps=steps)
        ccn = t.linspace(start=ac, end=bc, steps=steps)

        # Test against the control tensors
        self.assertTrue(t.all((lfn - cfn).abs() <= 1e-4))
        self.assertTrue(t.all((ltfn - cfn).abs() <= 1e-4))
        self.assertTrue(t.all((lcn - ccn).abs() <= 1e-4))
        self.assertTrue(t.all((ltcn - ccn).abs() <= 1e-4))
    
    def testSingleValues(self):
        # Create the testing numbers
        a:float = random()
        ac:complex = a + (random() * 1j)
        b:float = abs(random()) + a
        bc:complex = b + (random() * 1j)
        avg:float = (a + b) / 2.
        avgc:complex = (ac + bc) / 2.

        # Create testing tensors
        lf1 = linspace(start=a, end=b, steps=1)
        lc1 = linspace(start=ac, end=bc, steps=1)
        ltf1 = linspace(start=t.tensor(a, dtype=DEFAULT_DTYPE), end=t.tensor(b, dtype=DEFAULT_DTYPE), steps=1)
        ltc1 = linspace(start=t.tensor(ac, dtype=DEFAULT_COMPLEX_DTYPE), end=t.tensor(bc, dtype=DEFAULT_COMPLEX_DTYPE), steps=1)

        # Test the results against the straight averages
        self.assertTrue(t.all((lf1 - avg).abs() <= 1e-4))
        self.assertTrue(t.all((lc1 - avgc).abs() <= 1e-4))
        self.assertTrue(t.all((ltf1 - avg).abs() <= 1e-4))
        self.assertTrue(t.all((ltc1 - avgc).abs() <= 1e-4))



class IrregularGaussTest(unittest.TestCase):
    SIZE = SUPERSINGULAR_PRIMES_HL[:4]

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
        xhigh = t.randn_like(x)

        # Calculate
        gauss = irregularGauss(x=x, mean=xmean, lowStd=xlow, highStd=xhigh, reg=False)
        gaussreg = irregularGauss(x=x, mean=xmean, lowStd=xlow, highStd=xhigh, reg=True)

        # Calculate control values
        TAU = tau()
        PHI = phi()
        softlow = ((1. / PHI) * t.log(1 + t.exp(PHI * xlow))).clamp(min=1e-12, max=1e12)
        softhigh = ((1. / PHI) * t.log(1 + t.exp(PHI * xhigh))).clamp(min=1e-12, max=1e12)
        reglow = 1. / (softlow * t.sqrt(TAU))
        reflow = t.exp(-0.5 * t.pow((x - xmean) / softlow, 2.))
        reghigh = 1. / (softhigh * t.sqrt(TAU))
        refhigh = t.exp(-0.5 * t.pow((x - xmean) / softhigh, 2.))

        # Test the values to make sure a normal guassian is happening on
        # either side of the mean
        xbottom = (x <= xmean).to(t.uint8)
        xtop = (x >= xmean).to(t.uint8)
        self.assertTrue(t.all(
            t.logical_or(t.logical_not(xbottom), (gauss - reflow) < 1e-4)
        ), msg=f'Lower curve off by approx: {t.mean(gauss-reflow)}')
        self.assertTrue(t.all(
            t.logical_or(t.logical_not(xtop), (gauss - refhigh) < 1e-4)
        ), msg=f'Upper curve off by approx: {t.mean(gauss-refhigh)}')
        self.assertTrue(t.all(
            (xbottom * (gaussreg - nantonum(reflow * reglow))) < 1e-3
        ), msg=f'Lower regular curve off by approx: {t.max((gaussreg-nantonum(reflow * reglow)) * xbottom)}')
        self.assertTrue(t.all(
            (xtop * (gaussreg - nantonum(refhigh * reghigh))) < 1e-3
        ), msg=f'Upper regular curve off by approx: {t.max((gaussreg-nantonum(refhigh * reghigh)) * xtop)}')



class LinearGaussTest(unittest.TestCase):
    SIZE = SUPERSINGULAR_PRIMES_HL[:4]

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
