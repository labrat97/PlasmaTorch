import unittest
import test

import torch
from plasmatorch import *
from plasmatorch.defaults import DEFAULT_DTYPE
import math
from random import randint

class EntangleTest(unittest.TestCase):
    def testParameters(self):
        # Create the modules required to test the enclosed parameters for consistency
        signals = randint(1, 13)
        channels = randint(1, 13)
        subject = Entangle(inputSignals=signals, curveChannels=channels, \
            samples=test.TEST_FFT_SAMPLES, useKnowledgeMask=True, \
            outputMode=EntangleOutputMode.BOTH, dtype=DEFAULT_DTYPE)
        subjectc = Entangle(inputSignals=signals, curveChannels=channels, \
            samples=test.TEST_FFT_SAMPLES, useKnowledgeMask=True, \
            outputMode=EntangleOutputMode.BOTH, dtype=DEFAULT_COMPLEX_DTYPE)
        

        # Assert that both structures have the same settings after construction
        self.assertEqual(subject.signalCount, subjectc.signalCount)
        self.assertEqual(subject.curveChannels, subjectc.curveChannels)
        self.assertEqual(subject.samples, subjectc.samples)
        self.assertEqual(subject.outputMode, subjectc.outputMode)
        self.assertEqual(len(subject.entangleActivation), len(subjectc.entangleActivation))
        self.assertEqual(subject.entanglePolarization.size(), subjectc.entanglePolarization.size())
        self.assertEqual(subject.knowledgeMask.size(), subject.knowledgeMask.size())

        # Make sure that the weights aren't the wrong size resulting in less effecient computation
        self.assertEqual(subject.signalCount, signals)
        self.assertEqual(subject.curveChannels, channels)
        self.assertEqual(subject.samples, test.TEST_FFT_SAMPLES)
        self.assertEqual(subject.outputMode, EntangleOutputMode.BOTH)
        self.assertEqual(len(subject.entangleActivation), signals)
        self.assertEqual(len(subject.entanglePolarization), signals)
        self.assertEqual(subject.knowledgeMask.size(), (signals, channels, test.TEST_FFT_SAMPLES, test.TEST_FFT_SAMPLES))

        # Assert that the identity matrix is properly being transfered over to the
        #   real numbers only to initialize the knowledge mask weights
        self.assertTrue(torch.all(subject.knowledgeMask == toComplex(torch.eye(test.TEST_FFT_SAMPLES))))
        self.assertTrue(torch.all(subjectc.knowledgeMask == toComplex(torch.eye(test.TEST_FFT_SAMPLES))))

    def testSizing(self):
        # Parameter initialization
        batches = randint(1, 4)
        signals = randint(1, 7)
        channels = randint(1, 5)

        # Create the modules required to test the enclosed parameters for consistency
        subject = Entangle(inputSignals=signals, curveChannels=channels, \
            samples=test.TEST_FFT_SAMPLES, useKnowledgeMask=True, \
            outputMode=EntangleOutputMode.BOTH, dtype=DEFAULT_DTYPE)
        subjectc = Entangle(inputSignals=signals, curveChannels=channels, \
            samples=test.TEST_FFT_SAMPLES, useKnowledgeMask=True, \
            outputMode=EntangleOutputMode.BOTH, dtype=DEFAULT_COMPLEX_DTYPE)

        # Tensors for evaluation
        x = torch.randn((batches, signals, channels, test.TEST_FFT_SAMPLES), dtype=DEFAULT_DTYPE)
        xc = torch.randn((batches, signals, channels, test.TEST_FFT_SAMPLES), dtype=DEFAULT_COMPLEX_DTYPE)

        # Push data through entanglers
        col, sup = subject.forward(x)
        colc, supc = subjectc.forward(xc)

        # Assert equivalence between real and complex data sizes
        self.assertEqual(col.size(), colc.size())
        self.assertEqual(sup.size(), supc.size())
        
        # Assert equivalence between input and output sizes
        self.assertEqual(x.size(), col.size())
        self.assertEqual(xc.size(), colc.size())

        # Assert equivalence between the knowledge mask sizing and the
        #   the superposition sizing
        self.assertEqual(sup.size()[1:], subject.knowledgeMask.size())
        self.assertEqual(supc.size()[1:], subjectc.knowledgeMask.size())

    def testValues(self):
        # Parameter initialization
        batches = randint(1, 4)
        signals = randint(1, 7)
        channels = randint(1, 5)

        # Create the modules required to test the enclosed parameters for consistency
        subject = Entangle(inputSignals=signals, curveChannels=channels, \
            samples=test.TEST_FFT_SAMPLES, useKnowledgeMask=True, \
            outputMode=EntangleOutputMode.BOTH, dtype=DEFAULT_DTYPE)
        subjectc = Entangle(inputSignals=signals, curveChannels=channels, \
            samples=test.TEST_FFT_SAMPLES, useKnowledgeMask=True, \
            outputMode=EntangleOutputMode.BOTH, dtype=DEFAULT_COMPLEX_DTYPE)

        # Tensors for evaluation
        x = torch.zeros((batches, signals, channels, test.TEST_FFT_SAMPLES), dtype=DEFAULT_DTYPE)
        xc = torch.zeros((batches, signals, channels, test.TEST_FFT_SAMPLES), dtype=DEFAULT_COMPLEX_DTYPE)

        # Push data through entanglers
        col, sup = subject.forward(x)
        colc, supc = subjectc.forward(xc)

        # Make sure that the values recieved on the other end come through as zero
        self.assertTrue(torch.all(col == 0))
        self.assertTrue(torch.all(sup == 0))
        self.assertTrue(torch.all(colc == 0))
        self.assertTrue(torch.all(supc == 0))

        # More tensors for evaluation, but for more value checking
        x = torch.randn_like(x)
        xc = torch.randn_like(xc)

        # Push data through the entanglers
        colrr, suprr = subject.forward(x)
        colrc, suprc = subject.forward(xc)
        colcr, supcr = subject.forward(x)
        colcc, supcc = subject.forward(xc)
        colrc0, suprc0 = subject.forward(toComplex(x))
        colcc0, supcc0 = subjectc.forward(toComplex(x))

        # Make sure the values that come through on the other side consistent with themselves
        self.assertTrue(torch.all(colrr == colcr))
        self.assertTrue(torch.all(suprr == supcr))
        self.assertTrue(torch.all(colrc == colcc))
        self.assertTrue(torch.all(suprc == supcc))

        # Cannot garuntee self similar collapse state across output types due to rfft vs fft
        #self.assertTrue(torch.all(colrr - colrc0.real < 0.0001))
        self.assertTrue(torch.all(suprr.real - suprc0.real < 0.0001))
        self.assertTrue(torch.all(suprr.imag - suprc0.imag < 0.0001))
        #self.assertTrue(torch.all(colcr - colcc0.real < 0.0001))
        self.assertTrue(torch.all(supcr.real - supcc0.real < 0.0001))
        self.assertTrue(torch.all(supcr.imag - supcc0.imag < 0.0001))

    def testDifference(self):
        # Parameter initialization
        batches = randint(1, 4)
        signals = randint(1, 7)
        channels = randint(1, 5)

        # Create the modules required to test the enclosed parameters for inconsistency
        subject = Entangle(inputSignals=signals, curveChannels=channels, \
            samples=test.TEST_FFT_SAMPLES, useKnowledgeMask=True, \
            outputMode=EntangleOutputMode.BOTH, dtype=DEFAULT_DTYPE)
        subjectc = Entangle(inputSignals=signals, curveChannels=channels, \
            samples=test.TEST_FFT_SAMPLES, useKnowledgeMask=True, \
            outputMode=EntangleOutputMode.BOTH, dtype=DEFAULT_COMPLEX_DTYPE)

        # Replace parameters to generate inconsistency
        subject.knowledgeMask = nn.Parameter(torch.randn_like(subject.knowledgeMask))
        subjectc.knowledgeMask = nn.Parameter(torch.randn_like(subjectc.knowledgeMask))

        # Tensors for evaluation
        x = torch.randn((batches, signals, channels, test.TEST_FFT_SAMPLES), dtype=DEFAULT_DTYPE)
        xc = torch.view_as_complex(torch.stack((x, x), dim=-1))

        # Push data through the entanglers
        colrr, suprr = subject.forward(x)
        colrc, suprc = subject.forward(xc)
        colcr, supcr = subject.forward(x)
        colcc, supcc = subject.forward(xc)

        # All of these should not be the same
        self.assertTrue(torch.all(colrr != colrc))
        self.assertTrue(torch.all(suprr != suprc))
        self.assertTrue(torch.all(colcr != colcc))
        self.assertTrue(torch.all(supcr != supcc))
