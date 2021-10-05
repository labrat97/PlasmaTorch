import unittest
import test

import torch
from defaults import DEFAULT_DTYPE
from plasmatorch import *
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
        
        self.assertEqual(subject.signalCount, subjectc.signalCount)
        self.assertEqual(subject.curveChannels, subjectc.curveChannels)
        self.assertEqual(subject.samples, subjectc.samples)
        self.assertEqual(subject.outputMode, subjectc.outputMode)
        self.assertEqual(len(subject.entangleActivation), len(subjectc.entangleActivation))
        self.assertEqual(subject.entanglePolarization.size(), subjectc.entanglePolarization.size())
        self.assertEqual(subject.knowledgeMask.size(), subject.knowledgeMask.size())

        self.assertEqual(subject.signalCount, signals)
        self.assertEqual(subject.curveChannels, channels)
        self.assertEqual(subject.samples, test.TEST_FFT_SAMPLES)
        self.assertEqual(subject.outputMode, EntangleOutputMode.BOTH)
        self.assertEqual(len(subject.entangleActivation), signals)
        self.assertEqual(len(subject.entanglePolarization), signals)
        self.assertEqual(subject.knowledgeMask.size(), (signals, channels, test.TEST_FFT_SAMPLES, test.TEST_FFT_SAMPLES))

        self.assertTrue(torch.all(subject.knowledgeMask == toComplex(torch.eye(test.TEST_FFT_SAMPLES))))
        self.assertTrue(torch.all(subjectc.knowledgeMask == toComplex(torch.eye(test.TEST_FFT_SAMPLES))))

    def testSizing(self):
        # Parameter initialization
        signals = randint(1, 13)
        channels = randint(1, 13)

        # Create the modules required to test the enclosed parameters for consistency
        subject = Entangle(inputSignals=signals, curveChannels=channels, \
            samples=test.TEST_FFT_SAMPLES, useKnowledgeMask=True, \
            outputMode=EntangleOutputMode.BOTH, dtype=DEFAULT_DTYPE)
        subjectc = Entangle(inputSignals=signals, curveChannels=channels, \
            samples=test.TEST_FFT_SAMPLES, useKnowledgeMask=True, \
            outputMode=EntangleOutputMode.BOTH, dtype=DEFAULT_COMPLEX_DTYPE)
