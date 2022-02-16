import torch as t
import torch.nn as nn
import torch.nn.functional as nnf
import torch.fft as tfft

from .routing import KnowledgeFilter

class WeightedAggregator(KnowledgeFilter):
