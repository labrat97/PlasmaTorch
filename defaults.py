import torch

DEFAULT_FFT_SAMPLES = 384
KNOTS_WITHOUT_LOSS = 8
# a peak and a trough are needed at minimum for a legible signal
DEFAULT_KNOT_WAVES = int(DEFAULT_FFT_SAMPLES / (KNOTS_WITHOUT_LOSS * 2))
DEFAULT_DTYPE = torch.float32
DEFAULT_SPACE_PRIME = 11