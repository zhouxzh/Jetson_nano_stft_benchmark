# -*- coding: utf-8 -*-

"""
Provides a 2-minute test file for STFT experiments.

Author: Jan Schlüter
"""

import numpy as np

def make_test_signal(length=22050 * 120, seed=42):
    """
    Creates a test signal of `length` samples, with given random `seed`,
    returned as a numpy array of dtype np.int16.
    """
    rng = np.random.RandomState(seed)
    t = 2 * np.pi * np.arange(length) / 22050.
    s = sum(rng.randn() * np.sin(t * (1000 + 10000 * rng.rand() +
                                      10 * np.sin(t * rng.rand())))
            for _ in range(10))
    s *= (2**15 - 1) / s.max()
    return s.astype(np.int16)