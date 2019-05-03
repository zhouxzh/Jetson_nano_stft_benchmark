#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Computes the spectrogram of a test signal using cupy and cuFFT.

Author: Jan Schlüter
"""

import sys
import os
import timeit

import numpy as np
import cupy as cp

INPUT_ON_GPU = True
OUTPUT_ON_GPU = True

from testfile import make_test_signal


def spectrogram(signal, sample_rate=22050, frame_len=1024, fps=70):
    """
    Computes a magnitude spectrogram at a given sample rate (in Hz), frame
    length (in samples) and frame rate (in Hz), on CUDA using cupy.
    """
    if not INPUT_ON_GPU:
        signal = cp.array(signal.astype(np.float32))  # already blown up to a list of frames
    win = cp.hanning(frame_len).astype(cp.float32)
    # apply window function
    #signal *= win  # this doesn't work correctly for some reason.
    signal = signal * win
    # perform FFT
    spect = cp.fft.rfft(signal)
    # convert into magnitude spectrogram
    spect = cp.abs(spect)
    # return
    if OUTPUT_ON_GPU:
        cp.cuda.get_current_stream().synchronize()
    else:
        return spect.get()


def main():
    # load input
    global x, spectrogram
    x = make_test_signal()
    # we do the following here because cupy cannot do stride tricks
    # the actual copying work is included in the benchmark unless INPUT_ON_GPU
    hop_size = 22050 // 70
    frame_len = 1024
    frames = len(x) - frame_len + 1
    x = np.lib.stride_tricks.as_strided(
            x, (frames, frame_len), (x.strides[0], x.strides[0]))[::hop_size]
    if INPUT_ON_GPU:
        x = cp.array(x.astype(np.float32))

    # benchmark
    times = timeit.repeat(
            setup='from __main__ import x, spectrogram',
            stmt='spectrogram(x)',
            repeat=5, number=32)
    print("Took %.3fs." % (min(times) / 32))

    # save result
    #assert not OUTPUT_ON_GPU
    #np.save(sys.argv[0][:-2] + 'npy', spectrogram(x))


if __name__=="__main__":
    main()