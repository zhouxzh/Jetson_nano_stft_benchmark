#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Computes the spectrogram of a test signal using numpy and fftw.
Author: Jan Schlüter
"""

import sys
import os
import timeit
import numpy as np
import scipy.fftpack

def rfft_builder(*args, **kwargs):
    return scipy.fftpack.rfft

from testfile import make_test_signal

INPUT_AS_FLOAT = False


def spectrogram(samples, sample_rate=22050, frame_len=1024, fps=70, batch=50):
    """
    Computes a magnitude spectrogram for a given vector of samples at a given
    sample rate (in Hz), frame length (in samples) and frame rate (in Hz).
    Allows to transform multiple frames at once for improved performance (with
    a default value of 50, more is not always better). Returns a numpy array.
    """
    if len(samples) < frame_len:
        return np.empty((0, frame_len // 2 + 1), dtype=samples.dtype)
    win = np.hanning(frame_len).astype(samples.dtype)
    hopsize = sample_rate // fps
    num_frames = max(0, (len(samples) - frame_len) // hopsize + 1)
    batch = min(batch, num_frames)
    if batch <= 1 or not samples.flags.c_contiguous:
        rfft = rfft_builder(samples[:frame_len], n=frame_len)
        spect = np.vstack(np.abs(rfft(samples[pos:pos + frame_len] * win))
                          for pos in range(0, len(samples) - frame_len + 1,
                                           int(hopsize)))
    else:
        rfft = rfft_builder(np.empty((batch, frame_len), samples.dtype),
                            n=frame_len, threads=1)
        frames = np.lib.stride_tricks.as_strided(
                samples, shape=(num_frames, frame_len),
                strides=(samples.strides[0] * hopsize, samples.strides[0]))
        spect = [np.abs(rfft(frames[pos:pos + batch] * win))
                 for pos in range(0, num_frames - batch + 1, batch)]
        if num_frames % batch:
            spect.append(spectrogram(
                    samples[(num_frames // batch * batch) * hopsize:],
                    sample_rate, frame_len, fps, batch=1))
        spect = np.vstack(spect)
    return spect


def main():
    # load input
    global x
    x = make_test_signal()
    if INPUT_AS_FLOAT:
        x = x.astype(np.float32)

    # benchmark
    times = timeit.repeat(
            setup='from __main__ import x, spectrogram, np',
            stmt='spectrogram(np.asarray(x, np.float32))',
            repeat=5, number=32)
    print("Took %.3fs." % (min(times) / 32))

    # save result
    #np.save(sys.argv[0][:-2] + 'npy', spectrogram(x.astype(np.float32)))


if __name__=="__main__":
    main()
