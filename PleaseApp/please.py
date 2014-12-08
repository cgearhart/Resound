
import numpy as np
from numpy.fft import fft


def spectrogram(t_signal, freq, window_size, time_shift):
    """
    Calculate the STFT of a time-domain signal. The frame window size and
    time shift are specified in seconds, and the frequency in hertz.
    """
    fs = int(freq * window_size)
    offset = int(freq * time_shift)
    w = np.hamming(fs)

    num_frames = (len(t_signal) - fs) // offset
    num_components = (fs + 1) // 2
    f_signal = np.empty([num_frames, num_components], dtype='float64')

    for i, t in enumerate(range(0, len(t_signal) - fs), offset):
        # only keep the first half of the components
        f_signal[i, :] = fft(w * t_signal[t:t + fs])[:num_components]

    return abs(f_signal)


class Please(object):
    """
    This class provides a simple implementation of the Shazam algorithm for
    audio file identification.
    """

    _ROUND_LIMIT = 10  # number of search rounds for signal threshold

    # dimension constraints for the target zone window
    _HEIGHT = 0.10  # half-height (percentage)
    _DEPTH = 3  # window depth (number of seconds)
    _DELAY = 1  # delay from anchor time to start of target zone window

    def __init__(self, freq=44100, frame_time=.05, density=3, tolerance=0.2):
        """
        Set the expected sampling frequency of the audio signal and define the
        window function time frame, and offset between window frames. The
        density argument is the desired number of peaks per second.
        """
        self.freq = freq
        self.frame_width = frame_time
        self.offset = 0.5 * frame_time
        self.density = density * frame_time  # expected peaks per frame
        self.tolerance = tolerance * self.density

    def fingerprints(self, t_signal):
        """
        Create a list of fingerprint hashes from a time-domain audio signal as
        a set of tuples (f1, f2, t2-t1), where f1 and f2 are peak frequencies
        and t2-t1 is the time offset between the peaks. Each f1 is an "anchor"
        point, and each f2 is within the target window as defined by
        experimentally determined constants.
        """

        f_signal = spectrogram(t_signal, self.freq,
                               self.frame_width, self.offset)

        f_signal /= f_signal.max()  # scale the data to range [0, 1.0]
        peaks = self._find_peaks(f_signal)  # extract peaks at target density

        # calculate the constellation target zone dimensions
        h_max, t_max = peaks.shape
        dy = self._HEIGHT * h_max
        dx = self._DEPTH * self.frame_width
        delta = self._DELAY * self.frame_width

        for f1, t1 in np.transpose(np.nonzero(peaks)):
            # filter anchor points based on target zone dimensions
            if f1 < dy or f1 > h_max - dy or t1 > t_max - dx - delta:
                continue

            x = t1 + delta
            for f2, t2 in np.transpose(np.nonzero(peaks[x:x+dx, f1-dy:f1+dy])):
                yield (f1, f2, t2 - t1)
            
    def _find_peaks(self, f_signal):
        """
        Use a binary search to automatically determine an appropriate threshold
        value to find peaks in the spectrogram
        """
        threshold, delta = 0.5, 0.5
        for _ in range(self._ROUND_LIMIT):

            tData = f_signal > threshold

            # Calculate the average number of peaks per frame
            peaks_per_frame = np.mean(np.sum(tData, axis=0))

            # return if the average peak count is within tolerance otherwise
            # loop to refine the threshold
            if abs(peaks_per_frame - self.density) <= self.tolerance:
                return f_signal * tData
            elif peaks_per_frame > self.density:
                threshold -= delta
            elif peaks_per_frame < self.density:
                threshold += delta
            delta /= 2

        # Raise an error because the search did not converge
        raise Exception("Threshold convergence failed in _find_peaks.")
