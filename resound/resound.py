
import numpy as np
from numpy.fft import rfft

from scipy.ndimage.filters import convolve
from scipy.ndimage.morphology import grey_dilation

# Spectrogram parameter defaults
# FREQ is the number of samples/sec in the audio file
# FRAME_WIDTH is the number of samples to include in each window frame of
# the DTFT. FRAME_OVERLAP is the step size (in samples/sec) to advance
# the window
FREQ = 44100
FRAME_WIDTH = 4096
FRAME_STRIDE = 1024

# Define the number of partitions for subcells in peak finding, and the
# threshold value for the minimum peak/avg ratio required to accept a peak
# Ex. If there are 2048 frequency terms in the DTFT of each window and 16
# frequency bins then the exclusion window for non-max suppression will
# contain 64 frequency values; 2048 / 16 = 64. If there are 18 time frames
# per second, then the exclusion window will be 18 / 4 = 4 bins (1/4 second)
FREQ_STRIDE = 32
TIME_STRIDE = 4

ROWS_PER_SECOND = (1 + (FREQ - FRAME_WIDTH) // FRAME_STRIDE)

# T_STEP is the number of DTFT windows (rows in specgram) to include in the
# exclusion window for non-max suppression
# Ex. 1 + (44100 - 4096) / 1024 = 40.06  : there are 40 sample windows/sec
#     40 // 3 = 13 rows
T_STEP = ROWS_PER_SECOND // TIME_STRIDE

# F_STEP is the number of DTFT frequencies (columns in specgram) to include
# in the non-max suppression exclusion window.
# Ex. (4096 // 2) / 16 = 128
# The DTFT can only return frequencies up to 1/2 the frame width.
F_STEP = (FRAME_WIDTH // 2) // FREQ_STRIDE

F_WIDTH = F_STEP * 2


def get_peaks(t_signal, window_size=(T_STEP, F_STEP), threshold=0.4):

    specgram = spectrogram(t_signal)

    # window size is used for non-maximal suppression
    peaks = _extract_peaks(specgram, neighborhood=window_size, threshold=threshold)

    return peaks


def hashes(peaks, f_width=F_WIDTH, t_gap=ROWS_PER_SECOND, t_width=2*ROWS_PER_SECOND):
    """
    Generator function for successive hashes calculated from a mono-channel
    time-domain audio signal as a set of tuples, (<long>, <int>). The <long>
    is an integral 64-bit hash so it can be used as a database ID, and
    the <int> is the frame number associated with the beginning of
    the time bin for the anchor point.

    The frequency window of each peak for constellation is +/- 1 octave

    Time gap and width recommendations:

    To calculate N seconds in rows (DTFT time windows):
        rows = N * (1 + (FREQ - FRAME_WIDTH) // FRAME_STRIDE)

    """
    for i, (t1, f1) in enumerate(peaks):

        # limit constellations to a window -- a box constrained by a min
        # and max time limit, and a min and max frequency bound
        t_min = t1 + t_gap
        t_max = t_min + t_width
        f_min = f1 - f_width // 2
        f_max = f1 + f_width // 2

        for t2, f2 in peaks[i:]:
            if t2 < t_min or f2 < f_min:
                continue
            elif t2 > t_max:
                break
            elif f2 < f_max:
                yield (_get_hash(f1, f2, t2 - t1), t1)


def spectrogram(t_signal, frame_width=FRAME_WIDTH, overlap=FRAME_STRIDE):
    """
    Calculate the magnitude spectrogram of a single-channel time-domain signal
    from the real frequency components of the STFT with a hanning window
    applied to each frame. The frame size and overlap between frames should
    be specified in number of samples.
    """
    frame_width = min(t_signal.shape[0], frame_width)
    w = np.hanning(frame_width)
    num_components = frame_width // 2 + 1
    num_frames = 1 + (len(t_signal) - frame_width) // overlap

    f_signal = np.empty([num_frames, num_components], dtype=np.complex_)
    for i, t in enumerate(range(0, len(t_signal) - frame_width, overlap)):
        # using rfft avoids computing negative frequency components
        f_signal[i] = rfft(w * t_signal[t:t + frame_width])

    # amplitude in decibels
    return 20 * np.log10(1 + np.absolute(f_signal))


def _get_hash(f1, f2, dt):
    """
    Calculate a 64-bit integral hash from <f1:f2:dt>, where f1 and f2 are
    FFT frequency bins (based on frame width), and dt is propotional to the
    time difference between f1 and f2 as the the difference in frame number
    between the points.
    """
    return ((long(f1) & 0xffff) << 48 |
            (long(f2) & 0xffff) << 32 |
            (long(dt) & 0x3fff))


def _extract_peaks(specgram, neighborhood, threshold):
    """
    Partition the spectrogram into subcells and extract peaks from each
    cell if the peak is sufficiently energetic compared to the neighborhood.
    """
    kernel = np.ones(shape=neighborhood)
    local_averages = convolve(specgram, kernel / kernel.sum(), mode="constant", cval=0)

    # suppress all points below the floor value
    floor = (1 + threshold) * local_averages
    candidates = np.where(specgram > floor, specgram, 0)

    # grayscale dilation is equivalent to non-maximal suppression
    local_maximums = grey_dilation(candidates, footprint=kernel)
    peak_coords = np.argwhere(specgram == local_maximums)
    peaks = zip(peak_coords[:, 0], peak_coords[:, 1])

    return peaks
