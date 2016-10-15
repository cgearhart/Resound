
import numpy as np
from numpy.fft import rfft

from scipy.ndimage.filters import convolve
from scipy.ndimage.morphology import grey_dilation

# Spectrogram parameter defaults
FREQ = 44100
FRAME_WIDTH = 4096
FRAME_OVERLAP = 2048

# Define the number of partitions for subcells in peak finding, and the
# threshold value for the minimum peak/avg ratio required to accept a peak
FREQ_BINS = 16
TIME_BINS = 3  # number of time bins per second

# Define the number of column bins to skip (the gap from a peak to the
# constellation window) and the size of the constellation window
WINDOW_GAP = 1  # time in seconds
WINDOW_F_BINS = 3
WINDOW_TIME = 2  # time in seconds


def hashes(t_signal, freq=FREQ, threshold=0.4):
    """
    Generator function for successive hashes calculated from a mono-channel
    time-domain audio signal as a set of tuples, (<long>, <int>). The <long>
    should be an integral 64-bit hash so it can be used as a database ID, and
    the <int> should be the frame number associated with the beginning of
    the time bin for the anchor point.
    """
    specgram = spectrogram(t_signal)

    frames_per_tbin = (FRAME_OVERLAP * TIME_BINS)

    t_step = freq // frames_per_tbin  # TIME_BINS splits in 1 sec
    f_step = FRAME_WIDTH // FREQ_BINS // 2  # FREQ_BINS splits on freq axis
    peaks = _extract_peaks(specgram, neighborhood=(t_step, f_step), threshold=threshold)

    for i, (t1, f1) in enumerate(peaks):

        if t1 > t_signal.shape[0] - (WINDOW_GAP + WINDOW_TIME) * freq:
            break

        # limit constellations to a window
        t_min = int(t1 + WINDOW_GAP * freq / FRAME_WIDTH / 2)
        t_max = int(t_min + WINDOW_TIME * freq / FRAME_WIDTH / 2)
        f_min = max(0, f1 - WINDOW_F_BINS * f_step // 4)
        f_max = min(specgram.shape[1], f1 + WINDOW_F_BINS * f_step // 4)

        # print t1, f1, t_min, t_max, f_min, f_max
        for t2, f2 in peaks[i:]:
            if t2 < t_min or f2 < f_min:
                continue
            elif t2 > t_max:
                break
            elif f2 < f_max:
                yield (_get_hash(f1, f2, t2 - t1), t1)


def spectrogram(t_signal, frame_width=FRAME_WIDTH, overlap=FRAME_OVERLAP):
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
    return 20 * np.log10(np.absolute(f_signal))


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
