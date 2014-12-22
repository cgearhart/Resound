
import numpy as np
from numpy.fft import rfft

# Spectrogram parameter defaults
FREQ = 44100
FRAME_WIDTH = 4096
FRAME_OVERLAP = 2048

# Define the number of partitions for subcells in peak finding, and the
# threshold value for the minimum peak/avg ratio required to accept a peak
FREQ_BINS = 16
TIME_BINS = 3  # number of time bins per second

# Threshold function biases towards low frequency peaks (there is almost no
# signal at high frequency; samples are very close to the noise floor, and
# it's easier to ignore those points than to deal with finding the noise
# floor)
THRESHOLD = lambda x, y: 0.9 * (1 - (1.0 * x / y) ** 4)

# Define the number of column bins to skip (the gap from a peak to the
# constellation window) and the size of the constellation window
WINDOW_GAP = 1  # time in seconds
WINDOW_F_BINS = 3
WINDOW_TIME = 2  # time in seconds


def hashes(t_signal, freq=FREQ):
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
    peaks = _extract_peaks(specgram, strides=(t_step, f_step))

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
                yield (_get_hash(f1, f2, t2 - t1), t1 // frames_per_tbin)


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
    key = (f1 & 0xffff) << 48 | (f2 & 0xffff) << 32 | (dt & 0x3fff)
    return key.astype(np.uint64).item()


def _extract_peaks(specgram, strides, threshold=THRESHOLD):
    """
    Partition the spectrogram into subcells and extract peaks from each
    cell if the peak is sufficiently energetic compared to the neighborhood.
    """
    sr, sc = strides
    peaks = []
    for i in range(0, specgram.shape[0], sr):
        for j in range(0, specgram.shape[1], sc):
            cell = specgram[i:i + sr, j:j + sc]
            cmax = cell.max()
            cavg = np.mean(cell)

            if not (np.isfinite(cmax) and np.isfinite(cavg)):
                continue

            if cmax * threshold(j, specgram.shape[1]) > cavg:
                x, y = np.unravel_index(cell.argmax(), cell.shape)

                # ignore values on the border (at 0 or max frequency)
                if 0 < j + y < specgram.shape[1]:
                    peaks.append((i + x, j + y))

    return peaks
