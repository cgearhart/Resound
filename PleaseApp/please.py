
import numpy as np
from numpy.fft import rfft

# number of rounds to allow in the threshold filter search before aborting
SEARCH_LIMIT = 12


def fingerprints(t_signal, freq=None, frame_width=4096, overlap=2048,
                 density=7, fanout=.1, maxpairs=5, floor=0.2):
    """
    Generator function for successive fingerprint hashes from a mono-channel
    time-domain audio signal. as a set of tuples (f1, f2, t2-t1), where f1 and
    f2 are peak frequencies and t2-t1 is the time offset between the peaks.
    The density constant defines the target number of peaks per second.
    """
    # Enforce default parameter constraints for STFT
    freq = freq or 44100
    frame_width = min(t_signal.shape[0], frame_width)

    specgram = spectrogram(t_signal, frame_width=frame_width, overlap=overlap)

    pps = 1.0 * density * len(t_signal) / freq
    peaks = _extract_peaks(specgram, num_peaks=pps, floor=floor)

    if not peaks:
        raise StopIteration

    # generate feature constellations from peak anchors
    offset = freq // overlap
    w_height = fanout * frame_width // 2
    for f1, t1 in peaks:
        paircount = 0

        if t1 > t_signal.shape[0] - offset:
            break

        # limit constellations to a window
        t_min, t_max = t1 + offset, t1 + 2 * offset
        f_min, f_max = f1 - w_height, f1 + w_height

        for f2, t2 in peaks:
            if t_min < t2 < t_max and f_min < f2 < f_max:
                paircount += 1
                yield _fingerprint_hash(f1, f2, t2 - t1)
            if paircount >= maxpairs:
                break


def spectrogram(t_signal, frame_width, overlap):
    """
    Calculate the magnitude spectrogram of a single-channel time-domain signal
    from the real frequency components of the STFT with a hanning window
    applied to each frame. The frame size and overlap between frames should
    be specified in number of samples.
    """
    w = np.hanning(frame_width)
    num_components = frame_width // 2 + 1
    num_frames = 1 + (len(t_signal) - frame_width) // overlap

    f_signal = np.empty([num_frames, num_components], dtype=np.complex_)
    for i, t in enumerate(range(0, len(t_signal) - frame_width, overlap)):
        # using rfft avoids computing negative frequency components
        f_signal[i] = rfft(w * t_signal[t:t + frame_width])

    # amplitude in decibels
    return 20 * np.log10(np.abs(f_signal))


def _fingerprint_hash(f1, f2, dt):
    """
    Calculate a 64-bit int hash of the fingerprint: <f1:f2:dt> f1 and f2 are
    in the range [0, 65536), and dt in the range [0, 1024)
    """
    key = (f1 & 0xffff) << 48 | (f2 & 0xffff) << 32 | (dt & 0x3ff)
    return key.astype(np.uint64).item()


def _extract_peaks(specgram, num_peaks, floor):
    """
    Generator function for extracting peaks from a spectrogram (in dB) by
    parabolic interpolation.

    https://ccrma.stanford.edu/~jos/parshl/Peak_Detection_Steps_3.html
    """
    peaks = []
    # quit if the specgram is empty
    if np.allclose(specgram, 0) or np.isinf(specgram.max()):
        return peaks

    specgram /= specgram.max()

    while len(peaks) < num_peaks and specgram.max() > floor:

        # find the max element in the spectrogram
        f, kb = np.unravel_index(specgram.argmax(), specgram.shape)
        peaks.append((f, kb))

        # remove the peak from the spectrogram
        start, end = _find_floor(specgram[f], kb, floor)
        first, last = max(0, f - 1), min(specgram.shape[0], f + 2)
        specgram[first:last, start:end] = 0

    return peaks


def _find_floor(frame, peak, threshold):
    """
    Estimate the width of frequency peaks in a STFT signal frame by performing
    a linear scan through the frame from a peak center to the left and right
    edge points where the signal falls below a threshold.
    """
    offset_l, offset_r = 2, 2
    incr_l, incr_r = 1, 1
    r_max = frame.shape[0] - 1

    while incr_l and incr_r:
        l_edge, r_edge = peak - offset_l, peak + offset_r

        if l_edge <= 0 or frame[l_edge] < threshold:
            l_edge, incr_l = 0, 0
        if r_edge >= r_max or frame[r_edge] < threshold:
            r_edge, incr_r = r_max, 0

        offset_l += incr_l
        offset_r += incr_r

    return (l_edge, r_edge)
