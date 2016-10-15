
import argparse
import cPickle as pickle

from collections import Counter

import numpy as np
import resound

from pydub import AudioSegment
from scipy.io import wavfile

# resound defaults are WIDTH=4096, FRAME_STRIDE=1034. Increasing frame stride
# produces fewer peaks (runs faster), but it is important for the database
# songs to be sampled at a strde at least as small as the clip to be
# identified.
WIDTH = 4096
FRAME_STRIDE = 1024
FREQ_STRIDE = 32
TIME_STRIDE = 2


def find_matches(fingerprints, songs_db):
    # The absolute offset is the time measured from the beginning of the song
    # when it was fingerprinted. The relative offset is the time measured from
    # the beginning of the sample window (the clip being identified). The
    # delta will be "similar" for many keypoints for the song that the clip
    # matches. Increasing the offset tolerance allows more permissive matches.
    offset_tolerance = 100
    matches = Counter()
    for fp, rel_offset in fingerprints:
        for abs_offset, name in songs_db.get(fp, []):
            delta = (abs_offset - rel_offset) // offset_tolerance
            matches[(name, delta)] += 1
    return matches


def main(input_filename, format):
    """
    Calculate the fingerprint hashses of the referenced audio file and save
    to disk as a pickle file
    """

    # open the file & convert to wav
    song_data = AudioSegment.from_file(input_filename, format=format)
    song_data = song_data.set_channels(1)  # convert to mono
    wav_tmp = song_data.export(format="wav")  # write to a tmp file buffer
    wav_tmp.seek(0)
    rate, wav_data = wavfile.read(wav_tmp)

    rows_per_second = (1 + (rate - WIDTH)) // FRAME_STRIDE

    # Calculate a coarser window for matching
    window_size = (rows_per_second // TIME_STRIDE, (WIDTH // 2) // FREQ_STRIDE)
    peaks = resound.get_peaks(np.array(wav_data), window_size=window_size)

    # half width (nyquist freq) & half size (window is +/- around the middle)
    f_width = WIDTH // (2 * FREQ_STRIDE) * 2
    t_gap = 1 * rows_per_second
    t_width = 2 * rows_per_second
    fingerprints = resound.hashes(peaks, f_width=f_width, t_gap=t_gap, t_width=t_width)  # hash, offset pairs

    return fingerprints


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fingerprint the specified " +
                                     "audio file and save the fingerprint to disk " +
                                     "as a pickled dict.")
    parser.add_argument('filename')
    parser.add_argument('fingerprints')
    parser.add_argument("-f", "--format", default="mp3",
                        help="See pydub documentation for supported formats.")
    args = parser.parse_args()

    with open(args.fingerprints, 'rb') as f:
        songs_db = pickle.load(f)

    print "Processing audio."
    fingerprints = main(args.filename, args.format)

    matches = find_matches(fingerprints, songs_db)

    if not matches:
        print "No matches found."
        exit()

    print "Best matches"
    print list(reversed(sorted(matches.items(), key=lambda x: x[-1])[-10:]))
