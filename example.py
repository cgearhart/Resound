
import argparse
import cPickle as pickle

from collections import Counter

import numpy as np
import resound

from pydub import AudioSegment
from scipy.io import wavfile


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
    fingerprints = resound.hashes(np.array(wav_data))  # hash, offset pairs

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

    matches = Counter()
    for fp, offset in fingerprints:
        entries = songs_db.get(fp, [])
        for value, name in entries:
            if offset == value:
                matches[name] += 1

    if not matches:
        print "No matches found."
        exit()

    print "Best matches"
    print list(reversed(sorted(matches.items())[-10:]))
