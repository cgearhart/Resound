
import argparse
import cPickle as pickle

from collections import defaultdict

import numpy as np
import resound

from pydub import AudioSegment
from scipy.io import wavfile


def main(input_filename, songname, format):
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
    fingerprints = list(resound.hashes(np.array(wav_data)))  # hash, offset pairs

    if not fingerprints:
        raise RuntimeError("No fingerprints detected in source file - check your parameters passed to Resound.")

    # Combine duplicate keys
    counter = defaultdict(lambda: [])
    for fp, offset in fingerprints:
        counter[fp].append((offset, songname))  # store the (offset, title) pair for each fingerprint

    # return the results as a standard dictionary
    return dict(counter)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fingerprint the specified " +
                                 "audio file and save the fingerprint to disk " +
                                 "as a pickled dict.")
    parser.add_argument('filename')
    parser.add_argument('songname')
    parser.add_argument("-f", "--format", default="mp3",
                        help="See pydub documentation for supported formats.")
    args = parser.parse_args()

    print "Processing file: {}".format(args.filename)
    results = main(args.filename, args.songname, args.format)

    with open('output.pickle', 'wb') as out_file:
        pickle.dump(results, out_file)

    print "Done."
