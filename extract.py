
import argparse
import cPickle as pickle
import os

from glob import glob
from collections import defaultdict

import numpy as np
import resound

from pydub import AudioSegment
from scipy.io import wavfile


def main(input_filename, songname, format, counter):
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

    # extract peaks and compute constellation hashes & offsets
    peaks = resound.get_peaks(np.array(wav_data))
    fingerprints = list(resound.hashes(peaks))  # hash, offset pairs

    if not fingerprints:
        raise RuntimeError("No fingerprints detected in source file - check your parameters passed to Resound.")

    # Combine duplicate keys
    for fp, abs_offset in fingerprints:
        counter[fp].append((abs_offset, songname))

    print "    Identified {} keypoints in '{}'.".format(len(counter), songname)

    return counter


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fingerprint the specified " +
                                     "audio file and save the fingerprint to disk " +
                                     "as a pickled dict.")
    parser.add_argument('folder')
    parser.add_argument("-f", "--format", default="mp3",
                        help="See pydub documentation for supported formats.")
    args = parser.parse_args()

    counter = defaultdict(lambda: [])
    for filename in glob(os.path.join(args.folder, "*.{}".format(args.format))):
        path, _ = os.path.splitext(filename)
        _, song_name = os.path.split(path)
        print "Processing file: {} as '{}'".format(filename, song_name)
        counter = main(filename, song_name, args.format, counter)

    with open('output.pickle', 'wb') as out_file:
        pickle.dump(dict(counter), out_file)

    print "Done."
