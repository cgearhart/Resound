
import unittest

import numpy as np

import resound


class PleaseTest(unittest.TestCase):

    def setUp(self):
        # import sample data from a wav file
        self.data = np.fromfile('tests/sample.dat', dtype=np.int16)

    def test_spectrogram(self):
        """
        spectrogram should return half the frame width + 1 frequency
        components - the 0 component and all positive components
        """
        specgram = resound.spectrogram(self.data, 4096, 2048)
        self.assertEqual(len(specgram[0, :]), 2049)

    def test_fingerprints(self):
        """
        fingerprint keys should be of type<long> (native python type) for
        compatibility with app engine datastore
        """
        fpg = resound.hashes(self.data, 44100)
        self.assertIsInstance(next(fpg)[0], long)


if __name__ == "__main__":
    unittest.main()
