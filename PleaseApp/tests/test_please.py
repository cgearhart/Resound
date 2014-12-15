
import unittest

import wavfile
import please


class PleaseTest(unittest.TestCase):

    def setUp(self):
        _, self.data = wavfile.read('tests/sample.wav')

    def test_spectrogram(self):
        """
        spectrogram should return half the frame width + 1 frequency
        components - the 0 component and all positive components
        """
        specgram = please.spectrogram(self.data, 4096, 2048)
        self.assertEqual(len(specgram[0, :]), 2049)

    def test_fingerprints(self):
        """
        fingerprint keys should be of type<long> (native python type) for
        compatibility with app engine datastore
        """
        fpg = please.fingerprints(self.data, 44100)
        self.assertIsInstance(next(fpg), long)


if __name__ == "__main__":
    unittest.main()
