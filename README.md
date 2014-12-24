# Resound

## Introduction
Resound is a python library containing a single module for generating fingerprint hashes from audio files based on the algorithm described in the paper ["An Industrial-Strength Audio Search Algorithm"](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.217.8882) and used in the Shazam app.

## Installation
The easiest way to to install `resound` is the `pip` utility.

    $pip install resound

It can also be installed as a [submodule](http://git-scm.com/docs/git-submodule) in another git repository (provided it has access to Numpy at runtime).

    $git submodule add https://github.com/cgearhart/Resound.git

## Testing
Resound unit tests rely only on the unittest module, so they can most easily be run from the command line from the root project directory.

    $python -m unittest discover

## Using Resound
The [wavfile.read()](http://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.io.wavfile.read.html) module from SciPy can be used to read the data:

    import resound

    from scipy.io import wavfile

    sample_rate, data = wavfile.read('filename.wav')
    hashes = list(resound.hashes(data, freq=sample_rate))

## License
Resound is released to the public domain under the CC0 license. See <http://creativecommons.org/publicdomain/zero/1.0/> for details.
