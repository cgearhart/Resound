from setuptools import setup
import os

long_description = 'A Python audio clip fingerprinting library'
if os.path.exists('README.txt'):
    long_description = open('README.txt').read()

# https://pythonhosted.org/setuptools/setuptools.html#id7
setup(
    name='Resound',
    version='1.1',
    packages=['resound'],
    install_requires=[
        'scipy',
        'numpy'
    ],
    author="Chris Gearhart",
    author_email="chris@gearley.com",
    description="Audio clip fingerprinting library",
    long_description=long_description,
    license="CC0",
    keywords="audio, fingerprinting",
    url="https://github.com/cgearhart/Resound",
)
