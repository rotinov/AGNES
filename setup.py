import sys
from os import path

import setuptools
import version

if sys.version_info.major != 3:
    print('This Python is only compatible with Python 3, but you are running '
          'Python {}. The installation will likely fail.'.format(sys.version_info.major))

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setuptools.setup(
    name='agnes',
    packages=setuptools.find_packages(),
    install_requires=[
        'gym',
        'scipy',
        'Tensorboard',
        'cloudpickle',
        'numpy',
        'opencv-python'
    ],
    extras_require={
        'distributed': ['mpi4py']
    },
    author='Rotinov Egor',
    url='https://github.com/rotinov/AGNES',
    version=version.version,
    description='AGNES - Flexible Reinforcement Learning Framework with PyTorch',
    keywords="pytorch reinforcement learning ppo a2c framework random network distillation",
    long_description=long_description,
    long_description_content_type='text/markdown',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Operating System :: OS Independent",
    ]
)
