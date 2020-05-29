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

with open(path.join(this_directory, 'requirements.txt'), encoding='utf-8') as f:
    list_of_requires = [line for line in f]

setuptools.setup(
    name='agnes',
    packages=setuptools.find_packages(),
    install_requires=list_of_requires,
    extras_require={
        'distributed': ['mpi4py'],
        'tensorboard': ['Tensorboard'],
        'all': ['mpi4py', 'Tensorboard']
    },
    author='Rotinov Egor',
    author_email='rotinov-github@pm.me',
    url='https://github.com/rotinov/AGNES',
    version=version.version,
    description='AGNES - Flexible Reinforcement Learning Framework with PyTorch',
    keywords="PyTorch reinforcement learning ppo a2c framework random network distillation",
    long_description=long_description,
    long_description_content_type='text/markdown',
    license='MIT LICENSE',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Operating System :: OS Independent",
    ]
)
