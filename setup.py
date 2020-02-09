import setuptools
import sys
import re

if sys.version_info.major != 3:
    print('This Python is only compatible with Python 3, but you are running '
          'Python {}. The installation will likely fail.'.format(sys.version_info.major))

extras = {
    'distributed': [
        'mpi4py'
    ]
}

setuptools.setup(name='agnes',
                 packages=setuptools.find_packages(),
                 install_requires=[
                     'gym',
                     'scipy',
                     'Tensorboard',
                     'cloudpickle',
                     'numpy',
                     'opencv-python'
                 ],
                 extras_require=extras,
                 description='AGNES - Flexible Reinforcement Learning Framework with PyTorch',
                 author='Rotinov Egor',
                 url='https://github.com/rotinov/AGNES',
                 version='0.0.7.2',
                 classifiers=[
                     "Programming Language :: Python :: 3",
                     "License :: OSI Approved :: MIT License",
                     "Operating System :: OS Independent",
                 ]
                 )
