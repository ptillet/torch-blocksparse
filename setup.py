#!/usr/bin/env python

import os
import torch
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CppExtension

cmdclass = {}
cmdclass['build_ext'] = BuildExtension


import setuptools

ext_modules = [
    CppExtension(name='torch_blocksparse_cpp_utils',
                 sources=['csrc/utils.cpp'],
                 extra_compile_args={'cxx': ['-O2',
                                             '-fopenmp']})
]

setuptools.setup(
    name             = 'torch-blocksparse',
    version          = '1.1.1',
    description      = 'Block-sparse primitives for PyTorch',
    author           = 'Philippe Tillet',
    maintainer       = 'Philippe Tillet',
    maintainer_email = 'ptillet@g.harvard.edu',
    install_requires = ['triton', 'torch'],
    url              = 'https://github.com/ptillet/torch-blocksparse',
    test_suite       = 'nose.collector',
    tests_require    = ['nose', 'parameterized'],
    license          = 'MIT',
    packages         = find_packages(exclude=["csrc"]),
    ext_modules      = ext_modules,
    cmdclass         = cmdclass
)
