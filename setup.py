#!/usr/bin/env python

import setuptools

setuptools.setup(
    name             = 'torch-blocksparse',
    version          = '1.0.1',
    description      = 'Block-sparse primitives for PyTorch',
    author           = 'Philippe Tillet',
    maintainer       = 'Philippe Tillet',
    maintainer_email = 'ptillet@g.harvard.edu',
    install_requires = ['triton', 'torch'],
    packages         = ['torch_blocksparse'],
    url              = 'https://github.com/ptillet/torch-blocksparse',
    license          = 'MIT'
)