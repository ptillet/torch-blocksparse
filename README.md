# Torch-Blocksparse

Block-sparse operations for PyTorch

## Current State

For now, only block-sparse linear operations (i.e., matrix-multiplication) are supported, for both FP32 and FP16 using tensor cores.

## Installation

Torch-Blocksparse depends on the [Triton](https://github.com/ptillet/triton) language and compiler:
```
pip install -e "git+https://github.com/ptillet/triton.git#egg=triton&subdirectory=python"
```

Test your installation using the following:
```
python test.py
```
The first run will take some time as all the necessary CUDA code will be JIT-compiled and cached in `$HOME/.triton/cache`.

You can just copy the `torch_blocksparse.py` file and use it in your project.

## Performance

Performance should be comparable to the OpenAI Blocksparse package.