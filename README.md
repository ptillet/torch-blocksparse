# Torch-Blocksparse

Block-sparse operations for PyTorch

## Current State

The following operations are supported:
```
SPARSE = DENSE x DENSE
DENSE = SPARSE x DENSE
DENSE = DENSE x SPARSE
```
For each of these modes, either input can be transposed. Additionally, FP16 inputs are supported and will use tensor cores.

## Installation

Torch-Blocksparse depends on CUDA 10.1 and the [Triton](https://github.com/ptillet/triton) language and compiler:
```
sudo apt-get install llvm-8-dev;
pip install -e "git+https://github.com/ptillet/triton.git#egg=triton&subdirectory=python"
```

Test your installation using the following:
```
python test.py
```
The first run will take some time as all the necessary CUDA code will be JIT-compiled and cached in `$HOME/.triton/cache`.

You can just copy the `torch_blocksparse.py` file and use it in your project.

## Performance

Here is the performance of this package compared to OpenAI blocksparse for the DDS layout (dense = dense x sparse) with square, non-transposed inputs:

![](https://docs.google.com/spreadsheets/d/e/2PACX-1vTMh8lJHOYq07d2g7AQZOKb6-WgTQqK3iudLJ8I1LCgGKw_B9eKv1KFT0nKbrizy9fw-p2VjvIbTgLJ/pubchart?oid=717347395&format=image)

![](https://docs.google.com/spreadsheets/d/e/2PACX-1vTMh8lJHOYq07d2g7AQZOKb6-WgTQqK3iudLJ8I1LCgGKw_B9eKv1KFT0nKbrizy9fw-p2VjvIbTgLJ/pubchart?oid=1552535399&format=image)

![](https://docs.google.com/spreadsheets/d/e/2PACX-1vTMh8lJHOYq07d2g7AQZOKb6-WgTQqK3iudLJ8I1LCgGKw_B9eKv1KFT0nKbrizy9fw-p2VjvIbTgLJ/pubchart?oid=399094259&format=image)



The file `test.py` includes simple benchmarking code.