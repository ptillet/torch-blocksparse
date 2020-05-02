FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-runtime
LABEL maintainer="Alexandr Notchenko <a.notchenko@skoltech.ru>"
USER root

# installing full CUDA toolkit
RUN apt update
RUN pip install --upgrade pip
RUN apt install -y build-essential g++ llvm-8-dev git cmake wget
RUN conda install -y -c conda-forge cudatoolkit-dev
# setting environment variables
ENV CUDA_HOME "/opt/conda/pkgs/cuda-toolkit"
ENV CUDA_TOOLKIT_ROOT_DIR $CUDA_HOME
ENV LIBRARY_PATH "$CUDA_HOME/lib64:$LIBRARY_PATH"
ENV LD_LIBRARY_PATH "$CUDA_HOME/lib64:$CUDA_HOME/extras/CUPTI/lib64:$LD_LIBRARY_PATH"
ENV CFLAGS "-I$CUDA_HOME/include $CFLAGS"
# installing triton
WORKDIR /workspace
RUN pip install --no-cache-dir -e "git+https://github.com/ptillet/triton.git#egg=triton&subdirectory=python"
# installing torch-blocksparse
RUN git clone https://github.com/ptillet/torch-blocksparse.git
WORKDIR /workspace/torch-blocksparse
RUN python setup.py develop
WORKDIR /workspace
# Adding paths to PYTHONPATH
ENV PYTHONPATH "${PYTHONPATH}:/workspace/src/triton/python:/workspace/torch-blocksparse"

ENTRYPOINT ["/bin/bash"]
