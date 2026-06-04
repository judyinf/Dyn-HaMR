python -m pip install pip==23.3.2 setuptools==59.5.0 wheel
pip install \
  numpy==1.23.5 \
  scipy==1.10.1 \
  scikit-image==0.21.0 \
  opencv-python==4.8.1.78
pip install ninja cython

pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 \
  --extra-index-url https://download.pytorch.org/whl/cu117

conda install --override-channels -c nvidia/label/cuda-11.7.0 \
  cuda-nvcc \
  cuda-cudart-dev \
  cuda-cccl \
  libcublas-dev \
  libcusolver-dev \
  libcusparse-dev \
  -y

# conda install --override-channels -c nvidia \
#   cuda-toolkit=11.7 \
#   -y
# verify cuda installation
# python - <<'PY'
# import torch, torchvision, os
# from torch.utils.cpp_extension import CUDA_HOME
# print("torch:", torch.__version__)
# print("torch cuda:", torch.version.cuda)
# print("torchvision:", torchvision.__version__)
# print("env CUDA_HOME:", os.environ.get("CUDA_HOME"))
# print("torch CUDA_HOME:", CUDA_HOME)
# PY

# which nvcc
# nvcc --version

pip install --no-cache-dir --no-index torch-scatter \
  -f https://data.pyg.org/whl/torch-1.13.0+cu117.html

pip install --no-build-isolation git+https://github.com/mattloper/chumpy
pip install --no-build-isolation git+https://github.com/nghorbani/configer
pip install --no-build-isolation mmcv==1.3.9

pip install open3d==0.19.0 \
  -i https://mirrors.aliyun.com/pypi/simple \
  --timeout 120 \
  --retries 10 \
  --no-cache-dir

