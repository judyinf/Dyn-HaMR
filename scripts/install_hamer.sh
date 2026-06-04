pip install \
  yacs \
  xtcocotools \
  hydra-core \
  hydra-submitit-launcher \
  hydra-colorlog \
  pyrootutils \
  rich \
  webdataset 

pip install "pytorch-lightning==1.9.5" --no-deps  
pip install "torchmetrics==0.11.4" "lightning-utilities==0.10.1"

pip install -v --no-build-isolation \
  "detectron2@git+https://github.com/facebookresearch/detectron2.git@v0.6"