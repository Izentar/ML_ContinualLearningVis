#!/bin/bash

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install pytorch-lightning==1.6.4 torch-lucent cox==0.1.post3 pandas dill==0.3.5.1 tensorboardX==2.5.1 wandb pytorchcv==0.0.67 rich==12.4.4 matplotlib
pip install --upgrade requests
pip install wandb