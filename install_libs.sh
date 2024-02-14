#!/bin/bash
set -e

python3 -m venv pythonEnv
source pythonEnv/bin/activate

python3 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
python3 -m pip install pytorch-lightning==1.6.4 torch-lucent cox==0.1.post3 pandas dill==0.3.5.1 tensorboardX==2.5.1 wandb pytorchcv==0.0.67 rich==12.4.4 matplotlib colorama
python3 -m pip install --upgrade requests
python3 -m pip install wandb