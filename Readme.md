


What frameworks we use:
* Lucid library - https://github.com/greentfrapp/lucent
A library for visualisation of the neural networks. It generates an input image based on target.


source /home/user/.cache/pypoetry/virtualenvs/continual-dreaming-IGEDk-VN-py3.10/bin/activate

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

pip install pytorch-lightning==1.6.4 torch-lucent cox==0.1.post3 pandas dill==0.3.5.1 tensorboardX==2.5.1 wandb pytorchcv==0.0.67 rich==12.4.4 matplotlib

pip install --upgrade requests

wandb server start OR python3 -m wandb server start