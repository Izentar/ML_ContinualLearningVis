from model.vgg import VGG19_BN
from utils.data_manipulation import count_parameters

if(__name__ == '__main__'):
    model = VGG19_BN(num_classes=10, num_tasks=5)
    print(model)
    print("Model params:", count_parameters(model))