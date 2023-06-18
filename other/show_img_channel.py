import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchvision.utils import save_image

transform = transforms.Compose(
    [transforms.ToTensor(),
     #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
     ])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=False)


it = iter(trainloader)
img, cl = next(it)

print(img.shape)
save_image(img, 'tmp/main.png')
mean = img.mean([0, 2, 3])
var = img.var((0, 2, 3), unbiased=False)

print('mean', mean)
print('var', var)
#plt.imshow(mean.permute(1, 2, 0))
#plt.imshow(var.permute(1, 2, 0))