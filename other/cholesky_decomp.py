import torch
from torchvision.datasets import CIFAR10
from torchvision import transforms

tr = transforms.Compose(
    [
        transforms.ToTensor(),
        #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ]
)

dataset = CIFAR10("data", train=False, transform=tr)

image_r = []
image_g = []
image_b = []
for image, target in dataset:
    image_r.append(image[0])
    image_g.append(image[1])
    image_b.append(image[2])

image_r = torch.stack(image_r)
image_g = torch.stack(image_g)
image_b = torch.stack(image_b)
a = torch.linalg.cholesky(image_r)
print(a)

torch.corre