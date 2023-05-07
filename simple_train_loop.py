import torch
import torchvision
import torchvision.transforms as transforms
from model.ResNet import ResNet34
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch import autocast
from torch.cuda.amp.grad_scaler import GradScaler

# from https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

def show_image():
    import matplotlib.pyplot as plt
    import numpy as np

    # functions to show an image


    def imshow(img):
        img = img / 2 + 0.5     # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()


    # get some random training images
    dataiter = iter(trainloader)
    images, labels = next(dataiter)

    # show images
    imshow(torchvision.utils.make_grid(images))
    # print labels
    print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))

transform = transforms.Compose(
    [transforms.ToTensor(),
     #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
     ])

batch_size = 4
epoches = 80
device = 'cuda' if torch.cuda.is_available() else 'cpu'

trainset = torchvision.datasets.CIFAR10(root='./data_ssd', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data_ssd', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = ResNet34(10)
#net = Net()
net.to(device)
net.train()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.9)

try:
    scaler = GradScaler()
    for epoch in range(epoches):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            with autocast(device_type='cuda'):
                train_outputs = net(inputs)
                loss = criterion(train_outputs, labels)
            
            scaler.scale(loss).backward()
            #scaler.unscale_(optimizer)
            #torch.nn.utils.clip_grad_norm_(net.parameters(), 10)
            scaler.step(optimizer)
            scaler.update()

            #loss.backward()
            #optimizer.step()


            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0
        scheduler.step()
except KeyboardInterrupt:
    print("Interrupted training")

print('Finished Training')

PATH = './simple_train_model.pth'
torch.save(net.state_dict(), PATH)

correct = 0
total = 0
net.eval()
try:
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)

            with autocast(device_type=device):
                # calculate outputs by running images through the network
                test_outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            #print(torch.sum(test_outputs), torch.sum(images))
            _, predicted = torch.max(test_outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
except KeyboardInterrupt:
    print("Interrupted testing")
print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')


# prepare to count predictions for each class
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

try:
    # again no gradients needed
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            with autocast(device_type=device):
                t_outputs = net(images)
            _, predictions = torch.max(t_outputs, 1)
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1
except KeyboardInterrupt:
    print("Interrupted training")

# print accuracy for each class
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')