import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()

        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)

        self.fc1 = nn.Linear(16*4*4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))

        x = x.view(-1, 16*4*4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x
import torchvision
import torchvision.transforms as transforms

trainset = torchvision.datasets.MNIST(root='./data/minst/', train=True, download=True,
                                        transform=transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.1307,), (0.3081,))
                                        ]))
trainloader = torch.utils.data.DataLoader(trainset, batch_size=40000, shuffle=True)

'''testset = torchvision.datasets.MNIST(root='./data/minst/', train=False, download=False,
                                        transform=transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.1307,), (0.3081,))
                                        ]))
#testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=True)'''

from torch.autograd import Variable
import torch.optim as optim

mnistNet = ConvNet()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(mnistNet.parameters(), lr=0.01, momentum=0.9)

total_loss = []
total_accuracy = []
total, correct = 0, 0
epoch = 50
for m in range(epoch):
    epoch_loss = 0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = Variable(inputs), Variable(labels)

        optimizer.zero_grad()
        outputs = mnistNet(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        A = predicted == labels
        correct += torch.numel(A[A==1])
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    acc = correct/total
    total_loss.append(epoch_loss)
    total_accuracy.append(acc)
    print('epoch=',m,"Loss=",epoch_loss,'Accuracy=',acc)
epoches = np.linspace(1.0,float(epoch),num = epoch)
plt.figure()
plt.plot(epoches,total_loss)
plt.title('loss')
plt.show()
plt.figure()
plt.plot(epoches,total_accuracy)
plt.title('accuracy')
plt.show()