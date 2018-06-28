import torch
import torch.nn as nn
from torchvision.transforms import transforms
import torchvision
import torch.utils as utils
import torch.optim as optim

data_root = "data"


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.features = nn.Sequential(nn.Conv2d(1, 6, 5),
                                      nn.ReLU(inplace=True),
                                      nn.MaxPool2d(kernel_size=2),
                                      nn.Conv2d(6, 16, 5),
                                      nn.ReLU(inplace=True),
                                      nn.MaxPool2d(kernel_size=2),
                                      )
        self.classifier = nn.Sequential(nn.Linear(16 * 5 * 5, 120),
                                        nn.Dropout(p=0.5),
                                        nn.Linear(120, 84),
                                        nn.Linear(84, 10),
                                        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


transforms = transforms.Compose([transforms.Resize(32),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),])

dataset = torchvision.datasets.MNIST(root=data_root, transform=transforms, download=True, train=True)
train_data = utils.data.DataLoader(dataset, shuffle=True, batch_size=100, num_workers=2)

test_dataset = torchvision.datasets.MNIST(root=data_root, transform=transforms, download=True, train=False)
test_data = utils.data.DataLoader(test_dataset, shuffle=False, batch_size=100, num_workers=2)


def model_test(model, test_data):
    correct = 0
    total = 0
    for (images, labels) in test_data:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()

    accuracy = 100 * correct / total
    print('Accuracy of the network on the test images: %d %%' % (accuracy))
    print('Testing is Done!')
    return accuracy


def train():
    net = Net()
    net.train()

    criterion = nn.CrossEntropyLoss()
    lr = 0.001
    optimizer = optim.SGD(net.parameters(), lr=lr, weight_decay=5e-4)

    for epoch in range(100):
        print("epoch : %d" % (epoch + 1))
        running_loss = 0
        for batch_index, (inputs, target) in enumerate(train_data):
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if batch_index % 1000 == 0:
                print("[%d     , %5d] loss:%.4f" % (epoch + 1, batch_index, running_loss / 1000))

    torch.save(net, "l-lenet.pth")
    model_test(net, test_data=test_data)

train()