import torch
from torchvision.transforms import transforms
import torchvision
import torch.utils as utils
from compressor import DeepCompressor, NET

data_root = "data"


transforms = transforms.Compose([transforms.Resize(32),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),])

dataset = torchvision.datasets.MNIST(root=data_root, transform=transforms, download=True, train=True)
train_data = utils.data.DataLoader(dataset, shuffle=True, batch_size=100, num_workers=2)

test_dataset = torchvision.datasets.MNIST(root=data_root, transform=transforms, download=True, train=False)
test_data = utils.data.DataLoader(test_dataset, shuffle=False, batch_size=100, num_workers=2)


def compress():
    compressor = DeepCompressor("l-lenet.pth", test_data=test_data, train_data=train_data, k=32, lr=0.001)
    model = compressor.train(epoches=10)
    print(model)
    torch.save(model, "DC-lenet.pth")
    net = NET.copy(model)
    torch.save(net, "DC-new-forward-lenet.pth")


compress()