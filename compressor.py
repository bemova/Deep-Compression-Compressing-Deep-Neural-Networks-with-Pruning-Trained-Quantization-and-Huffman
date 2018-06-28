import torch
import torch.optim as optim
from torch.autograd import Variable
from sklearn.cluster import KMeans
import numpy as np
import torch.nn as nn



class DeepCompressor():
    def __init__(self, model_path, test_data, train_data, k, lr):
        self.test_data = test_data
        self.train_data = train_data
        self.model = torch.load(model_path)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.k = k
        self.lr = lr


    def train(self, optimizer=None, epoches=10):
        self.model = self.model.cuda()
        if optimizer is None:
            optimizer = \
                optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)


        for i in range(epoches):
            print("Epoch: ", i)
            self.train_epoch(optimizer, weight_share=True)
        print("Finished fine tuning.")
        return self.update_weights()

    def update_weights(self):
        model = self.model
        for layer, (name, module) in enumerate(model.classifier._modules.items()):
            module.register_backward_hook(self.scalar_quantization)
            weight = module.weight.data.cpu().numpy()
            weight_shape = weight.shape
            centroids = module.centroids
            labels = module.labeled_weight
            new_weight = self.get_finilized_weight(weight=weight, centroids=centroids, labels=labels)
            new_weight = new_weight.reshape(weight_shape[0], weight_shape[1], dtype=np.int8)
            module.weight = torch.from_numpy(new_weight).cuda().int()
            del module.labeled_weight
        return model

    def get_finilized_weight(self, weight, centroids, labels):
        for index, label in enumerate(labels):
            weight[index] = centroids[label][0]
        return weight


    def train_batch(self, optimizer, batch, label, weight_share):
        self.model.zero_grad()
        input = Variable(batch)
        if weight_share:
            output = self.forward(input)
            self.criterion(output, Variable(label)).backward()
        else:
            self.criterion(self.model(input), Variable(label)).backward()
            optimizer.step()

    def train_epoch(self, optimizer=None, weight_share=False):
        index = 1
        for batch_index, (batch, label) in enumerate(self.train_data, 0):
            self.train_batch(optimizer, batch.cuda(), label.cuda(), weight_share)
            if batch_index % 100 == 0:
                print(batch_index)

    def forward(self, x):
        x = self.model.features(x)
        x = x.view(x.size(0), -1)
        for layer, (name, module) in enumerate(self.model.classifier._modules.items()):
            if isinstance(module, torch.nn.modules.Linear):
                module.register_backward_hook(self.scalar_quantization)
                weight = module.weight.data.cpu().numpy()
                weight_shape = weight.shape
                sorted_centroids, centroids, labeled_weight = self.find_centroids(weight, self.k)
                new_weight = self.get_converted_weight(labeled_weight=labeled_weight, centroids=centroids)
                new_weight = new_weight.reshape(weight_shape[0], weight_shape[1])
                module.labeled_weight = labeled_weight
                module.centroids = centroids
                module.weight.data = torch.from_numpy(new_weight).float().cuda()
            x = module(x)
        return x

    def find_centroids(self, weight, num_class):
        a = weight.reshape(-1, 1)
        kmeans = KMeans(n_clusters=num_class, random_state=0).fit(a)
        centroids = kmeans.cluster_centers_
        labels = kmeans.labels_
        sorted_centroids = -np.sort(-centroids, axis=0)
        return sorted_centroids, centroids, labels

    def get_converted_weight(self, labeled_weight, centroids):
        new_weight = np.zeros(shape=labeled_weight.shape, dtype=np.float32)
        for index, label in enumerate(labeled_weight):
            new_weight[index] = centroids[label][0]
        return labeled_weight

    def get_centroids_gradients(self, grad_input, labeled_weight, dw, grad_output):
        w_grad = grad_input[2].t().data.cpu().numpy()
        grad_w = w_grad.reshape(-1, 1)
        for index, label in enumerate(labeled_weight):
            dw[label][0] += grad_w[index]
        return dw

    def scalar_quantization(self, module, grad_input, grad_output):
        if isinstance(module, nn.Linear):
            labeled_weight = module.labeled_weight
            centroids = module.centroids
            dw = np.zeros(shape=centroids.shape, dtype=np.float32)
            dw = self.get_centroids_gradients(grad_input, labeled_weight, dw, grad_output)
            module.centroids = centroids - (self.lr * dw)


class NET(nn.Module):
    def __init__(self):
        super(NET, self).__init__()

    @classmethod
    def copy(cls, source, **kw):
        instance = cls(**kw)
        for name in dir(source):
            if not(name is 'forward' or name.startswith("__")):
                value = getattr(source, name)
                setattr(instance, name, value)
        return instance

    def forward(self, x):
        x = self.model.features(x)
        x = x.view(x.size(0), -1)
        for layer, (name, module) in enumerate(self.model.classifier._modules.items()):
            if isinstance(module, torch.nn.modules.Linear):
                weight_shape = module.weight.shape
                new_weight = self.get_converted_weight(labeled_weight=module.weight, centroids=module.centroids)
                new_weight = new_weight.reshape(weight_shape[0], weight_shape[1])
                new_weight = Variable(torch.from_numpy(new_weight).float().cuda())
                if x.dim() == 2 and module.bias is not None:
                    return torch.addmm(module.bias, x, new_weight.t())
                output = x.matmul(new_weight.t())
                if module.bias is not None:
                    output += module.bias
                x = output
        return x

