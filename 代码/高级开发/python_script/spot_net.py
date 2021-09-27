import os

import d2l.torch
import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt
from torch import nn

import MyTools
import warnings
from PIL import Image


def try_gpu(i=0):
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


def try_all_gpus():
    devices = [torch.device(f'cuda:{i}')
               for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]


warnings.filterwarnings("ignore")

normalize = torchvision.transforms.Normalize(
    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
)

train_augs = torchvision.transforms.Compose([
    torchvision.transforms.RandomResizedCrop(224),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    normalize
])
test_augs = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    normalize
])

# 初始化网络模型
net = torchvision.models.resnet18(pretrained=True)
net.fc = nn.Linear(net.fc.in_features, 2)
nn.init.xavier_uniform_(net.fc.weight)


def train(net, learning_rate, batch_size=128, num_epochs=10,
          param_group=True):
    train_iter = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder('train_data/train', transform=train_augs), batch_size=batch_size,
        shuffle=True)
    test_iter = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder('train_data/test', transform=test_augs), batch_size=batch_size)
    devices = try_all_gpus()
    loss = nn.CrossEntropyLoss(reduction='none')
    if param_group:
        params = [param for name, param in net.named_parameters() if name not in ['fc.weight', 'fc.bias']]
        trainer = torch.optim.SGD([{'params': params},
                                   {'params': net.fc.parameters(), 'lr': learning_rate * 10}],
                                  lr=learning_rate, weight_decay=0.001)
    else:
        trainer = torch.optim.SGD(net.parameters(), lr=learning_rate, weight_decay=0.001)
    MyTools.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)


def load(image, device):
    image = torch.unsqueeze(image, 0)
    # gpu
    image = image.to(device)
    y_hat = net(image).argmax(axis=1)
    return y_hat

def train_save(name):
    train(net, 5e-5)
    torch.save(net, name)

if __name__ == '__main__':
    train_save('data/MyNet')
