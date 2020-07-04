#!/usr/bin/env python

import argparse
import random
import numpy as np
import pandas as pd
from pathlib import Path
import torch
from torch import optim
from torch import nn
from torch import cuda
import torchvision

from uniform_augment import ImageTransform
from model import load_model
from train import train_model
from utils import visualize_logs

random.seed(123)
np.random.seed(123)
torch.manual_seed(123)

parser = argparse.ArgumentParser()
parser.add_argument('base_dir', type=str)
parser.add_argument('model', type=str)
parser.add_argument('--num_epochs', default=100, type=int)
parser.add_argument('--dataset', default='CIFAR10', type=str)
args = parser.parse_args()

# Loading and normalizing CIFAR10
size = 224
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
transform_train = ImageTransform(size, mean, std, train=True)
transform_test = ImageTransform(size, mean, std, train=False)
if args.dataset == 'CIFAR10':
    dataset_train = torchvision.datasets.CIFAR10(root=args.base_dir, train=True, download=True, transform=transform_train)
    dataset_test = torchvision.datasets.CIFAR10(root=args.base_dir, train=False, download=True, transform=transform_test)
elif args.dataset == 'MNIST':
    dataset_train = torchvision.datasets.MNIST(root=args.base_dir, train=True, download=True, transform=transform_train)
    dataset_test = torchvision.datasets.MNIST(root=args.base_dir, train=False, download=True, transform=transform_test)

# Setting parameters
LEARNING_RATE = 1e-3
BATCH_SIZE = 32
device = 'cuda' if cuda.is_available() else 'cpu'
print(f'device: {device}')

# Loading a pretrained model
net = load_model(args.model, 10)

# Defining a loss function
criterion = nn.CrossEntropyLoss()

# Defining an optimizer
optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE, momentum=0.9)

# Training the network
torch.backends.cudnn.benchmark = True
print(f'model: {args.model}')
log = train_model(args.model,
                  dataset_train,
                  dataset_test,
                  BATCH_SIZE,
                  net,
                  criterion,
                  optimizer,
                  args.num_epochs,
                  args.base_dir,
                  device=device)

# Visualizing logs
visualize_logs(log, Path(args.base_dir, f'train_log_{args.model}.png'))
