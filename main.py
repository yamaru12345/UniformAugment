#!/usr/bin/env python

import argparse
import random
import numpy as np
import pandas as pd
import torch
from torch import optim
from torch import nn
from torch import cuda
from torch.utils.data import sampler
import torchvison

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
parser.add_argument('--num_workers', default=2, type=int)
args = parser.parse_args()

# Loading and normalizing CIFAR10
size = 224
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
transform = ImageTransform(size, mean, std)
trainset = torchvision.datasets.CIFAR10(root=args.base_dir, train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=args.num_workers)

testset = torchvision.datasets.CIFAR10(root=root=args.base_dir, train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=args.num_workers)
