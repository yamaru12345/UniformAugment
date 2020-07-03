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
args = parser.parse_args()

# Loading and normalizing CIFAR10
size = 224
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
transform_train = ImageTransform(size, mean, std, train=True)
transform_test = ImageTransform(size, mean, std, train=False)
dataset_train = torchvision.datasets.CIFAR10(root=args.base_dir, train=True, download=True, transform=transform_train)
dataset_test = torchvision.datasets.CIFAR10(root=root=args.base_dir, train=False, download=True, transform=transform_test)

# Setting parameters
device = 'cuda' if cuda.is_available() else 'cpu'
print(f'device: {device}')
LEARNING_RATE = 1e-3
BATCH_SIZE = 32

# Loading pretrained model
net = load_model(args.model, 8)
if args.load_model:
    net.load_state_dict(torch.load(Path(args.base_dir, f'state_dict_{args.model}.pt'), map_location=device))

# Defining loss function
criterion = nn.CrossEntropyLoss()

# Defining optimizer
optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE, momentum=0.9)

# Creating sampler
if args.debug:
    sampler = None
else:
    class_sample_count = np.array([len(np.where(train['gender_status'] == t)[0]) for t in np.unique(train['gender_status'])])
    weight = 1. / class_sample_count
    samples_weight = np.array([weight[t] for t in train['gender_status']])
    sampler = sampler.WeightedRandomSampler(samples_weight, len(samples_weight))

# Training model
torch.backends.cudnn.benchmark = True
print(f'model: {args.model}')
log = train_model(args.model,
                  dataset_train,
                  dataset_valid,
                  BATCH_SIZE,
                  net,
                  criterion,
                  optimizer,
                  args.num_epochs,
                  args.base_dir,
                  sampler=sampler,
                  device=device)

# Logging
visualize_logs(log, Path(args.base_dir, f'train_log_{args.model}.png'))
