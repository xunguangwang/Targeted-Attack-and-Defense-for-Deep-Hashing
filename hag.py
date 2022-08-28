import os
os.environ["CUDA_VISIBLE_DEVICES"]='0'
import torch
import time

import numpy as np
import torch.optim as optim
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import models
from datetime import datetime

from utils.data_provider import *
from utils.hamming_matching import *


def load_model(path):
    model = torch.load(path)
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()
    return model


def adv_loss(clean_output, noisy_output, t=0.5):
    w = (torch.abs(noisy_output - torch.sign(clean_output))<(1+t)).int()
    m = w.sum()
    loss = clean_output*noisy_output
    loss = loss.sum() / m + 1
    loss = loss**2

    return loss


def get_alpha(n):
    if n < 1000:
        return 0.1
    elif n >= 1000 and n < 1200:
        return 0.2
    elif n >= 1200 and n < 1400:
        return 0.3
    elif n >= 1400 and n < 1600:
        return 0.5
    elif n >= 1600 and n < 1800:
        return 0.7
    else:
        return 1


def hash_adv(model, query, epsilon, step=1, iteration=2000, randomize=False):
    if iteration < 1:
        return query

    delta = torch.zeros_like(query).cuda()
    if randomize:
        delta.uniform_(-epsilon, epsilon)
        delta.data = (query.data + delta.data).clamp(0, 1) - query.data
    delta.requires_grad = True

    clean_output = model(query).detach()
    clean_output = torch.tanh(clean_output)

    for i in range(iteration):
        alpha = get_alpha(i)
        # noisy_output = model(query+delta, alpha)
        noisy_output = model(query+delta)
        noisy_output = torch.tanh(noisy_output)
        loss = adv_loss(clean_output, noisy_output)
        loss.backward(retain_graph=True)

        # delta.data = delta - step * delta.grad.detach() / torch.norm(delta.grad.detach(), 2)
        # delta.data = delta - step / 255 * torch.sign(delta.grad.detach())
        delta.data = delta - step * (delta.grad.detach())
        delta.data = delta.data.clamp(-epsilon, epsilon)
        delta.data = (query.data + delta.data).clamp(0, 1) - query.data
        delta.grad.zero_()

    delta = delta.detach()
    return query + delta


def GenerateCode(model, data_loader, num_data, bit, use_gpu=True):
    B = np.zeros([num_data, bit], dtype=np.float32)
    for iter, data in enumerate(data_loader, 0):
        data_input, _, data_ind = data
        if use_gpu:
            data_input = Variable(data_input.cuda())
        else: data_input = Variable(data_input)
        output = model(data_input)
        if use_gpu:
            B[data_ind.numpy(), :] = torch.sign(output.cpu().data).numpy()
        else:
            B[data_ind.numpy(), :] = torch.sign(output.data).numpy()
    return B


def generate_hash(model, samples, num_data, bit):
    output = model(samples)
    B = torch.sign(output.cpu().data).numpy()
    return B


dataset = 'NUS-WIDE'
DATA_DIR = '../data/{}'.format(dataset)
DATABASE_FILE = 'database_img.txt'
TEST_FILE = 'test_img.txt'
DATABASE_LABEL = 'database_label.txt'
TEST_LABEL = 'test_label.txt'

epsilon = 8
epsilon = epsilon / 255.
iteration = 0
method = 'HAG'

bit = 32
batch_size = 32
model_name = 'DPH'
backbone = 'VGG11'

model_path = 'checkpoint/adv_{}_{}_{}_{}.pth'.format(dataset, model_name, backbone, bit)
model = load_model(model_path)
database_code_path = 'log/adv_database_code_{}_{}_{}_{}.txt'.format(dataset, model_name, backbone, bit)
test_code_path = 'log/test_code_{}_{}_{}.txt'.format(dataset, method, bit)


# data processing
dset_database = HashingDataset(DATA_DIR, DATABASE_FILE, DATABASE_LABEL)
dset_test = HashingDataset(DATA_DIR, TEST_FILE, TEST_LABEL)
database_loader = DataLoader(dset_database, batch_size=batch_size, shuffle=False, num_workers=8)
test_loader = DataLoader(dset_test, batch_size=batch_size, shuffle=False, num_workers=8)
num_database, num_test = len(dset_database), len(dset_test)


if os.path.exists(database_code_path):
    database_hash = np.loadtxt(database_code_path, dtype=float)
else:
    database_hash = GenerateCode(model, database_loader, num_database, bit)
    np.savetxt(database_code_path, database_hash, fmt="%d")
print('database hash codes prepared!')
# database_hash = GenerateCode(model, database_loader, num_database, bit)


test_labels_int = np.loadtxt(os.path.join(DATA_DIR, TEST_LABEL), dtype=int)
database_labels_int = np.loadtxt(os.path.join(DATA_DIR, DATABASE_LABEL), dtype=int)


qB = np.zeros([num_test, bit], dtype=np.float32)
for it, data in enumerate(test_loader):
    query, _, index = data
    query = query.cuda()
    batch_size_ = index.size(0)
    print(index[-1].item())

    query_adv = hash_adv(model, query, epsilon, iteration=iteration, randomize=True)
    u_ind = np.linspace(it * batch_size,
                        np.min((num_test, (it + 1) * batch_size)) - 1,
                        batch_size_,
                        dtype=int)
    query_code = generate_hash(model, query_adv, batch_size_, bit)
    qB[u_ind, :] = query_code

# qB = -qB

print(qB.shape)
np.savetxt(test_code_path, qB, fmt="%d")
map = CalcTopMap(qB, database_hash, test_labels_int, database_labels_int, 5000)
print('[Retrieval Phase] MAP(retrieval database): %3.5f' % map)
map = CalcMap(qB, database_hash, test_labels_int, database_labels_int)
print('[Retrieval Phase] MAP(retrieval database): %3.5f' % map)
