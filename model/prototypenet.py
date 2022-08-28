import torch
import torch.nn as nn


class PrototypeNet(nn.Module):
    def __init__(self, bit, num_classes):
        super(PrototypeNet, self).__init__()

        self.feature = nn.Sequential(nn.Linear(num_classes, 4096),
                                     nn.ReLU(True), nn.Linear(4096, 512))
        self.hashing = nn.Sequential(nn.Linear(512, bit), nn.Tanh())

    def forward(self, label):
        f = self.feature(label)
        h = self.hashing(f)
        return h
