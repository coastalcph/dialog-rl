from models.statenet import *
import numpy as np
import torch

DIM = 100


class Featurizer:

    def __init__(self):
        self.dimensionality = DIM


fs = {'user': Featurizer(), 'actions': Featurizer(), 'slots': Featurizer()}

s2v = {'tel': ['1', '2', '3'], 'loc': ['a', 'b']}

u = [torch.randn(1, DIM)] * 5
a = torch.randn(1, DIM)
s = torch.randn(1, DIM)

sn = StateNet(fs, 200, 4)

sn.forward_turn(u, a, s2v, 25)

