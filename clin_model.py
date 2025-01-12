import torch
import torch.nn as nn
from collections import OrderedDict
from video_swin_transformer import SwinTransformer3D
import numpy as np
import pandas as pd
from torch import optim
import pytorch_lightning as pl
import torch.nn.functional as F
import os
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning.loggers import CSVLogger
import gc
import scipy.ndimage as sci
import random
import sklearn.metrics as sk

class ClinNet(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.glu = nn.GLU()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(5,40)
        self.fc2 = nn.Linear(20,60)
        self.fc3 = nn.Linear(60, 20)
        self.fc4 = nn.Linear(20, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.glu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        return x

