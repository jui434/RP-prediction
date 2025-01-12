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
#import scipy.ndimage as sci
import random
import sklearn.metrics as sk
   
#prep CT feature extractor
class CTNet(pl.LightningModule):
    def __init__(self):
        super().__init__()


        self.swin_model = SwinTransformer3D(embed_dim=96,
                          depths=[2, 2, 18, 2],
                          num_heads=[3, 6, 12, 24],
                          patch_size=(2,4,4),
                          window_size=(8,7,7),
                          drop_path_rate=0.1,
                          patch_norm=True)


        #prep extended classification head
        self.pool1 = nn.MaxPool3d((2,2,2), 2) #was 5 3 4
        self.flatten = nn.Flatten()
        self.pool2 = nn.MaxPool1d(3, 2)
        self.fc1 = nn.Linear(21887, 2500)
        self.relu = nn.ReLU()
        self.glu = nn.GLU()
        #self.fc2 = nn.Linear(200, 30)
        self.fc3 = nn.Linear(1250, 100)
        self.fc4 = nn.Linear(100, 1)

    def forward(self, x):
        x = self.swin_model(x)
        gc.collect()
        #apply extended classification head
        x = self.pool1(x)
        x = self.flatten(x)
        x = self.pool2(x)
        x = self.glu(x[:,:-1])
        x = self.fc1(x)
        x = self.relu(x)
        x = self.glu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        gc.collect()

        return x


class CombNet(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.extractor_full = Net().load_from_checkpoint() #load path to feature extractor here
        for param in self.extractor_full.parameters():
            param.requires_grad = False 
        self.extractor = self.extractor_full.swin_model
        self.pool1 = self.extractor_full.pool1
        self.flatten = self.extractor_full.flatten
        self.pool2 = self.extractor_full.pool2
        self.glu = self.extractor_full.glu
        self.fc1 = self.extractor_full.fc1
        self.relu = self.extractor_full.relu
        self.fc3 = self.extractor_full.fc3
        self.fc4 = self.extractor_full.fc4
        gc.collect()

        #prep extended classification head for after addition of clinical variables
        self.l1 = nn.Linear(6, 40)
        self.l2 = nn.Linear(20, 60)
        self.l3 = nn.Linear(60, 20)
        self.l4 = nn.Linear(20, 1) 
        self.sig = nn.Sigmoid() 

    def forward(self, x, clin):
        with torch.no_grad():
            x = self.extractor(x.double())
            gc.collect()
            x = self.pool1(x)
            x = self.flatten(x)
            x = self.pool2(x)
            x = self.glu(x[:,:-1])
            x = self.fc1(x)
            x = self.relu(x)
            x = self.glu(x)
            x = self.fc3(x)
            x = self.relu(x)
            x = self.fc4(x)
        holdscore = self.sig(x)
        cat = torch.cat((clin, holdscore), axis = 1) #add the clinical variables
        x = self.l1(cat)
        x = self.glu(x)
        x = self.l2(x)
        x = self.relu(x)
        x = self.l3(x)
        x = self.relu(x)
        x = self.l4(x)
        return x

