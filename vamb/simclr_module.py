# import math
# from argparse import ArgumentParser
from typing import Callable, Optional, Tuple

import numpy as np
# import pytorch_lightning as pl
import torch
# from torch import nn
# from torch.nn import functional as F

# import torch.distributed as dist


from . import augmentation
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, SequentialSampler
from torch.utils.data.dataset import TensorDataset
from argparse import Namespace
class AugmentationDataModule(object):
# class AugmentationDataModule(pl.LightningDataModule):
    def __init__(self, hparams, ds: Dataset, aug_mode=(None,None)):
        hparams = Namespace(**hparams) if isinstance(hparams, dict) else hparams
        super().__init__()
        self.hparams = hparams
        self.ds = ds
        self.dataset = None
        self.aug_mode = aug_mode

    def prepare_data(self):
        # download
        pass

    def setup(self, stage: Optional[str] = None):

        augment_met = [augmentation.GaussianNoise(), augmentation.NumericalChange(),
                        augmentation.WordDrop(), augmentation.FragmentTransfer(),
                        augmentation.Reverse()]
        # aug_parameters = {'GaussianNoise':{'GaussianNoise':,}}
        aug_method0 = None if self.aug_mode[0] == None else augment_met[self.aug_mode[0]]
        aug_method1 = None if self.aug_mode[1] == None else augment_met[self.aug_mode[1]]
        
        ds_aug0 = self.ds if aug_method0 is None else aug_method0(self.ds,mode=self.aug_mode)
        ds_aug1 = self.ds if aug_method1 is None else aug_method1(self.ds,mode=self.aug_mode)
        # print(aug_method0, aug_method1)
        # print('o ',torch.sum(torch.abs(self.ds)))
        # print('d0 ',torch.sum(torch.pow(torch.sub(ds_aug0,self.ds), 2)))
        # print('d1 ',torch.sum(torch.pow(torch.sub(ds_aug1,self.ds), 2)))
        # print('d',torch.sum(torch.pow(torch.sub(ds_aug0,ds_aug1), 2)))

        self.dataset = TensorDataset(ds_aug0, ds_aug1)

    def train_dataloader(self):
        assert self.dataset is not None
        return DataLoader(dataset=self.dataset,batch_size=self.hparams.batch_size, 
                        #   shuffle=True, 
                          num_workers=0, pin_memory=True if self.hparams.device else False,
                          sampler=SubsetRandomSampler(list(range(self.hparams.train_size))),
                          drop_last=False)

    def val_dataloader(self):
        assert self.dataset is not None
        return DataLoader(dataset=self.dataset, batch_size=self.hparams.validation_size, 
                        # shuffle=True, 
                        num_workers=0, pin_memory=True if self.hparams.device else False,
                        sampler=SubsetRandomSampler(list(range(self.hparams.train_size))),
                        drop_last=True)
    def test_dataloader(self):
        pass

