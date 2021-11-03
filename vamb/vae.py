import math
from argparse import ArgumentParser
from typing import Callable, Optional, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.core.optimizer import LightningOptimizer
from torch import nn
from torch.nn import functional as F
from torch.optim.optimizer import Optimizer

from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, SequentialSampler
from torch.utils.data.dataset import TensorDataset
from argparse import Namespace
class BatchNormFor1d(nn.BatchNorm1d):
    def _check_input_dim(self, input):
        pass

class VAE(pl.LightningModule):
    """
    Standard VAE with Gaussian Prior and approx posterior.
    """

    def __init__(
        self,
        input_dim: int,
        enc_out_dim: int = 52,
        latent_dim: int = 32,
        kl_coeff: float = 0.1,
        lr: float = 1e-4,
        **kwargs
    ):
        """
        Args:
            input_dim: dim of the imput
            enc_out_dim: set according to the out_channel count of
                encoder used (512 for resnet18, 2048 for resnet50)
            kl_coeff: coefficient for kl term of the loss
            latent_dim: dim of latent space
            lr: learning rate for Adam
        """

        super(VAE, self).__init__()

        self.save_hyperparameters()

        self.lr = lr
        self.kl_coeff = kl_coeff
        self.enc_out_dim = enc_out_dim
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.encoder = nn.Sequential(nn.Linear(self.input_dim, self.enc_out_dim),
                        BatchNormFor1d(self.enc_out_dim),
                        nn.ReLU())

        self.fc_mu = nn.Linear(self.enc_out_dim, self.latent_dim)
        self.fc_var = nn.Linear(self.enc_out_dim, self.latent_dim)

        self.decoder = nn.Sequential(nn.Linear(self.latent_dim, self.enc_out_dim),
                        BatchNormFor1d(self.enc_out_dim),
                        nn.ReLU(),
                        nn.Linear(self.enc_out_dim, self.input_dim, bias=False))

    def forward(self, x):
        x = x.to('cuda')
        self.encoder.cuda()
        self.fc_mu.cuda()
        self.fc_var.cuda()
        self.decoder.cuda()
        # check the device for model and data
        # print(next(self.encoder.parameters()).is_cuda)
        # print(x.device)
        x = self.encoder(x)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        p, q, z = self.sample(mu, log_var)
        return self.decoder(z)

    def _run_step(self, x):
        self.encoder.cuda()
        self.fc_mu.cuda()
        self.fc_var.cuda()
        self.decoder.cuda()
        x = self.encoder(x)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        p, q, z = self.sample(mu, log_var)
        return z, self.decoder(z), p, q

    def sample(self, mu, log_var):
        std = torch.exp(log_var / 2)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()
        return p, q, z

    def step(self, batch, batch_idx):
        x = batch
        x = x.to('cuda')
        z, x_hat, p, q = self._run_step(x)

        recon_loss = F.mse_loss(x_hat, x, reduction='mean')

        log_qz = q.log_prob(z)
        log_pz = p.log_prob(z)

        kl = log_qz - log_pz
        kl = kl.mean()
        kl *= self.kl_coeff

        loss = kl + recon_loss

        logs = {
            "recon_loss": recon_loss,
            "kl": kl,
            "loss": loss,
        }
        return loss, logs

    def training_step(self, batch, batch_idx):
        loss, logs = self.step(batch, 0)
        self.log_dict({f"train_{k}": v for k, v in logs.items()}, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logs = self.step(batch, 0)
        self.log_dict({f"val_{k}": v for k, v in logs.items()})
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

class VaeDataModule(pl.LightningDataModule):
    def __init__(self, hparams, ds):
        hparams = Namespace(**hparams) if isinstance(hparams, dict) else hparams
        super().__init__()
        self.hparams = hparams
        self.dataset = ds

    def prepare_data(self):
        # download
        pass

    def setup(self, stage: Optional[str] = None):
        pass

    def train_dataloader(self):
        assert self.dataset is not None
        return DataLoader(dataset=self.dataset,batch_size=self.hparams.batch_size, 
                          shuffle=False, num_workers=2,
                        #   sampler=SubsetRandomSampler(list(range(self.hparams.train_size))),
                          drop_last=False)

    def val_dataloader(self):
        assert self.dataset is not None
        return DataLoader(dataset=self.dataset, batch_size=self.hparams.validation_size, 
                        shuffle=False, num_workers=0,
                        sampler=SequentialSampler(list(range(self.hparams.validation_size))),
                        drop_last=True)

    def test_dataloader(self):
        pass

def cli_main():
# my-vae
    # hparams2 = Namespace(
    #     lr=1e-2,
    #     epochs=nepochs,
    #     patience=5,
    #     num_nodes=1,
    #     gpus=0 if device=='cpu' else 1,
    #     batch_size=8192,
    #     train_size=len(mask) - n_discarded,
    #     validation_size=2560,
    #     hidden_mlp=103, # input_dim
    #     enc_out_dim=98,
    #     latent_dim=256,
    #     device=device
    # )
    # prelatent = prelatent.cpu()

    # vaemodel = vae.VAE(input_dim=hparams2.hidden_mlp,enc_out_dim=hparams2.enc_out_dim,latent_dim=hparams2.latent_dim).to(device)
    # vaedm = vae.VaeDataModule(hparams2,prelatent)
    # logger = WandbLogger(project="vae-blogpost")
    # logger.watch(vaemodel, log="all", log_freq=50)
    # model_checkpoint = ModelCheckpoint(save_last=True, save_top_k=1, monitor='val_loss')
    # early_stop = EarlyStopping(monitor='val_loss',min_delta=hparams2.lr,patience=hparams2.patience,mode='min')
    # callbacks = [model_checkpoint,early_stop]
    # trainer2 = pl.Trainer(
    #     max_epochs=hparams2.epochs,
    #     max_steps=None,
    #     gpus=hparams2.gpus,
    #     num_nodes=hparams2.num_nodes,
    #     distributed_backend='ddp' if hparams2.gpus > 1 else None,
    #     sync_batchnorm=True if hparams2.gpus > 1 else False,
    #     precision=32,
    #     callbacks=callbacks,
    #     auto_lr_find=True,
    #     logger=logger
    # )
    # lr_finder2 = trainer2.tuner.lr_find(vaemodel, datamodule=vaedm)
    # vaemodel.start_lr = hparams2.lr = lr_finder2.suggestion()
    # vaemodel.final_lr = vaemodel.start_lr * 1e-2
    # trainer2.fit(vaemodel, datamodule=vaedm)

    # model_file = os.path.join(outdir.replace('results','data'), "vae.pt")
    # # torch.save(vaemodel.state_dict(),model_file)
    # vae_new_model=vae.VAE(input_dim=hparams2.hidden_mlp,enc_out_dim=hparams2.enc_out_dim,latent_dim=hparams2.latent_dim).to(device)
    # vae_new_model.to(device)
    # vae_new_model.load_state_dict(torch.load(model_file))
    # dataloader = _DataLoader(dataset=_TensorDataset(prelatent), batch_size=hparams2.batch_size, drop_last=False,
    #                          shuffle=False, num_workers=3)
    # i = 0
    # with torch.no_grad():
    #     for idx, batch in enumerate(dataloader):
    #         c = batch[0].cuda()
    #         m = vae_new_model(c)
    #         if i == 0:
    #             i = i + 1
    #             latent = m.clone().detach()
    #         else:
    #             latent = torch.cat((latent,m),0)
    # del prelatent
    # print(latent.shape)
    # latent = latent.cpu().numpy()
    pass