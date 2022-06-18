__doc__ = """Encode a depths matrix and a tnf matrix to latent representation.

Creates a variational autoencoder in PyTorch and tries to represent the depths
and tnf in the latent space under gaussian noise.

Usage:
>>> vae = VAE(nsamples=6)
>>> dataloader, mask = make_dataloader(depths, tnf)
>>> vae.trainmodel(dataloader)
>>> latent = vae.encode(dataloader) # Encode to latent representation
>>> latent.shape
(183882, 32)
"""

__cmd_doc__ = """Encode depths and TNF using a VAE to latent representation"""

import os
import random
import glob
import numpy as _np
import torch as _torch
_torch.manual_seed(42)
#_torch.seed()

import math
import pickle
from argparse import Namespace

from torch import nn as _nn
from torch.optim import Adam, SGD
from . import lars
from torch.nn.functional import softmax as _softmax
from torch.utils.data import DataLoader as _DataLoader
from torch.utils.data.dataset import TensorDataset as _TensorDataset
from torch.utils.data import SubsetRandomSampler, BatchSampler
import torch.nn.functional as F
import vamb.vambtools as _vambtools

if _torch.__version__ < '0.4':
    raise ImportError('PyTorch version must be 0.4 or newer')

def make_dataloader(rpkm, tnf, batchsize=256, destroy=False, cuda=False, contrastive=True):
    """Create a DataLoader and a contig mask from RPKM and TNF.

    The dataloader is an object feeding minibatches of contigs to the VAE.
    The data are normalized versions of the input datasets, with zero-contigs,
    i.e. contigs where a row in either TNF or RPKM are all zeros, removed.
    The mask is a boolean mask designating which contigs have been kept.

    Inputs:
        rpkm: RPKM matrix (N_contigs x N_samples)
        tnf: TNF matrix (N_contigs x N_TNF)
        batchsize: Starting size of minibatches for dataloader
        destroy: Mutate rpkm and tnf array in-place instead of making a copy.
        cuda: Pagelock memory of dataloader (use when using GPU acceleration)

    Outputs:
        DataLoader: An object feeding data to the VAE
        mask: A boolean mask of which contigs are kept
    """
    if not isinstance(rpkm, _np.ndarray) or not isinstance(tnf, _np.ndarray):
        raise ValueError('TNF and RPKM must be Numpy arrays')

    if batchsize < 1:
        raise ValueError('Minimum batchsize of 1, not {}'.format(batchsize))

    if len(rpkm) != len(tnf):
        raise ValueError(f'Lengths of RPKM:{len(rpkm)} and TNF:{ len(tnf)} must be the same')

    if not (rpkm.dtype == tnf.dtype == _np.float32):
        raise ValueError('TNF and RPKM must be Numpy arrays of dtype float32')

    mask = tnf.sum(axis=1) != 0

    # If multiple samples, also include nonzero depth as requirement for accept
    # of sequences
    if rpkm.shape[1] > 1:
        depthssum = rpkm.sum(axis=1)
        mask &= depthssum != 0
        depthssum = depthssum[mask]

#    if mask.sum() < batchsize:
#        raise ValueError('Fewer sequences left after filtering than the batch size.')

    if destroy:
        rpkm = _vambtools.numpy_inplace_maskarray(rpkm, mask)
        tnf = _vambtools.numpy_inplace_maskarray(tnf, mask)
    else:
        # The astype operation does not copy due to "copy=False", but the masking
        # operation does.
        rpkm = rpkm[mask].astype(_np.float32, copy=False)
        tnf = tnf[mask].astype(_np.float32, copy=False)

    # If multiple samples, normalize to sum to 1, else zscore normalize
    if rpkm.shape[1] > 1:
        rpkm /= depthssum.reshape((-1, 1))
    else:
        _vambtools.zscore(rpkm, axis=0, inplace=True)

    # Normalize arrays and create the Tensors (the tensors share the underlying memory)
    # of the Numpy arrays
    _vambtools.zscore(tnf, axis=0, inplace=True)
    depthstensor = _torch.from_numpy(rpkm)
    tnftensor = _torch.from_numpy(tnf)
    # cattensor = _torch.cat((tnftensor,depthstensor),1)
    # dataset = _TensorDataset(cattensor)

    # Create dataloader
    n_workers = 0
    dataset = _TensorDataset(depthstensor, tnftensor)
    dataloader = _DataLoader(dataset=dataset, batch_size=batchsize, drop_last=False,
                             shuffle=False, num_workers=n_workers, pin_memory=cuda)

    return dataloader, mask

# Automatic Loss training for multi-task learning
# Copied from https://github.com/Mikoto10032/AutomaticWeightedLoss
class AutomaticWeightedLoss(_nn.Module):
    """automatically weighted multi-task loss
    Params：
        num: int，the number of loss
        x: multi-task loss
    Examples：
        loss1=1
        loss2=2
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl(loss1, loss2)
    """
    def __init__(self, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        params = _torch.ones(num, requires_grad=True)
        self.params = _torch.nn.Parameter(params)

    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            #print(self.params[i].retain_grad(), self.params[i])
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + _torch.log(1 + self.params[i] ** 2)
        print('loss_sum',loss_sum)
        return loss_sum

class VAE(_nn.Module):
    """Variational autoencoder, subclass of torch.nn.Module.
    Instantiate with:
        nsamples: Number of samples in abundance matrix
        nhiddens: List of n_neurons in the hidden layers [None=Auto]
        nlatent: Number of neurons in the latent layer [32]
        alpha: Approximate starting TNF/(CE+TNF) ratio in loss. [None = Auto]
        beta: Multiply KLD by the inverse of this value [200]
        dropout: Probability of dropout on forward pass [0.2]
        cuda: Use CUDA (GPU accelerated training) [False]
    vae.trainmodel(dataloader, nepochs batchsteps, lrate, logfile, modelfile)
        Trains the model, returning None
    vae.encode(self, data_loader):
        Encodes the data in the data loader and returns the encoded matrix.
    If alpha or dropout is None and there is only one sample, they are set to
    0.99 and 0.0, respectively
    """

    def __init__(self, ntnf, nsamples, k=4, nhiddens=None, nlatent=32, alpha=None,
                 beta=200, dropout=0.2, cuda=False, c=False):
        if nlatent < 1:
            raise ValueError('Minimum 1 latent neuron, not {}'.format(nlatent))

        if nsamples < 1:
            raise ValueError('nsamples must be > 0, not {}'.format(nsamples))

        # If only 1 sample, we weigh alpha and nhiddens differently
        if alpha is None:
            alpha = 0.15 if nsamples > 1 else 0.50

        if nhiddens is None:
            nhiddens = [512, 512] if nsamples > 1 else [256, 256]

        if dropout is None:
            dropout = 0.0 if nsamples > 1 else 0.0

        if any(i < 1 for i in nhiddens):
            raise ValueError('Minimum 1 neuron per layer, not {}'.format(min(nhiddens)))

        if beta <= 0:
            raise ValueError('beta must be > 0, not {}'.format(beta))

        if not (0 < alpha < 1):
            raise ValueError('alpha must be 0 < alpha < 1, not {}'.format(alpha))

        if not (0 <= dropout < 1):
            raise ValueError('dropout must be 0 <= dropout < 1, not {}'.format(dropout))

        super(VAE, self).__init__()

        # Initialize simple attributes
        self.usecuda = cuda
        self.nsamples = nsamples
        self.ntnf = ntnf
        self.k = k
        self.alpha = alpha
        self.beta = beta
        self.nhiddens = nhiddens
        self.nlatent = nlatent
        self.dropout = dropout
        self.contrast = c

        # Initialize lists for holding hidden layers
        self.encoderlayers = _nn.ModuleList()
        self.encodernorms = _nn.ModuleList()
        self.decoderlayers = _nn.ModuleList()
        self.decodernorms = _nn.ModuleList()

        # Add all other hidden layers
        for nin, nout in zip([self.nsamples + self.ntnf] + self.nhiddens, self.nhiddens):
            self.encoderlayers.append(_nn.Linear(nin, nout))
            self.encodernorms.append(_nn.BatchNorm1d(nout))

        # Latent layers
        self.mu = _nn.Linear(self.nhiddens[-1], self.nlatent)
        self.logsigma = _nn.Linear(self.nhiddens[-1], self.nlatent)

        # Add first decoding layer
        for nin, nout in zip([self.nlatent] + self.nhiddens[::-1], self.nhiddens[::-1]):
            self.decoderlayers.append(_nn.Linear(nin, nout))
            self.decodernorms.append(_nn.BatchNorm1d(nout))

        # Reconstruction (output) layer
        self.outputlayer = _nn.Linear(self.nhiddens[0], self.nsamples + self.ntnf)

        # Activation functions
        self.relu = _nn.LeakyReLU()
        self.softplus = _nn.Softplus()
        self.dropoutlayer = _nn.Dropout(p=self.dropout)

        # Hidden layer monitor
        # gradient monitor using hook (require more memory and time cost)
        # self.register_full_backward_hook(backward_hook)
        # self.register_forward_hook(forward_hook)
        # for param in self.parameters():
        #     print('self',type(param), param.size())

        if cuda:
            self.cuda()

    def _encode(self, tensor):
        #initial_tensor = tensor.clone()
        tensors = list()

        # Hidden layers
        for encoderlayer, encodernorm in zip(self.encoderlayers, self.encodernorms):
            tensor = encodernorm(self.dropoutlayer(self.relu(encoderlayer(tensor))))
            #tensor = encodernorm(self.relu(encoderlayer(tensor)))
            tensors.append(tensor)
        #print('test_tensors',_torch.sum(tensors[0]-tensors[1]))
        # Latent layers
        #mu = self.mu(tensors[0]+tensor)
        mu = self.mu(tensor)
        # Note: This softplus constrains logsigma to positive. As reconstruction loss pushes
        # logsigma as low as possible, and KLD pushes it towards 0, the optimizer will
        # always push this to 0,meaning that the logsigma layer will be pushed towards
        # negative infinity. This creates a nasty numerical instability in VAMB. Luckily,
        # the gradient also disappears as it decreases towards negative infinity, avoiding
        # NaN poisoning in most cases. We tried to remove the softplus layer, but this
        # necessitates a new round of hyperparameter optimization, and there is no way in
        # hell I am going to do that at the moment of writing.
        # Also remove needless factor 2 in definition of latent in reparameterize function.
        #logsigma = self.softplus(self.logsigma(tensors[0]+tensor))
        logsigma = self.softplus(self.logsigma(tensor))
        return mu, logsigma

    # sample with gaussian noise
    def reparameterize(self, mu, logsigma):
        epsilon = _torch.randn(mu.size(0), mu.size(1))

        if self.usecuda:
            epsilon = epsilon.cuda()

        epsilon.requires_grad = True

        # See comment above regarding softplus
        latent = mu + epsilon * _torch.exp(logsigma/2)

        return latent

    def _decode(self, tensor):
#        initial_tensor = tensor.clone()
        tensors = list()

        for decoderlayer, decodernorm in zip(self.decoderlayers, self.decodernorms):
            tensor = decodernorm(self.dropoutlayer(self.relu(decoderlayer(tensor))))
            #tensor = decodernorm(self.relu(decoderlayer(tensor)))
            tensors.append(tensor)

   #     reconstruction = self.outputlayer(tensors[0]+tensor)
        reconstruction = self.outputlayer(tensor)
        # Decompose reconstruction to depths and tnf signal
        depths_out = reconstruction.narrow(1, 0, self.nsamples)
        tnf_out = reconstruction.narrow(1, self.nsamples, self.ntnf)

        # If multiple samples, apply softmax
        if self.nsamples > 1:
            depths_out = _softmax(depths_out, dim=1)

        return depths_out, tnf_out

    def forward(self, depths, tnf):
        tensor = _torch.cat((depths, tnf), 1)
        mu, logsigma = self._encode(tensor)
        latent = self.reparameterize(mu, logsigma)
        depths_out, tnf_out = self._decode(latent)

        return depths_out, tnf_out, mu, logsigma

    def calc_loss(self, depths_in, depths_out, tnf_in, tnf_out, mu, logsigma):
        # If multiple samples, use cross entropy, else use SSE for abundance
        if self.nsamples > 1:
            # Add 1e-9 to depths_out to avoid numerical instability.
            ce = - ((depths_out + 1e-9).log() * depths_in).sum(dim=1).mean()
            ce_weight = (1 - self.alpha) / math.log(self.nsamples)
        else:
            ce = (depths_out - depths_in).pow(2).sum(dim=1).mean()
            ce_weight = 1 - self.alpha

        sse = (tnf_out - tnf_in).pow(2).sum(dim=1).mean()
        kld = -0.5 * (1 + logsigma - mu.pow(2) - logsigma.exp()).sum(dim=1).mean()
        sse_weight = self.alpha / self.ntnf
        kld_weight = 1 / (self.nlatent * self.beta)
        loss = ce * ce_weight + sse * sse_weight + kld * kld_weight

        return loss, ce*ce_weight, sse*sse_weight, kld*kld_weight

    def trainepoch(self, data_loader, epoch, optimizer, batchsteps, logfile, hparams, awl=None):
        self.train()
# VAMB
        if hparams==Namespace():
            epoch_loss = 0
            epoch_kldloss = 0
            epoch_sseloss = 0
            epoch_celoss = 0

            for depths_in, tnf_in in data_loader:
                depths_in.requires_grad = True
                tnf_in.requires_grad = True

                if self.usecuda:
                    depths_in = depths_in.cuda()
                    tnf_in = tnf_in.cuda()

                optimizer.zero_grad()

                depths_out, tnf_out, mu, logsigma = self(depths_in, tnf_in)

                loss, ce, sse, kld = self.calc_loss(depths_in, depths_out, tnf_in,
                                                    tnf_out, mu, logsigma)

                loss.backward()
                optimizer.step()

                epoch_loss += loss.data.item()
                epoch_kldloss += kld.data.item()
                epoch_sseloss += sse.data.item()
                epoch_celoss += ce.data.item()

            if logfile is not None:
                print('\tEpoch: {}\tLoss: {:.6f}\tCE: {:.7f}\tSSE: {:.6f}\tKLD: {:.4f}\tBatchsize: {}'.format(
                    epoch + 1,
                    epoch_loss / len(data_loader),
                    epoch_celoss / len(data_loader),
                    epoch_sseloss / len(data_loader),
                    epoch_kldloss / len(data_loader),
                    data_loader.batch_size,
                    ), file=logfile)

                logfile.flush()
    # simclr
        else:
            epoch_loss = 0
            epoch_kldloss = 0
            epoch_cesseloss = 0
            epoch_clloss = 0
            # grad_block.clear()
            for depths, tnf_in, tnf_aug1, tnf_aug2 in data_loader:
                # print(_torch.sum(tnf_in1),tnf_in1.shape, file=logfile)
                # depths_in1, tnf_in1, depths_in2, tnf_in2 = depths_in1[0], tnf_in1[0], depths_in2[0], tnf_in2[0]
                depths.requires_grad = True
                tnf_in.requires_grad = True
                tnf_aug1.requires_grad = True
                tnf_aug2.requires_grad = True

                if self.usecuda:
                    depths = depths.cuda()
                    tnf_in = tnf_in.cuda()
                    tnf_aug1 = tnf_aug1.cuda()
                    tnf_aug2 = tnf_aug2.cuda()

                optimizer.zero_grad()

                depths_out, tnf_out, mu, logsigma = self(depths, tnf_in)
                depths_out1, tnf_out_aug1, mu1, logsigma1 = self(depths, tnf_aug1)
                depths_out2, tnf_out_aug2, mu2, logsigma2 = self(depths, tnf_aug2)

                #loss3 = self.nt_xent_loss(_torch.cat((depths_out1, tnf_out1), 1), _torch.cat((depths_out2, tnf_out2), 1), temperature=hparams.temperature)
                loss_contrast1 = self.nt_xent_loss(tnf_out_aug1, tnf_out_aug2, temperature=hparams.temperature)
                loss_contrast2 = self.nt_xent_loss(tnf_out_aug2, tnf_out, temperature=hparams.temperature)
                loss_contrast3 = self.nt_xent_loss(tnf_out, tnf_out_aug1, temperature=hparams.temperature)
                loss1, ce1, sse1, kld1 = self.calc_loss(depths, depths_out, tnf_in, tnf_out, mu, logsigma)

                # Add weight to avoid gradient disappearance
                loss = awl(100*loss_contrast1, 100*loss_contrast2, 100*loss_contrast3) + 10000*loss1
                loss.backward()
                optimizer.step()
                print('loss',loss1,loss_contrast1,loss_contrast2,loss_contrast3,file=logfile)

                epoch_loss += loss.data.item()
                epoch_kldloss += (kld1).data.item()
                epoch_cesseloss += (ce1).data.item()
                epoch_clloss += (sse1).data.item()

        # Gradient monitor using hook (require more memory and time cost)
        #    for i in range(len(grad_block)):
        #        print('grad', grad_block[i], file=logfile, end='\t\t')

            if logfile is not None:
                print('\tEpoch: {}\tLoss: {:.6f}\tCL: {:.7f}\tCE SSE: {:.6f}\tKLD: {:.4f}\tBatchsize: {}'.format(
                    epoch + 1,
                    epoch_loss / len(data_loader),
                    epoch_clloss / len(data_loader),
                    epoch_cesseloss / len(data_loader),
                    epoch_kldloss / len(data_loader),
                    data_loader.batch_size,
                    ), file=logfile)

                logfile.flush()

        return None

    def encode(self, data_loader):
        """Encode a data loader to a latent representation with VAE
        Input: data_loader: As generated by train_vae
        Output: A (n_contigs x n_latent) Numpy array of latent repr.
        """

        self.eval()

        new_data_loader = _DataLoader(dataset=data_loader.dataset,
                                      batch_size=data_loader.batch_size,
                                      shuffle=False,
                                      drop_last=False,
                                      num_workers=0,
                                      pin_memory=data_loader.pin_memory)

        depths_array, tnf_array = data_loader.dataset.tensors
        length = len(depths_array)

        # We make a Numpy array instead of a Torch array because, if we create
        # a Torch array, then convert it to Numpy, Numpy will believe it doesn't
        # own the memory block, and array resizes will not be permitted.
        latent = _np.empty((length, self.nlatent), dtype=_np.float32)

        row = 0
        with _torch.no_grad():
            for depths, tnf in new_data_loader:
                # Move input to GPU if requested
                if self.usecuda:
                    depths = depths.cuda()
                    tnf = tnf.cuda()

                # Evaluate
                out_depths, out_tnf, mu, logsigma = self(depths, tnf)

                if self.usecuda:
                    mu = mu.cpu()

                latent[row: row + len(mu)] = mu
                row += len(mu)

        assert row == length
        return latent

    def save(self, filehandle):
        """Saves the VAE to a path or binary opened file. Load with VAE.load
        Input: Path or binary opened filehandle
        Output: None
        """
        state = {'nsamples': self.nsamples,
                 'alpha': self.alpha,
                 'beta': self.beta,
                 'dropout': self.dropout,
                 'nhiddens': self.nhiddens,
                 'nlatent': self.nlatent,
                 'state': self.state_dict(),
                }

        _torch.save(state, filehandle)

    @classmethod
    def load(cls, path, cuda=False, evaluate=True, c=False):
        """Instantiates a VAE from a model file.
        Inputs:
            path: Path to model file as created by functions VAE.save or
                  VAE.trainmodel.
            cuda: If network should work on GPU [False]
            evaluate: Return network in evaluation mode [True]
        Output: VAE with weights and parameters matching the saved network.
        """

        # Forcably load to CPU even if model was saves as GPU model
        # dictionary = _torch.load(path, map_location=lambda storage, loc: storage)
        dictionary = _torch.load(path)
        nsamples = dictionary['nsamples']
        alpha = dictionary['alpha']
        beta = dictionary['beta']
        dropout = dictionary['dropout']
        nhiddens = dictionary['nhiddens']
        nlatent = dictionary['nlatent']
        state = dictionary['state']

        vae = cls(nsamples, nhiddens, nlatent, alpha, beta, dropout, cuda, c=c)
        vae.load_state_dict(state)

        if cuda:
            vae.cuda()

        if evaluate:
            vae.eval()

        return vae

    def trainmodel(self, dataloader, nepochs=500, lrate=1e-3,
                   batchsteps=[25, 75, 150, 300], logfile=None, modelfile=None, hparams=None, augmentationpath=None, augdatashuffle=False):
        """Train the autoencoder from depths array and tnf array.
        Inputs:
            dataloader: DataLoader made by make_dataloader
            nepochs: Train for this many epochs before encoding [500]
            lrate: Starting learning rate for the optimizer [0.001]
            batchsteps: None or double batchsize at these epochs [25, 75, 150, 300]
            logfile: Print status updates to this file if not None [None]
            modelfile: Save models to this file if not None [None]
        Output: None
        """

        if lrate < 0:
            raise ValueError('Learning rate must be positive, not {}'.format(lrate))

        if nepochs < 1:
            raise ValueError('Minimum 1 epoch, not {}'.format(nepochs))

        if batchsteps is None:
            batchsteps_set = set()
        else:
            # First collect to list in order to allow all element types, then check that
            # they are integers
            batchsteps = list(batchsteps)
            if not all(isinstance(i, int) for i in batchsteps):
                raise ValueError('All elements of batchsteps must be integers')
            if max(batchsteps, default=0) >= nepochs:
                raise ValueError('Max batchsteps must not equal or exceed nepochs')
            last_batchsize = dataloader.batch_size * 2**len(batchsteps)
            #if len(dataloader.dataset) < last_batchsize:
            #    raise ValueError('Last batch size exceeds dataset length')
            batchsteps_set = set(batchsteps)

        # Get number of features
        depthstensor, tnftensor = dataloader.dataset.tensors
        ncontigs, nsamples = depthstensor.shape

        if logfile is not None:
            print('\tNetwork properties:', file=logfile)
            print('\tCUDA:', self.usecuda, file=logfile)
            print('\tAlpha:', self.alpha, file=logfile)
            print('\tBeta:', self.beta, file=logfile)
            print('\tDropout:', self.dropout, file=logfile)
            print('\tN hidden:', ', '.join(map(str, self.nhiddens)), file=logfile)
            print('\tN latent:', self.nlatent, file=logfile)
            print('\n\tTraining properties:', file=logfile)
            print('\tN epochs:', nepochs, file=logfile)
            print('\tStarting batch size:', dataloader.batch_size, file=logfile)
            batchsteps_string = ', '.join(map(str, sorted(batchsteps))) if batchsteps_set else "None"
            print('\tBatchsteps:', batchsteps_string, file=logfile)
            print('\tLearning rate:', lrate, file=logfile)
            print('\tN sequences:', ncontigs, file=logfile)
            print('\tN samples:', nsamples, file=logfile, end='\n\n')

        # Train
        # simclr
        if self.contrast:
            # Optimizer setting
            awl = AutomaticWeightedLoss(3)
            optimizer = Adam([{'params':self.parameters(), 'lr':lrate}, {'params': awl.parameters(),'lr':0.001, 'weight_decay': 0}])
            # optimizer.add_param_group({'params':self.parameters(), 'lr':lrate})
            # for param in awl.parameters():
            #     print('awl',type(param), param.size())
            # Other optimizer options (nor complemented)
            # optimizer = lars.LARS([{'params':self.parameters(), 'lr':lrate, 'weight_decay': 0.01}, {'params': awl.parameters(),'lr':0.1, 'weight_decay': 0}])
            # optimizer.add_param_group({'params': awl.parameters(),'lr':0.1, 'weight_decay': 0})
            # print('optimizer',optimizer.param_groups)

            # Read augmentation data from indexed files
            count = 0
            while count * count < nepochs:
                count += 1
            for epoch in range(nepochs):
                aug_archive1_file, aug_archive2_file = glob.glob(rf'{augmentationpath+os.sep}pool0*k{self.k}*index{epoch//count}*'), glob.glob(rf'{augmentationpath+os.sep}pool1*k{self.k}*index{epoch%count}*')
                # Read augmentation data from shuffled-indexed files
                if augdatashuffle:
                    shuffle_file = random.randrange(0, 3 * count - 1)
                    if shuffle_file > 2 * count -1:
                        aug_archive1_file = glob.glob(rf'{augmentationpath+os.sep}pool2*k{self.k}*index{shuffle_file - 2 * count}*')
                    shuffle_file2 = random.randrange(0, 3 * count - 1)
                    if shuffle_file2 > 2 * count -1:
                        aug_archive2_file = glob.glob(rf'{augmentationpath+os.sep}pool2*k{self.k}*index{shuffle_file2 - 2 * count}*')
                aug_archive1, aug_archive2 = _np.load(aug_archive1_file[0]), _np.load(aug_archive2_file[0])
                aug_arr1, aug_arr2 = aug_archive1['arr_0'], aug_archive2['arr_0']
                # zscore for augmentation data (same as the depth and tnf)
                _vambtools.zscore(aug_arr1, axis=0, inplace=True)
                _vambtools.zscore(aug_arr2, axis=0, inplace=True)
                aug_tensor1, aug_tensor2 = _torch.from_numpy(aug_arr1), _torch.from_numpy(aug_arr2)
                print('difference',_torch.sum(_torch.sub(aug_tensor1, aug_tensor2), axis=1))

                # Double the batchsize and halve the learning rate if epoch in batchsteps
                if epoch in batchsteps:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] *= 0.5
                    data_loader = _DataLoader(dataset=_TensorDataset(depthstensor, tnftensor, aug_tensor1, aug_tensor2),
                                        batch_size=dataloader.batch_size if epoch == 0 else data_loader.batch_size * 2,
                                        shuffle=True, drop_last=False, num_workers=dataloader.num_workers, pin_memory=dataloader.pin_memory)
                else:
                    data_loader = _DataLoader(dataset=_TensorDataset(depthstensor, tnftensor, aug_tensor1, aug_tensor2),
                                        batch_size=dataloader.batch_size if epoch == 0 else data_loader.batch_size,
                                        drop_last=False, shuffle=True, num_workers=dataloader.num_workers, pin_memory=dataloader.pin_memory)
                self.trainepoch(data_loader, epoch, optimizer, batchsteps_set, logfile, hparams, awl)

        # vamb
        else:
            optimizer = Adam(self.parameters(), lr=lrate)
            data_loader = _DataLoader(dataset=dataloader.dataset,
                                    batch_size=dataloader.batch_size,
                                    shuffle=True,
                                    drop_last=False,
                                    num_workers=dataloader.num_workers,
                                    pin_memory=dataloader.pin_memory)
            for epoch in range(nepochs):
                if epoch in batchsteps:
                    data_loader = _DataLoader(dataset=data_loader.dataset,
                                        batch_size=data_loader.batch_size * 2,
                                        shuffle=True,
                                        drop_last=False,
                                        num_workers=data_loader.num_workers,
                                        pin_memory=data_loader.pin_memory)
                self.trainepoch(data_loader, epoch, optimizer, batchsteps_set, logfile, Namespace())

        # Save weights - Lord forgive me, for I have sinned when catching all exceptions
        if modelfile is not None:
            try:
                self.save(modelfile)
            except:
                pass

        return None

    def nt_xent_loss(self, out_1, out_2, temperature=0.1, eps=1e-6):
        """
            assume out_1 and out_2 are normalized
            out_1: [batch_size, dim]
            out_2: [batch_size, dim]
        """
        # gather representations in case of distributed training
        # out_1_dist: [batch_size * world_size, dim]
        # out_2_dist: [batch_size * world_size, dim]
        out_1 = F.normalize(out_1, dim=1)
        out_2 = F.normalize(out_2, dim=1)

        out_1_dist = out_1
        out_2_dist = out_2

        # out: [2 * batch_size, dim]
        # out_dist: [2 * batch_size * world_size, dim]
        out = _torch.cat([out_1, out_2], dim=0)
        out_dist = _torch.cat([out_1_dist, out_2_dist], dim=0)

        # sum by dim, we set dim=1 since our data are sequences
        # [2 * batch_size, 2 * batch_size * world_size] or 1
        # L2_norm = _torch.mm(_torch.sum(_torch.pow(out,2),dim=1,keepdim=True), _torch.sum(_torch.pow(out_dist.t().contiguous(),2),dim=0,keepdim=True))
        # L2_norm = _torch.clamp(L2_norm, min=eps)
        L2_norm = 1.

        # cov and sim: [2 * batch_size, 2 * batch_size * world_size]
        # neg: [2 * batch_size]
        cov = _torch.div(_torch.mm(out, out_dist.t().contiguous()), L2_norm)
        sim = _torch.exp(cov / temperature)
        neg = sim.sum(dim=-1)

        # from each row, subtract e^1 to remove similarity measure for x1.x1
        row_sub = _torch.Tensor(neg.shape).fill_(math.e).to(neg.device)
        neg = _torch.clamp(neg - row_sub, min=eps)  # clamp for numerical stability

        # Positive similarity, pos becomes [2 * batch_size]
        pos = _torch.exp(_torch.sum(out_1 * out_2, dim=-1) / temperature)
        pos = _torch.cat([pos, pos], dim=0)

        loss = -_torch.log(pos / (neg + eps)).mean()
        #print('out',out,cov,sim,neg,row_sub,pos)

        return loss


# Hidden layer monitor
def backward_hook(module, grad_in, grad_out):
    print('grad_in', grad_in, 'grad_out', grad_out, end='\t\t')


def forward_hook(module, inp, outp):
    print('feature_map', inp, 'feature_out', outp, end='\t\t')
