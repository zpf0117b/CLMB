import sys
import time
import pickle
import collections
import random

from torch.utils.data import DataLoader as _DataLoader
from torch.utils.data import Dataset, SubsetRandomSampler, SequentialSampler, RandomSampler
from torch.utils.data.dataset import TensorDataset as _TensorDataset
import torch as torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn import decomposition, manifold
from sklearn import datasets
import umap

SAMPLE_TIMES = 1

def select_topk(pre=True,referencefile='/headless/thesis/data/metahit/reference.tsv', contignamesfile='/headless/thesis/data/metahit/contignames.txt', 
            statfile='/headless/thesis/data/metahit/binning2.tsv', pklfile='/headless/thesis/data/metahit/top314-genome.pkl', k=314):
    if pre:
        with open(pklfile, 'rb') as f:
            genome_index_raw = pickle.load(f)
            f.close()
        genome_index = genome_index_raw[:k]
        # for item in genome_index:
        #     print(item[0],len(item[1]))
        #     time.sleep(0.1)
    else:
        stat = pd.read_csv(statfile, sep='\t', index_col=False)
        genome_topk = stat.loc[:k-1,'genomename'].tolist()
        subject_topk = stat.loc[:k-1,'subjectname'].tolist()

        reference = pd.read_csv(referencefile, sep='\t', index_col=False, names=['contigname','genomename','subjectname'])
        criterion1,criterion2 = reference['genomename'].map(lambda x: x in genome_topk), reference['subjectname'].map(lambda x: x in subject_topk)
        # binning_topk = reference.loc[(criterion1 & criterion2),'contigname'].tolist()
        binning_topk = reference[(criterion1 & criterion2)]
        # print(binning_topk_test.loc[:,['genomename','subjectname']].value_counts())

        with open(contignamesfile,'r') as f:
            raw_contignames = f.readlines()
            contignames = [rawstr.replace('\n','') for rawstr in raw_contignames]
            f.close()
        genome_index,genome_idx_map = [],{}
        for i in range(len(genome_topk)):
            index_tuple = (genome_topk[i],subject_topk[i])
            genome_index.append([index_tuple,[]])
            genome_idx_map[index_tuple] = i
        assert len(genome_index) == k
        for j in range(len(contignames)):
            binning_search = binning_topk[binning_topk['contigname'] == contignames[j]]
            if not binning_search.empty:
                index_tuple = (str((binning_search.iloc[0])['genomename']), str((binning_search.iloc[0])['subjectname']))
                (genome_index[genome_idx_map[index_tuple]])[1].append(j)
                # time.sleep(1)
            # if j % 10000 == 0:
            #     print(j)
        with open(pklfile, 'wb') as f:
            pickle.dump(genome_index, f, pickle.HIGHEST_PROTOCOL)
            f.close()
        assert len(genome_index) == k

    return genome_index

def visualize(hparams1,dl,visual_model:dict,method='pca', **kwargs):
    if 'select' in kwargs:
        k = kwargs['select']
        assert 1 < k < 20, 'Please use a more useful value for k'
        genome_index = select_topk(k=k)
        ds = dl.dataset.tensors
        i = 0
        select_tensor = []
        for j in range(k):
            # print(len(genome_index[j][1]))
            if i == 0:
                select_tensor.append(torch.index_select(ds[0], 0, torch.tensor(genome_index[j][1])))
                select_tensor.append(torch.index_select(ds[1], 0, torch.tensor(genome_index[j][1])))
                i = i + 1
            else:
                select_tensor[0] = torch.cat((select_tensor[0], torch.index_select(ds[0], 0, torch.tensor(genome_index[j][1]))), 0)
                select_tensor[1] = torch.cat((select_tensor[1], torch.index_select(ds[1], 0, torch.tensor(genome_index[j][1]))), 0)
        del ds
        ds = select_tensor

    if 'simclr' in visual_model.keys():
        print('view the simclr-vae')
        dataloader = _DataLoader(dataset=_TensorDataset(ds[0], ds[1]), 
        # dataloader = _DataLoader(dataset=_TensorDataset(torch.cat((ds[0],ds[1]),1)), 
                                batch_size=ds[0].shape[0] if 'select' in kwargs else hparams1.visualize_size, 
                                shuffle=False, num_workers=2, pin_memory=(True if hparams1.device=='cuda' else False),
                                drop_last=False)
        i = 0
        new_model = visual_model['simclr']
        new_model.usecuda = True if hparams1.device=='cuda' else False
        prelatent_simclr = torch.from_numpy(new_model.encode(dataloader))
        # with torch.no_grad():
        #     for idx, batch in enumerate(dataloader):
        #         c = batch[0].to(hparams1.device)
        #         mu = new_model(c)
        #         if i == 0:
        #             i = i + 1
        #             prelatent_simclr = mu.clone().detach()
        #         else:
        #             prelatent_simclr = torch.cat((prelatent_simclr,mu),0)
        raw = torch.cat((ds[0],ds[1]),1)

    # # for clustering
    # d = prelatent_simclr
    # for i in range(len(genome_index[0][1])+1):
    #     for j in range(len(d)):
    #         print(torch.sum(torch.pow(torch.sub(d[i], d[j]),2)))
    #         if j == len(genome_index[0][1])+1:
    #             print('next')
    # sys.exit(0)
    # for i in range(len(d)-100, len(d)):
    #     for j in range(len(d)-100, len(d)):
    #         print(torch.sum(torch.pow(torch.sub(d[i], d[j]),2)))
    # print('next')
    # for i in range(100):
    #     for j in range(len(d)-100, len(d)):
    #         print(torch.sum(torch.pow(torch.sub(d[i], d[j]),2)))
    # print('next')


    if 'vae' in visual_model.keys():
        prelatent_vae=[]
        print('view the vae')
        dataloader = _DataLoader(dataset=_TensorDataset(ds[0], ds[1]), batch_size=ds[0].shape[0] if 'select' in kwargs else hparams1.visualize_size, 
                                shuffle=False, num_workers=2, pin_memory=(True if hparams1.device=='cuda' else False),
                                drop_last=False)
        i = 0
        new_model = visual_model['vae']
        new_model.usecuda = True if hparams1.device=='cuda' else False
        prelatent_vae = torch.from_numpy(new_model.encode(dataloader))

    np.random.seed(42)
    # X = prelatent.cpu().numpy()
    if k < 11:
        colors = ['#ffff00', '#ff66ff', '#ff0000', '#996633', '#66ccff', '#66cc00', '#660066', '#339999', '#000066', '#000000']
    else:
        colors = ['#'+(str(hex(0x100000+j*0x733aa))[2:]) for j in range(k)]
    # random.shuffle(colors)

        
    # plt.clf()
    # ax = Axes3D(fig)
    # pca = decomposition.PCA(n_components=3)
    if method not in ['t-sne','umap']:
        if method != 'pca':
            print(f'no valid method{method}, redirect to pca')
            method = 'pca'
    if method == 'pca':
        m_pca = decomposition.PCA(n_components=2)
    elif method == 't-sne':
        m_tsne = manifold.TSNE(n_components=2,init='pca')
    elif method == 'umap':
        m_umap = umap.UMAP()
    

    for i in range(SAMPLE_TIMES):
        if 'simclr' in visual_model.keys():
            fig_s = plt.figure(1)
            fig_r = plt.figure(2)
            ax1,ax2 = fig_s.subplots(), fig_r.subplots()
            X = prelatent_simclr.cpu().numpy()
            X2 = raw.cpu().numpy()
            # print(X)
            if X.shape[-1] > 2:
                if method == 'pca':
                    X = m_pca.fit_transform(X)
                elif method == 't-sne':
                    X = m_tsne.fit_transform(X)
                elif method == 'umap':
                    X = m_umap.fit_transform(X)
            if X2.shape[-1] > 2:
                if method == 'pca':
                    X2 = m_pca.fit_transform(X2)
                elif method == 't-sne':
                    X2 = m_tsne.fit_transform(X2)
                elif method == 'umap':
                    X2 = m_umap.fit_transform(X2)

            assert X.shape[-1] > 1 and X2.shape[-1] > 1

            ax1.set_title(f'{method} for our method')
            ax2.set_title(f'{method} for raw')
            # ax1.scatter(X[:,0],X[:,1],c=(X[:,0]*X[:,0]+X[:,1]*X[:,1]), s=0.5,marker='.')
            if 'select' in kwargs:
                idx1,idx2=0,0
                for j in range(k):
                    idx2 = idx2 + len(genome_index[j][1])
                    ax1.scatter(X[idx1:idx2,0],X[idx1:idx2,1],s=1.5,c=colors[j],marker='.')           
                    ax2.scatter(X2[idx1:idx2,0],X2[idx1:idx2,1], s=1.5, c=colors[j],marker='.')
                    idx1 = idx2
            else:
                ax1.scatter(X[:,0],X[:,1],s=0.5,marker='.')           
                ax2.scatter(X2[:,0],X2[:,1], s=0.5,marker='.')
        if 'vae' in visual_model.keys():
            fig_v = plt.figure(3)
            ax3 = fig_v.subplots()
            X3 = prelatent_vae.cpu().numpy()
            if X3.shape[-1] > 2:
                if method == 'pca':
                    X3 = m_pca.fit_transform(X3)
                elif method == 't-sne':
                    X3 = m_tsne.fit_transform(X3)
                elif method == 'umap':
                    X3 = m_umap.fit_transform(X3)
            assert X3.shape[-1] > 1
            ax3.set_title(f'{method} for vamb')
            if 'select' in kwargs:
                idx1,idx2=0,0
                for j in range(k):
                    idx2 = idx2 + len(genome_index[j][1])
                    ax3.scatter(X3[idx1:idx2,0],X3[idx1:idx2,1],s=1.5,c=colors[j],marker='.')
                    idx1 = idx2
            else:
                ax3.scatter(X3[:,0],X3[:,1], s=0.5,marker='.')
        plt.show()
        plt.clf()
    # plt.savefig('/headless/thesis/data/airways/pca2D.png')

    # ax.scatter(X[:,0],X[:,1],X[:,2], c='r', edgecolor='k')
    # ax.w_xaxis.set_ticklabels([])
    # ax.w_yaxis.set_ticklabels([])
    # ax.w_zaxis.set_ticklabels([])
    # plt.savefig('/headless/thesis/data/airways/pca3D.png')
    sys.exit(0)

import pandas as pd
def statistics(filepath):
    reference = pd.read_csv(filepath, sep='\t', index_col=False, names=['contigname','genomename','subjectname','start', 'end'])
    # print(reference)
    # genome_count = reference['genomename'].value_counts()
    # subject_count = reference['subjectname'].value_counts()
    # genome_count.to_csv(filepath.replace('reference','genome'), sep='\t')
    # subject_count.to_csv(filepath.replace('reference','subject'), sep='\t')
    binning = reference.loc[:,['genomename','subjectname']].value_counts(sort=False)
    binning2 = binning.sort_values(ascending=False)
    binning.to_csv(filepath.replace('reference','binning'), sep='\t')
    binning2.to_csv(filepath.replace('reference','binning2'), sep='\t')
    # print(pd.read_csv(filepath, sep='\t',index_col=False))

def demo():
    fig_s = plt.figure(1,figsize=(1,1))
    fig_r = plt.figure(2,figsize=(1,1))
    ax1,ax2 = fig_s.subplots(), fig_r.subplots()
    ax1.set_title('100')
    X = np.random.normal(0,1,(100,100))
    # ax1.scatter(X[:,0],X[:,1],c=(X[:,0]*X[:,0]+X[:,1]*X[:,1]), s=0.5,marker='.')
    ax1.scatter(X[:,0],X[:,1],s=0.5,marker='.')
    ax2.set_title('196')
    X2 = np.random.normal(0,1,(200,200))
    ax2.scatter(X2[:,0],X2[:,1], s=0.5,marker='.')
    fig_v = plt.figure(3,figsize=(1,1))
    ax3 = fig_v.subplots()
    ax3.set_title('289')
    X3 = np.random.normal(0,1,(300,300))
    ax3.scatter(X3[:,0],X3[:,1], s=0.5,marker='.')
    plt.show()

if __name__ == '__main__':
    statistics('/headless/thesis/data/metahit/reference.tsv')
    # demo()
    x = select_topk(pre=False)
