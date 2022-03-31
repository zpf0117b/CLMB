#!/usr/bin/env python3

# More imports below, but the user's choice of processors must be parsed before
# numpy can be imported.
import sys
import os
import argparse
import torch
import datetime
import time
import shutil
import random

DEFAULT_PROCESSES = min(os.cpu_count(), 8)

# These MUST be set before importing numpy
# I know this is a shitty hack, see https://github.com/numpy/numpy/issues/11826
os.environ["MKL_NUM_THREADS"] = str(DEFAULT_PROCESSES)
os.environ["NUMEXPR_NUM_THREADS"] = str(DEFAULT_PROCESSES)
os.environ["OMP_NUM_THREADS"] = str(DEFAULT_PROCESSES)
os.environ["CUDA_VISIBLE_DEVICES"]="1"
# Append vamb to sys.path to allow vamb import even if vamb was not installed
# using pip
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parentdir)

import numpy as np
import vamb

################################# DEFINE FUNCTIONS ##########################
def log(string, logfile, indent=0):
    print(('\t' * indent) + string, file=logfile)
    logfile.flush()

def calc_tnf(outdir, fastapath, tnfpath, namespath, lengthspath, mincontiglength, logfile):
    begintime = time.time()
    log('\nLoading TNF', logfile, 0)
    log('Minimum sequence length: {}'.format(mincontiglength), logfile, 1)
    # If no path to FASTA is given, we load TNF from .npz files
    if fastapath is None:
        log('Loading TNF from npz array {}'.format(tnfpath), logfile, 1)
        tnfs = vamb.vambtools.read_npz(tnfpath)
        log('Loading contignames from npz array {}'.format(namespath), logfile, 1)
        contignames = vamb.vambtools.read_npz(namespath)
        log('Loading contiglengths from npz array {}'.format(lengthspath), logfile, 1)
        contiglengths = vamb.vambtools.read_npz(lengthspath)

        if not tnfs.dtype == np.float32:
            raise ValueError('TNFs .npz array must be of float32 dtype')

        if not np.issubdtype(contiglengths.dtype, np.integer):
            raise ValueError('contig lengths .npz array must be of an integer dtype')

        if not (len(tnfs) == len(contignames) == len(contiglengths)):
            raise ValueError('Not all of TNFs, names and lengths are same length')

        # Discard any sequence with a length below mincontiglength
        mask = contiglengths >= mincontiglength
        tnfs = tnfs[mask]
        contignames = list(contignames[mask])
        contiglengths = contiglengths[mask]

    # Else parse FASTA files
    else:
        log('Loading data from FASTA file {}'.format(fastapath), logfile, 1)
        with vamb.vambtools.Reader(fastapath, 'rb') as tnffile:
            ret = vamb.parsecontigs.read_contigs(tnffile, minlength=mincontiglength)

        tnfs, contignames, contiglengths = ret
        vamb.vambtools.write_npz(os.path.join(outdir, 'tnf.npz'), tnfs)
        vamb.vambtools.write_npz(os.path.join(outdir, 'lengths.npz'), contiglengths)

    elapsed = round(time.time() - begintime, 2)
    ncontigs = len(contiglengths)
    nbases = contiglengths.sum()

    print('', file=logfile)
    log('Kept {} bases in {} sequences'.format(nbases, ncontigs), logfile, 1)
    log('Processed TNF in {} seconds'.format(elapsed), logfile, 1)

    return tnfs, contignames, contiglengths

def calc_rpkm(outdir, bampaths, rpkmpath, jgipath, mincontiglength, refhash, ncontigs,
              minalignscore, minid, subprocesses, logfile):
    begintime = time.time()
    log('\nLoading RPKM', logfile)
    # If rpkm is given, we load directly from .npz file
    if rpkmpath is not None:
        log('Loading RPKM from npz array {}'.format(rpkmpath), logfile, 1)
        rpkms = vamb.vambtools.read_npz(rpkmpath)

        if not rpkms.dtype == np.float32:
            raise ValueError('RPKMs .npz array must be of float32 dtype')
    
    else:
        log('Reference hash: {}'.format(refhash if refhash is None else refhash.hex()), logfile, 1)

    # Else if JGI is given, we load from that
    if jgipath is not None:
        log('Loading RPKM from JGI file {}'.format(jgipath), logfile, 1)
        with open(jgipath) as file:
            rpkms = vamb.vambtools._load_jgi(file, mincontiglength, refhash)

    else:
        log('Parsing {} BAM files with {} subprocesses'.format(len(bampaths) if bampaths is not None else 0, subprocesses),
           logfile, 1)
        log('Min alignment score: {}'.format(minalignscore), logfile, 1)
        log('Min identity: {}'.format(minid), logfile, 1)
        log('Min contig length: {}'.format(mincontiglength), logfile, 1)
        log('\nOrder of columns is:', logfile, 1)
        log('\n\t'.join(bampaths), logfile, 1)
        print('', file=logfile)

        dumpdirectory = os.path.join(outdir, 'tmp')
        rpkms = vamb.parsebam.read_bamfiles(bampaths, dumpdirectory=dumpdirectory,
                                            refhash=refhash, minscore=minalignscore,
                                            minlength=mincontiglength, minid=minid,
                                            subprocesses=subprocesses, logfile=logfile)
        print('', file=logfile)
        vamb.vambtools.write_npz(os.path.join(outdir, 'rpkm.npz'), rpkms)
        shutil.rmtree(dumpdirectory)

    if len(rpkms) != ncontigs:
        raise ValueError("Length of TNFs and length of RPKM does not match. Verify the inputs")

    elapsed = round(time.time() - begintime, 2)
    log('Processed RPKM in {} seconds'.format(elapsed), logfile, 1)

    return rpkms

from argparse import Namespace
def trainvae(outdir, rpkms, tnfs, nhiddens, nlatent, alpha, beta, dropout, cuda,
            batchsize, nepochs, lrate, batchsteps, logfile):

    begintime = time.time()
    log('\nCreating and training VAE', logfile)
    nsamples = rpkms.shape[1]
    dataloader, mask = vamb.encode.make_dataloader(rpkms, tnfs, batchsize,
                                                   destroy=True, cuda=cuda)
    log('Created dataloader and mask', logfile, 1)
    vamb.vambtools.write_npz(os.path.join(outdir, 'mask.npz'), mask)
    n_discarded = len(mask) - mask.sum()
    log('Number of sequences unsuitable for encoding: {}'.format(n_discarded), logfile, 1)
    log('Number of sequences remaining: {}'.format(len(mask) - n_discarded), logfile, 1)
    print('', file=logfile)

    # basic config
    aug_all_method = ['GaussianNoise','NumericalChange','WordDrop','FragmentTransfer','Reverse','CropAndResize']
    hparams1 = Namespace(
        lr=1e-1,
        epochs=nepochs,
        patience=50,
        num_nodes=1,
        gpus=0 if cuda else 1,
        batch_size=16384,
        train_size=len(mask) - n_discarded,
        validation_size=4096,
        visualize_size=25600,
        hidden_mlp=32,
        input_size=113,
        l2_norm=True, # if False, temperature = 1e+x; if True, temperature < 1
        temperature=0.1,
        feat_dim=32,
        device=cuda,
        aug_mode=(0,0)
    )

    # clmb
    if True:
        vae = vamb.encode.VAE(nsamples, nhiddens=nhiddens, nlatent=hparams1.hidden_mlp,alpha=alpha, beta=beta, dropout=dropout, cuda=cuda, c=True)
        modelpath = os.path.join(outdir.replace('results','data'), f"final-dim/{aug_all_method[hparams1.aug_mode[0]]+' '+aug_all_method[hparams1.aug_mode[1]]+' '+str(hparams1.hidden_mlp)}.pt")
        if False:
            from torch.utils.data import DataLoader as _DataLoader
            from torch.utils.data.dataset import TensorDataset as _TensorDataset
            tensor_data_depth, tensor_data_tnf = dataloader.dataset.tensors[0], dataloader.dataset.tensors[1]
            #print(tensor_data_tnf.shape)
            random_select = random.sample(range(0,len(mask),1),k=round(0.7*len(mask)))
            traindataset_subset = _TensorDataset(torch.index_select(tensor_data_depth, 0, torch.tensor(random_select)), torch.index_select(tensor_data_tnf, 0, torch.tensor(random_select)))
            #traindataset_subset, remain_dataset = _random_split(dataloader.dataset, [train_num, len(mask)-train_num])
            dataloader_subset = _DataLoader(dataset=traindataset_subset, batch_size=hparams1.batch_size, drop_last=False, shuffle=False, num_workers=0, pin_memory=cuda)
            vae.trainmodel(dataloader_subset, nepochs=nepochs, lrate=lrate, batchsteps=batchsteps,logfile=logfile, modelfile=modelpath, hparams1=hparams1)
        else:
            vae.trainmodel(dataloader, nepochs=nepochs, lrate=lrate, batchsteps=batchsteps,logfile=logfile, modelfile=modelpath, hparams1=hparams1)
    else:
        modelpath = os.path.join(outdir.replace('results','data'), f"final-dim/{aug_all_method[hparams1.aug_mode[0]]+' '+aug_all_method[hparams1.aug_mode[1]]+' '+str(hparams1.hidden_mlp)}.pt")
        vae = vamb.encode.VAE.load(modelpath,cuda=cuda,c=True)
        vae.to(('cuda' if cuda else 'cpu'))

    latent = vae.encode(dataloader)
    vamb.vambtools.write_npz(os.path.join(outdir, 'latent.npz'), latent)
    del vae # Needed to free "latent" array's memory references?

    # end
    # vamb.vambtools.write_npz(os.path.join(outdir, 'latent.npz'), latent)

    # visualize
    # from . import visualize
    # visual_model = {'simclr':vae} # 'vae':vae_model
    # visualize.visualize(hparams1,dataloader,visual_model,method='umap',**{'select':18})

    elapsed = round(time.time() - begintime, 2)
    log('Trained VAE and encoded in {} seconds'.format(elapsed), logfile, 1)
    return mask, latent

def cluster(clusterspath, latent, contignames, windowsize, minsuccesses, maxclusters,
            minclustersize, separator, cuda, logfile):
    begintime = time.time()

    log('\nClustering', logfile)
    log('Windowsize: {}'.format(windowsize), logfile, 1)
    log('Min successful thresholds detected: {}'.format(minsuccesses), logfile, 1)
    log('Max clusters: {}'.format(maxclusters), logfile, 1)
    log('Min cluster size: {}'.format(minclustersize), logfile, 1)
    log('Use CUDA for clustering: {}'.format(cuda), logfile, 1)
    log('Separator: {}'.format(None if separator is None else ('"'+separator+'"')),
        logfile, 1)
# other clustering algorithm
#    from sklearn.cluster import DBSCAN, MiniBatchKMeans
#    from sklearn.metrics.pairwise import pairwise_distances
#    from scipy.sparse import csr_matrix
#    import math
#    it = []
#    def get_all_index(lst: list=[], item=None):
#        return [index for (index,value) in enumerate(lst) if value == item]
    '''
    # source: https://gist.github.com/gdbassett/528d816d035f2deaaca1
    # X should be a numpy matrix, very likely sparse matrix: http://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.sparse.csr_matrix.html#scipy.sparse.csr_matrix
    # T1 > T2 for overlapping clusters
    # T1 = Distance to centroid point to not include in other clusters
    # T2 = Distance to centroid point to include in cluster
    # T1 > T2 for overlapping clusters
    # T1 < T2 will have points which reside in no clusters
    # T1 == T2 will cause all points to reside in mutually exclusive clusters
    # Distance metric can be any from here: http://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.pairwise_distances.html
    # filemap may be a list of point names in their order in X. If included, row numbers from X will be replaced with names from filemap. 
    '''
#    def canopy(X, T1, T2, distance_metric='euclidean', filemap=None):
#        X = csr_matrix(X)
#        canopies = dict()
#        X1_dist = pairwise_distances(X, metric=distance_metric)
#        canopy_points = set(range(X.shape[0]))
#        while canopy_points:
#            point = canopy_points.pop()
#            i = len(canopies)
#            canopies[i] = {"c":point, "points": list(np.where(X1_dist[point] < T2)[0])}
#            canopy_points = canopy_points.difference(set(np.where(X1_dist[point] < T1)[0]))
#        if filemap:
#            for canopy_id in canopies.keys():
#                canopy = canopies.pop(canopy_id)
#                canopy2 = {"c":filemap[canopy['c']], "points":list()}
#                for point in canopy['points']:
#                    canopy2["points"].append(filemap[point])
#                canopies[canopy_id] = canopy2
#        return canopies

# dbscan
    # gregarious,outlier = range(len(latent)),[]
    # dbscan_clustering = DBSCAN(eps=0.35, min_samples=2)
    # dbscan_labels = dbscan_clustering.fit_predict(latent)
    # all_value_dbscan = set(dbscan_labels)
    # outlier = get_all_index(dbscan_labels, -1)
    # print(len(all_value_dbscan),len(outlier))

    # for v in all_value_dbscan:
    #     if v != -1:
    #         v_cluster = get_all_index(dbscan_labels, v)
    #         medoid = contignames[v_cluster[0]]
    #         points = {contignames[i] for i in v_cluster}
    #         it.append((medoid, points))

    # for o in outlier:
    #     medoid = contignames[o]
    #     points = {contignames[i] for i in [o]}
    #     it.append((medoid, points))

# dbscan + kmeans 1
    # # exclude the outlier using DBSCAN
    # gregarious,outlier = range(len(latent)),[]
    # dbscan_clustering = DBSCAN(eps=0.16 * math.log2(int(latent.shape[1])), min_samples=2)
    # dbscan_labels = dbscan_clustering.fit_predict(latent)
    # all_value_dbscan = set(dbscan_labels)
    # outlier = get_all_index(dbscan_labels, -1)
    # print(len(all_value_dbscan),len(outlier))

    # gregarious = sorted(list(set(range(len(latent))).difference(set(outlier))))
    # gregarious_latent = latent[np.array(gregarious)]
    # print(len(gregarious))

    # # cluster the gregarious data using kmeans
    # n_clusters=len(all_value_dbscan)-1
    # kmeans_clustering = MiniBatchKMeans(n_clusters=n_clusters, random_state=0, batch_size=4096, max_iter=25, init_size=20000 if n_clusters < 20000 else n_clusters, reassignment_ratio=0.2)
    # kmeans_labels = kmeans_clustering.fit_predict(gregarious_latent)
    # all_value_kmeans = set(kmeans_labels)

    # for v in all_value_kmeans:
    #     v_cluster = get_all_index(kmeans_labels, v)
    #     medoid = contignames[gregarious[v_cluster[0]]]
    #     points = {contignames[gregarious[i]] for i in v_cluster}
    #     it.append((medoid, points))

    # for o in outlier:
    #     medoid = contignames[o]
    #     points = {contignames[i] for i in [o]}
    #     it.append((medoid, points))

# dbscan + kmeans 2
    # # exclude the outlier using DBSCAN
    # gregarious,outlier = range(len(latent)),[]
    # dbscan_clustering = DBSCAN(eps=15, min_samples=2)
    # dbscan_labels = dbscan_clustering.fit_predict(latent)
    # all_value_dbscan = set(dbscan_labels)
    # outlier = get_all_index(dbscan_labels, -1)
    # print(len(all_value_dbscan),len(outlier))

    # gregarious = range(len(latent))
    # gregarious_latent = latent[np.array(gregarious)]

    # # cluster using kmeans
    # n_clusters=len(all_value_dbscan) - 1 + len(outlier)
    # kmeans_clustering = MiniBatchKMeans(n_clusters=n_clusters, random_state=0, batch_size=4096, max_iter=25, init_size=20000 if n_clusters < 20000 else n_clusters, reassignment_ratio=0.2)
    # kmeans_labels = kmeans_clustering.fit_predict(gregarious_latent)
    # all_value_kmeans = set(kmeans_labels)

    # for v in all_value_kmeans:
    #     v_cluster = get_all_index(kmeans_labels, v)
    #     medoid = contignames[gregarious[v_cluster[0]]]
    #     points = {contignames[gregarious[i]] for i in v_cluster}
    #     it.append((medoid, points))

# canopy + kmeans
    # gregarious,outlier = range(len(latent)),[]

    # gregarious = sorted(list(set(range(len(latent))).difference(set(outlier))))
    # gregarious_latent = latent[np.array(gregarious)]
    # # print(len(gregarious))

    # # use canopy algorithm to estimate k for k-means
    # canopy_clustering = canopy(gregarious_latent, 2, 3)
    # print(len(canopy_clustering))

    # # cluster the gregarious data using kmeans
    # n_clusters=len(canopy_clustering)
    # kmeans_clustering = MiniBatchKMeans(n_clusters=n_clusters, random_state=0, batch_size=4096, max_iter=25, init_size=20000 if n_clusters < 20000 else n_clusters, reassignment_ratio=0.2)
    # kmeans_labels = kmeans_clustering.fit_predict(gregarious_latent)
    # all_value_kmeans = set(kmeans_labels)

    # for v in all_value_kmeans:
    #     v_cluster = get_all_index(kmeans_labels, v)
    #     medoid = contignames[gregarious[v_cluster[0]]]
    #     points = {contignames[gregarious[i]] for i in v_cluster}
    #     it.append((medoid, points))

# medoid
    it = vamb.cluster.cluster(latent, contignames, destroy=True, windowsize=windowsize,
                                normalized=False, minsuccesses=minsuccesses, cuda=cuda)

    renamed = ((str(i+1), c) for (i, (n,c)) in enumerate(it))

    # Binsplit if given a separator
    if separator is not None:
        renamed = vamb.vambtools.binsplit(renamed, separator)

    with open(clusterspath, 'w') as clustersfile:
        _ = vamb.vambtools.write_clusters(clustersfile, renamed, max_clusters=maxclusters,
                                          min_size=minclustersize, rename=False)
    clusternumber, ncontigs = _

    print('', file=logfile)
    log('Clustered {} contigs in {} bins'.format(ncontigs, clusternumber), logfile, 1)

    elapsed = round(time.time() - begintime, 2)
    log('Clustered contigs in {} seconds'.format(elapsed), logfile, 1)

def write_fasta(outdir, clusterspath, fastapath, contignames, contiglengths, minfasta, logfile):
    begintime = time.time()

    log('\nWriting FASTA files', logfile)
    log('Minimum FASTA size: {}'.format(minfasta), logfile, 1)

    lengthof = dict(zip(contignames, contiglengths))
    filtered_clusters = dict()

    with open(clusterspath) as file:
        clusters = vamb.vambtools.read_clusters(file)

    for cluster, contigs in clusters.items():
        size = sum(lengthof[contig] for contig in contigs)
        if size >= minfasta:
            filtered_clusters[cluster] = clusters[cluster]

    del lengthof, clusters
    keep = set()
    for contigs in filtered_clusters.values():
        keep.update(set(contigs))

    with vamb.vambtools.Reader(fastapath, 'rb') as file:
        fastadict = vamb.vambtools.loadfasta(file, keep=keep)

    vamb.vambtools.write_bins(os.path.join(outdir, "bins"), filtered_clusters, fastadict, maxbins=None)

    ncontigs = sum(map(len, filtered_clusters.values()))
    nfiles = len(filtered_clusters)
    print('', file=logfile)
    log('Wrote {} contigs to {} FASTA files'.format(ncontigs, nfiles), logfile, 1)

    elapsed = round(time.time() - begintime, 2)
    log('Wrote FASTA in {} seconds'.format(elapsed), logfile, 1)

def run(outdir, fastapath, tnfpath, namespath, lengthspath, bampaths, rpkmpath, jgipath,
        mincontiglength, norefcheck, minalignscore, minid, subprocesses, nhiddens, nlatent,
        nepochs, batchsize, cuda, alpha, beta, dropout, lrate, batchsteps, windowsize,
        minsuccesses, minclustersize, separator, maxclusters, minfasta, logfile):

    log('Starting Vamb version ' + '.'.join(map(str, vamb.__version__)), logfile)
    log('Date and time is ' + str(datetime.datetime.now()), logfile, 1)
    begintime = time.time()

    # Get TNFs, save as npz
    if False:
        tnfs, contignames, contiglengths = calc_tnf(outdir, fastapath, tnfpath, namespath,
                                                lengthspath, mincontiglength, logfile)
        vamb.vambtools.write_npz(os.path.join(outdir.replace('results','data'), 'contiglengths.npz'), contiglengths)
        vamb.vambtools.write_npz(os.path.join(outdir.replace('results','data'), 'tnfs.npz'), tnfs)
        with open(os.path.join(outdir.replace('results','data'), 'contignames.txt'),'w') as f:
            f.write('\n'.join(contignames))
            f.close()
    else:
        contiglengths = vamb.vambtools.read_npz(os.path.join(outdir.replace('results','data'), 'contiglengths.npz'))
        tnfs = vamb.vambtools.read_npz(os.path.join(outdir.replace('results','data'), 'tnfs.npz'))
        with open(os.path.join(outdir.replace('results','data'), 'contignames.txt'),'r') as f:
            raw_contignames = f.readlines()
            contignames = [rawstr.replace('\n','') for rawstr in raw_contignames]
            f.close()
    print(1)
    # Parse BAMs, save as npz
    refhash = None if norefcheck else vamb.vambtools._hash_refnames(contignames)
    if rpkmpath is None:
        rpkms = calc_rpkm(outdir, bampaths, rpkmpath, jgipath, mincontiglength, refhash,
                      len(tnfs), minalignscore, minid, subprocesses, logfile)
    else:
        if rpkmpath == r'padding$padding':
            rpkms = np.ones((len(contignames),1),dtype=np.float32)
        else:
            rpkms_data = np.load(rpkmpath)
            rpkms = rpkms_data[rpkms_data.files[0]]
    print(2)
    # Train, save model
    mask, latent = trainvae(outdir, rpkms, tnfs, nhiddens, nlatent, alpha, beta,
                           dropout, cuda, batchsize, nepochs, lrate, batchsteps, logfile)

    del tnfs, rpkms
    contignames = [c for c, m in zip(contignames, mask) if m]
    print(3)
    # Cluster, save tsv file
    clusterspath = os.path.join(outdir, 'clusters.tsv')
    cluster(clusterspath, latent, contignames, windowsize, minsuccesses, maxclusters,
            minclustersize, separator, cuda, logfile)
    print(4)
    del latent

    if minfasta is not None:
        write_fasta(outdir, clusterspath, fastapath, contignames, contiglengths, minfasta,
        logfile)

    elapsed = round(time.time() - begintime, 2)
    log('\nCompleted Vamb in {} seconds'.format(elapsed), logfile)

def main():
    doc = """Vamb: Variational autoencoders for metagenomic binning.

    Default use, good for most datasets:
    vamb --outdir out --fasta my_contigs.fna --bamfiles *.bam

    For advanced use and extensions of Vamb, check documentation of the package
    at https://github.com/RasmussenLab/vamb."""
    parser = argparse.ArgumentParser(
        prog="vamb",
        description=doc,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        usage="%(prog)s outdir tnf_input rpkm_input [options]",
        add_help=False)

    # Help
    helpos = parser.add_argument_group(title='Help', description=None)
    helpos.add_argument('-h', '--help', help='print help and exit', action='help')

    # Positional arguments
    reqos = parser.add_argument_group(title='Output (required)', description=None)
    reqos.add_argument('--outdir', metavar='', required=True, help='output directory to create')

    # TNF arguments
    tnfos = parser.add_argument_group(title='TNF input (either fasta or all .npz files required)')
    tnfos.add_argument('--fasta', metavar='', help='path to fasta file')
    tnfos.add_argument('--tnfs', metavar='', help='path to .npz of TNF')
    tnfos.add_argument('--names', metavar='', help='path to .npz of names of sequences')
    tnfos.add_argument('--lengths', metavar='', help='path to .npz of seq lengths')

    # RPKM arguments
    rpkmos = parser.add_argument_group(title='RPKM input (either BAMs, JGI or .npz required)')
    rpkmos.add_argument('--bamfiles', metavar='', help='paths to (multiple) BAM files', nargs='+')
    rpkmos.add_argument('--rpkm', metavar='', help='path to .npz of RPKM')
    rpkmos.add_argument('--jgi', metavar='', help='path to output of jgi_summarize_bam_contig_depths')

    # Optional arguments
    inputos = parser.add_argument_group(title='IO options', description=None)

    inputos.add_argument('-m', dest='minlength', metavar='', type=int, default=100,
                         help='ignore contigs shorter than this [100]')
    inputos.add_argument('-s', dest='minascore', metavar='', type=int, default=None,
                         help='ignore reads with alignment score below this [None]')
    inputos.add_argument('-z', dest='minid', metavar='', type=float, default=None,
                         help='ignore reads with nucleotide identity below this [None]')
    inputos.add_argument('-p', dest='subprocesses', metavar='', type=int, default=DEFAULT_PROCESSES,
                         help=('number of subprocesses to spawn '
                              '[min(' + str(DEFAULT_PROCESSES) + ', nbamfiles)]'))
    inputos.add_argument('--norefcheck', help='skip reference name hashing check [False]',
                         action='store_true')
    inputos.add_argument('--minfasta', dest='minfasta', metavar='', type=int, default=None,
                         help='minimum bin size to output as fasta [None = no files]')

    # VAE arguments
    vaeos = parser.add_argument_group(title='VAE options', description=None)

    vaeos.add_argument('-n', dest='nhiddens', metavar='', type=int, nargs='+',
                        default=None, help='hidden neurons [Auto]')
    vaeos.add_argument('-l', dest='nlatent', metavar='', type=int,
                        default=32, help='latent neurons [32]')
    vaeos.add_argument('-a', dest='alpha',  metavar='',type=float,
                        default=None, help='alpha, weight of TNF versus depth loss [Auto]')
    vaeos.add_argument('-b', dest='beta',  metavar='',type=float,
                        default=200.0, help='beta, capacity to learn [200.0]')
    vaeos.add_argument('-d', dest='dropout',  metavar='',type=float,
                        default=None, help='dropout [Auto]')
    vaeos.add_argument('--cuda', help='use GPU to train & cluster [False]', action='store_true')

    trainos = parser.add_argument_group(title='Training options', description=None)

    trainos.add_argument('-e', dest='nepochs', metavar='', type=int,
                        default=1250, help='epochs [900]')
    trainos.add_argument('-t', dest='batchsize', metavar='', type=int,
                        default=256, help='starting batch size [256]')
    trainos.add_argument('-q', dest='batchsteps', metavar='', type=int, nargs='*',
                        default=[25, 75, 150, 300], help='double batch size at epochs [25 75 150 300]')
    trainos.add_argument('-r', dest='lrate',  metavar='',type=float,
                        default=1e-3, help='learning rate [0.001]')

    # Clustering arguments
    clusto = parser.add_argument_group(title='Clustering options', description=None)
    clusto.add_argument('-w', dest='windowsize', metavar='', type=int,
                        default=200, help='size of window to count successes [200]')
    clusto.add_argument('-u', dest='minsuccesses', metavar='', type=int,
                        default=20, help='minimum success in window [20]')
    clusto.add_argument('-i', dest='minsize', metavar='', type=int,
                        default=1, help='minimum cluster size [1]')
    clusto.add_argument('-c', dest='maxclusters', metavar='', type=int,
                        default=None, help='stop after c clusters [None = infinite]')
    clusto.add_argument('-o', dest='separator', metavar='', type=str,
                        default=None, help='binsplit separator [None = no split]')

    ######################### PRINT HELP IF NO ARGUMENTS ###################
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit()

    args = parser.parse_args()
    ######################### CHECK INPUT/OUTPUT FILES #####################

    # Outdir does not exist
    args.outdir = os.path.abspath(args.outdir)
    if os.path.exists(args.outdir):
        os.system(f'rm -r {args.outdir}')
        # raise FileExistsError(args.outdir)

    # Outdir is in an existing parent dir
    parentdir = os.path.dirname(args.outdir)
    if parentdir and not os.path.isdir(parentdir):
        raise NotADirectoryError(parentdir)

    # Make sure only one TNF input is there
    if args.fasta is None:
        for path in args.tnfs, args.names, args.lengths:
            if path is None:
                raise argparse.ArgumentTypeError('Must specify either FASTA or the three .npz inputs')
            if not os.path.isfile(path):
                raise FileNotFoundError(path)
    else:
        for path in args.tnfs, args.names, args.lengths:
            if path is not None:
                raise argparse.ArgumentTypeError('Must specify either FASTA or the three .npz inputs')
        if not os.path.isfile(args.fasta):
            raise FileNotFoundError('Not an existing non-directory file: ' + args.fasta)

    # Make sure only one RPKM input is there
    rpkm_flag = True
    if sum(i is not None for i in (args.bamfiles, args.rpkm, args.jgi)) > 1:
        raise argparse.ArgumentTypeError('Must specify exactly one of BAM files, JGI file or RPKM input')
    elif sum(i is not None for i in (args.bamfiles, args.rpkm, args.jgi)) == 0:
        rpkm_flag = False

    for path in args.rpkm, args.jgi:
        if path is not None and not os.path.isfile(path):
            raise FileNotFoundError('Not an existing non-directory file: ' + args.rpkm)

    if args.bamfiles is not None:
        for bampath in args.bamfiles:
            if not os.path.isfile(bampath):
                raise FileNotFoundError('Not an existing non-directory file: ' + bampath)

    # Check minfasta settings
    if args.minfasta is not None and args.fasta is None:
        raise argparse.ArgumentTypeError('If minfasta is not None, '
                                         'input fasta file must be given explicitly')

    if args.minfasta is not None and args.minfasta < 0:
        raise argparse.ArgumentTypeError('Minimum FASTA output size must be nonnegative')

    ####################### CHECK ARGUMENTS FOR TNF AND BAMFILES ###########
    if args.minlength < 100:
        raise argparse.ArgumentTypeError('Minimum contig length must be at least 100')

    if args.minid is not None and (args.minid < 0 or args.minid >= 1.0):
        raise argparse.ArgumentTypeError('Minimum nucleotide ID must be in [0,1)')

    if args.minid is not None and args.bamfiles is None:
        raise argparse.ArgumentTypeError('If minid is set, RPKM must be passed as bam files')

    if args.minascore is not None and args.bamfiles is None:
        raise argparse.ArgumentTypeError('If minascore is set, RPKM must be passed as bam files')

    if args.subprocesses < 1:
        raise argparse.ArgumentTypeError('Zero or negative subprocesses requested')

    ####################### CHECK VAE OPTIONS ################################
    if args.nhiddens is not None and any(i < 1 for i in args.nhiddens):
        raise argparse.ArgumentTypeError('Minimum 1 neuron per layer, not {}'.format(min(args.hidden)))

    if args.nlatent < 1:
        raise argparse.ArgumentTypeError('Minimum 1 latent neuron, not {}'.format(args.latent))

    if args.alpha is not None and (args.alpha <= 0 or args.alpha >= 1):
        raise argparse.ArgumentTypeError('alpha must be above 0 and below 1')

    if args.beta <= 0:
        raise argparse.ArgumentTypeError('beta cannot be negative or zero')

    if args.dropout is not None and (args.dropout < 0 or args.dropout >= 1):
        raise argparse.ArgumentTypeError('dropout must be in 0 <= d < 1')

    if args.cuda and not torch.cuda.is_available():
        raise ModuleNotFoundError('Cuda is not available on your PyTorch installation')

    ###################### CHECK TRAINING OPTIONS ####################
    if args.nepochs < 1:
        raise argparse.ArgumentTypeError('Minimum 1 epoch, not {}'.format(args.nepochs))

    if args.batchsize < 1:
        raise argparse.ArgumentTypeError('Minimum batchsize of 1, not {}'.format(args.batchsize))

    args.batchsteps = sorted(set(args.batchsteps))
    if max(args.batchsteps, default=0) >= args.nepochs:
        raise argparse.ArgumentTypeError('All batchsteps must be less than nepochs')

    if min(args.batchsteps, default=1) < 1:
        raise argparse.ArgumentTypeError('All batchsteps must be 1 or higher')

    if args.lrate <= 0:
        raise argparse.ArgumentTypeError('Learning rate must be positive')

    ###################### CHECK CLUSTERING OPTIONS ####################
    if args.minsize < 1:
        raise argparse.ArgumentTypeError('Minimum cluster size must be at least 0')

    if args.windowsize < 1:
        raise argparse.ArgumentTypeError('Window size must be at least 1')

    if args.minsuccesses < 1 or args.minsuccesses > args.windowsize:
        raise argparse.ArgumentTypeError('Minimum cluster size must be in 1:windowsize')

    if args.separator is not None and len(args.separator) == 0:
        raise argparse.ArgumentTypeError('Binsplit separator cannot be an empty string')

    ###################### SET UP LAST PARAMS ############################

    # This doesn't actually work, but maybe the PyTorch folks will fix it sometime.
    subprocesses = args.subprocesses
    torch.set_num_threads(args.subprocesses)
    if args.bamfiles is not None:
        subprocesses = min(subprocesses, len(args.bamfiles))

    ################### RUN PROGRAM #########################
    try:
        os.mkdir(args.outdir)
    except FileExistsError:
        pass
    except:
        raise
        
    logpath = os.path.join(args.outdir, 'log.txt')

    with open(logpath, 'w') as logfile:
        run(args.outdir,
            args.fasta,
            args.tnfs,
            args.names,
            args.lengths,
            args.bamfiles,
            args.rpkm if rpkm_flag else r'padding$padding',
            args.jgi,
            mincontiglength=args.minlength,
            norefcheck=args.norefcheck,
            minalignscore=args.minascore,
            minid=args.minid,
            subprocesses=subprocesses,
            nhiddens=args.nhiddens,
            nlatent=args.nlatent,
            nepochs=args.nepochs,
            batchsize=args.batchsize,
            cuda=args.cuda,
            alpha=args.alpha,
            beta=args.beta,
            dropout=args.dropout,
            lrate=args.lrate,
            batchsteps=args.batchsteps,
            windowsize=args.windowsize,
            minsuccesses=args.minsuccesses,
            minclustersize=args.minsize,
            separator=args.separator,
            maxclusters=args.maxclusters,
            minfasta=args.minfasta,
            logfile=logfile)

if __name__ == '__main__':
    main()
