#!/usr/bin/env python3

# More imports below, but the user's choice of processors must be parsed before
# numpy can be imported.
from ast import arg
import math
import sys
import os
import argparse
import torch
import datetime
import time
import shutil
import random
import glob
import warnings
from math import sqrt

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

def calc_tnf(outdir, fastapath, k, tnfpath, namespath, lengthspath, mincontiglength, logfile, nepochs, augmentation_store_dir, augmode, contrastive=True):
    begintime = time.time()
    log('\nLoading TNF', logfile, 0)
    log('Minimum sequence length: {}'.format(mincontiglength), logfile, 1)
    # If no path to FASTA is given, we load TNF from .npz files
    if fastapath is None:
        log('Loading TNF from npz array {}'.format(tnfpath), logfile, 1)
        tnfs = vamb.vambtools.read_npz(tnfpath)
        log('Loading contiglengths from npz array {}'.format(lengthspath), logfile, 1)
        contiglengths = vamb.vambtools.read_npz(lengthspath)
        log('Loading contignames from text {}'.format(namespath), logfile, 1)
        with open(namespath,'r') as f:
            raw_contignames = f.readlines()
            contignames = [rawstr.replace('\n','') for rawstr in raw_contignames]
            f.close()

        if not tnfs.dtype == np.float32:
            raise ValueError('TNFs .npz array must be of float32 dtype')

        if not np.issubdtype(contiglengths.dtype, np.integer):
            raise ValueError('contig lengths .npz array must be of an integer dtype')

        if not (len(tnfs) == len(contignames) == len(contiglengths)):
            raise ValueError('Not all of TNFs, names and lengths are same length')

        # Discard any sequence with a length below mincontiglength
        mask = contiglengths >= mincontiglength
        tnfs = tnfs[mask]
        contignames = list(np.array(contignames)[mask])
        contiglengths = contiglengths[mask]

    # Else parse FASTA files
    else:
        log('Loading data from FASTA file {}'.format(fastapath), logfile, 1)
        if not contrastive:
            with vamb.vambtools.Reader(fastapath, 'rb') as tnffile:
                tnfs, contignames, contiglengths = vamb.parsecontigs.read_contigs(tnffile, minlength=mincontiglength, k=k)
                tnffile.close()
        else:
            os.system(f'mkdir -p {augmentation_store_dir}')

            backup_iteration = math.ceil(math.sqrt(nepochs))
            log('Generating {} augmentation data'.format(backup_iteration), logfile, 1)
            with vamb.vambtools.Reader(fastapath, 'rb') as tnffile:
                tnfs, contignames, contiglengths = vamb.parsecontigs.read_contigs_augmentation(tnffile, minlength=mincontiglength, k=k, store_dir=augmentation_store_dir, backup_iteration=backup_iteration, augmode=augmode)
                tnffile.close()

        # Discard any sequence with a length below mincontiglength
        mask = contiglengths >= mincontiglength
        tnfs = tnfs[mask]
        contignames = list(np.array(contignames)[mask])
        contiglengths = contiglengths[mask]

        vamb.vambtools.write_npz(os.path.join(outdir, 'tnf.npz'), tnfs)
        vamb.vambtools.write_npz(os.path.join(outdir, 'lengths.npz'), contiglengths)
        with open(os.path.join(outdir, 'contignames.txt'),'w') as f:
            f.write('\n'.join(contignames))
            f.close()

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
def trainvae(outdir, rpkms, tnfs, k, contrastive, augmode, augdatashuffle, augmentationpath, temperature, nhiddens, nlatent, alpha, beta, dropout, cuda,
            batchsize, nepochs, lrate, batchsteps, logfile):

    begintime = time.time()
    log('\nCreating and training VAE', logfile)
    nsamples = rpkms.shape[1]

    # basic config for contrastive learning
    aug_all_method = ['GaussianNoise','Transition','Transversion','Mutation','AllAugmentation']
    hparams = Namespace(
        validation_size=4096,   # Debug only. Validation size for training.
        visualize_size=25600,   # Debug only. Visualization (pca) size for training.
        temperature=temperature,        # The parameter for contrastive loss
        augmode=augmode,        # Augmentation method choices (in aug_all_method)
        sigma = 4000,           # Add weight on the contrastive loss to avoid gradient disappearance
        lrate_decent = 0.8,     # Decrease the learning rate by lrate_decent for each batchstep
        augdatashuffle = augdatashuffle     # Shuffle the augmented data for training to introduce more noise. Setting True is not recommended. [False]
    )

    dataloader, mask = vamb.encode.make_dataloader(rpkms, tnfs, batchsize,
                                                   destroy=True, cuda=cuda)

    log('Created dataloader and mask', logfile, 1)
    vamb.vambtools.write_npz(os.path.join(outdir, 'mask.npz'), mask)
    n_discarded = len(mask) - mask.sum()
    log('Number of sequences unsuitable for encoding: {}'.format(n_discarded), logfile, 1)
    log('Number of sequences remaining: {}'.format(len(mask) - n_discarded), logfile, 1)
    print('', file=logfile)

    # clmb
    if contrastive:
        if True:
            vae = vamb.encode.VAE(ntnf=int(tnfs.shape[1]), nsamples=nsamples, k=k, nhiddens=nhiddens, nlatent=nlatent,alpha=alpha, beta=beta, dropout=dropout, cuda=cuda, c=True)
            modelpath = os.path.join(outdir, f"{aug_all_method[hparams.augmode[0]]+'_'+aug_all_method[hparams.augmode[1]]}.pt")
            vae.trainmodel(dataloader, nepochs=nepochs, lrate=lrate, batchsteps=batchsteps,logfile=logfile, modelfile=modelpath, hparams=hparams, augmentationpath=augmentationpath, mask=mask)
        else:
            modelpath = os.path.join(outdir, f"final-dim/{aug_all_method[hparams.augmode[0]]+' '+aug_all_method[hparams.augmode[1]]+' '+str(hparams.hidden_mlp)}.pt")
            vae = vamb.encode.VAE.load(modelpath,cuda=cuda,c=True)
            vae.to(('cuda' if cuda else 'cpu'))
    else:
        vae = vamb.encode.VAE(ntnf=int(tnfs.shape[1]), nsamples=nsamples, k=k, nhiddens=nhiddens, nlatent=nlatent, alpha=alpha, beta=beta, dropout=dropout, cuda=cuda)
        modelpath = os.path.join(outdir, 'model.pt')
        vae.trainmodel(dataloader, nepochs=nepochs, lrate=lrate, batchsteps=batchsteps, logfile=logfile, modelfile=modelpath)
    latent = vae.encode(dataloader)
    vamb.vambtools.write_npz(os.path.join(outdir, 'latent.npz'), latent)
    del vae # Needed to free "latent" array's memory references?

    # end
    # vamb.vambtools.write_npz(os.path.join(outdir, 'latent.npz'), latent)

    # visualize
    # from . import visualize
    # visual_model = {'simclr':vae} # 'vae':vae_model
    # visualize.visualize(hparams,dataloader,visual_model,method='umap',**{'select':18})

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

def run(outdir, fastapath, k, tnfpath, namespath, lengthspath, 
        contrastive, augmode, augdatashuffle, augmentationpath, temperature,
        bampaths, rpkmpath, jgipath,
        mincontiglength, norefcheck, minalignscore, minid, subprocesses, nhiddens, nlatent,
        nepochs, batchsize, cuda, alpha, beta, dropout, lrate, batchsteps, windowsize,
        minsuccesses, minclustersize, separator, maxclusters, minfasta, logfile):

    if contrastive:
        log('Starting Clmb version ' + '.'.join(map(str, vamb.__version__)), logfile)
    else:
        log('Starting Vamb version ' + '.'.join(map(str, vamb.__version__)), logfile)
    log('Date and time is ' + str(datetime.datetime.now()), logfile, 1)
    begintime = time.time()
    log(f'Start at {round(time.time(), 2)}', logfile, 1)

    # Get TNFs, save as npz
    tnfs, contignames, contiglengths = calc_tnf(outdir, fastapath, k, tnfpath, namespath,
                                            lengthspath, mincontiglength, logfile, nepochs, augmentationpath, augmode, contrastive)

    # if not (os.path.exists(augmentationpath) and len(os.listdir(augmentationpath)) >= nepochs * 4):
    #     raise FileNotFoundError('Not an existing directory or not an directory with enough files for training' + augmentationpath)
    log(f'TNFs completed at {round(time.time(), 2)}', logfile, 1)

    # Parse BAMs, save as npz
    refhash = None if norefcheck else vamb.vambtools._hash_refnames(contignames)
    if rpkmpath is None:
        rpkms = calc_rpkm(outdir, bampaths, rpkmpath, jgipath, mincontiglength, refhash,
                      len(contignames), minalignscore, minid, subprocesses, logfile)
    else:
        if rpkmpath == r'<padding|For|None|Rpkm|Input>':
            rpkms = np.ones((len(contignames),1),dtype=np.float32)
        else:
            rpkms_data = np.load(rpkmpath)
            rpkms = rpkms_data[rpkms_data.files[0]]
    log(f'RPKM completed at {round(time.time(), 2)}', logfile, 1)

    # Train, save model
    mask, latent = trainvae(outdir, rpkms, tnfs, k, contrastive, augmode, augdatashuffle, augmentationpath, temperature, nhiddens, nlatent, alpha, beta,
                            dropout, cuda, batchsize, nepochs, lrate, batchsteps, logfile)

    del tnfs, rpkms
    contignames = [c for c, m in zip(contignames, mask) if m]
    log(f'Deep learning completed at {round(time.time(), 2)}', logfile, 1)

    # Cluster, save tsv file
    clusterspath = os.path.join(outdir, 'clusters.tsv')
    cluster(clusterspath, latent, contignames, windowsize, minsuccesses, maxclusters,
            minclustersize, separator, cuda, logfile)
    log(f'Clustering completed at {round(time.time(), 2)}', logfile, 1)
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
    tnfos.add_argument('-k', dest='k', metavar='', type=int, default=4, help='k for kmer calculation')
    tnfos.add_argument('--tnfs', metavar='', help='path to .npz of TNF')
    tnfos.add_argument('--names', metavar='', help='path to .npz of names of sequences')
    tnfos.add_argument('--lengths', metavar='', help='path to .npz of seq lengths')

    # Contrastive learning arguments
    contrastiveos = parser.add_argument_group(title='Contrastive learning input')
    contrastiveos.add_argument('--contrastive', action='store_true', help='Whether to perform contrastive learning(CLMB) or not(VAMB). [False]')
    contrastiveos.add_argument('--augmode', metavar='', nargs = 2, type = int, default=[3, 3],
                        help='The augmentation method. Requires 2 int. specify -1 if trying all augmentation methods. Choices: 0 for gaussian noise, 1 for transition, 2 for transversion, 3 for mutation, -1 for all. [3, 3]')
    contrastiveos.add_argument('--augdatashuffle', action='store_true', 
            help='Whether to shuffle the training augmentation data (True: For each training, random select the augmentation data from the augmentation dir pool.\n!!!BE CAUTIOUS WHEN TRUNING ON [False])')
    contrastiveos.add_argument('--augmentation', metavar='', help='path to augmentation dir. [outdir/augmentation]')
    contrastiveos.add_argument('--temperature', metavar='', default=1,type=float, help='The temperature for the normalized temperature-scaled cross entropy loss. [1]')

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
                        default=600, help='epochs [600]')
    trainos.add_argument('-t', dest='batchsize', metavar='', type=int,
                        default=256, help='starting batch size [256]')
    trainos.add_argument('-q', dest='batchsteps', metavar='', type=int, nargs='*',
                        default=[25, 75, 150, 300], help='double batch size at epochs [25 75 150 300]')
    trainos.add_argument('-r', dest='lrate',  metavar='',type=float,
                        default=0.001, help='learning rate [0.001], set -1 for square lrate adjustment(a little bigger and DANGEROUS!!!)')

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
    print(args)
    ######################### CHECK INPUT/OUTPUT FILES #####################

    # Outdir does not exist
    args.outdir = os.path.abspath(args.outdir)
    if os.path.exists(args.outdir):
        # os.system(f'rm -r {args.outdir}')
        # raise FileExistsError(args.outdir)
        warnings.warn("OUTDIR exists. Please be careful that some files might be rewrited", FutureWarning)

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

    # Check the running mode (CLMB or VAMB)
    if args.contrastive:
        if args.augmentation is None:
            augmentation_data_dir = os.path.join(args.outdir, 'augmentation')
        else:
            augmentation_data_dir = args.augmentation

        augmentation_number = [0, 0]
        aug_all_method = ['GaussianNoise','Transition','Transversion','Mutation','AllAugmentation']

        for i in range(2):
            if args.augmode[i] == -1:
                augmentation_number[i] = len(glob.glob(rf'{augmentation_data_dir+os.sep}pool{i}*k{args.k}*'))
            elif 0<= args.augmode[i] <= 3:
                augmentation_number[i] = len(glob.glob(rf'{augmentation_data_dir+os.sep}pool{i}*k{args.k}*_{aug_all_method[args.augmode[i]]}_*'))
            else:
                raise argparse.ArgumentTypeError('If contrastive learning is on, augmode must be int >-2 and <4')

        if args.fasta is None:
            warnings.warn("CLMB can't recognize the type of augmentation data, so please make sure your augmentation data in augmentation dir fit the augmode", UserWarning)
            if augmentation_number[0] == 0 or augmentation_number[1] == 0:
                raise argparse.ArgumentTypeError('Must specify either FASTA or the augmentation .npz inputs')
            if (2 * augmentation_number[0]) ** 2 < args.nepochs or (2 * augmentation_number[1]) ** 2 < args.nepochs:
                warnings.warn("Not enough augmentation, use replicated data in the training, which might decrease the performance", FutureWarning)
        else:
            if 0 < (2 * augmentation_number[0]) ** 2 < args.nepochs or 0 < (2 * augmentation_number[1]) ** 2 < args.nepochs:
                warnings.warn("Not enough augmentation, regenerate the augmentation to maintain the performance, please use ctrl+C to stop this process in 20 seconds if you would not like the augmentation dir to be rewritten. \
                    You can choose using the augmentations in the augmentation dir (without specifying --fasta) after interruptting this program, or continuing this program to erase the augmentation dir and regenerate the augmentation data", UserWarning)
                for sleep_time in range(4):
                    print(f'Program to be continued in {20-4*sleep_time}s, please use ctrl+C to stop this process if you would not like the augmentation dir to be rewritten')
                    time.sleep(5)
                warnings.warn("Not enough augmentation, regenerate the augmentation to maintain the performance, erasing the augmentation dir. We will regenerate the augmentation in the following function", UserWarning)
                for erase_file in glob.glob(rf'{augmentation_data_dir+os.sep}pool*k{args.k}*'):
                    print(f'removing {erase_file} ...')
                    os.system(f'rm {erase_file}')

    else:
        augmentation_data_dir = os.path.join(args.outdir, 'augmentation')

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

    if args.lrate == -1:
        warnings.warn('The learning rate might be a bit big.', UserWarning)
        lrate = 0.0075 * sqrt(args.batchsize) if args.contrastive else 1e-3
    else:
        lrate = args.lrate
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
            args.k,
            args.tnfs,
            args.names,
            args.lengths,
            args.contrastive,
            args.augmode,
            args.augdatashuffle,
            augmentation_data_dir,
            args.temperature,
            args.bamfiles,
            args.rpkm if rpkm_flag else r'<padding|For|None|Rpkm|Input>',
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
            lrate=lrate,
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
