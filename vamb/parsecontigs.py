__doc__ = """Calculate tetranucleotide frequency from a FASTA file.

Usage:
>>> with open('/path/to/contigs.fna', 'rb') as filehandle
...     tnfs, contignames, lengths = read_contigs(filehandle)
"""

import sys as _sys
import os as _os
import numpy as _np
from itertools import product
import random
import vamb.vambtools as _vambtools
from vamb._vambtools import _kmercounts
from . import mimics
# for debug
from time import time

# This kernel is created in src/create_kernel.py. See that file for explanation
_KERNEL = _vambtools.read_npz(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                              "kernel.npz"))

def _project(fourmers, kernel=_KERNEL):
    "Project fourmers(4-mer) down in dimensionality"
    s = fourmers.sum(axis=1).reshape(-1, 1)
    s[s == 0] = 1.0
    fourmers *= 1/s
    fourmers += -(1/256)
    return _np.dot(fourmers, kernel)

def _convert(raw, projected):
    "Move data from raw PushArray to projected PushArray, converting it."
    raw_mat = raw.take().reshape(-1, 256)
    projected_mat = _project(raw_mat)
    projected.extend(projected_mat.ravel())
    raw.clear()

def read_contigs(filehandle, minlength=100):
    """Parses a FASTA file open in binary reading mode.

    Input:
        filehandle: Filehandle open in binary mode of a FASTA file
        minlength: Ignore any references shorter than N bases [100]

    Outputs:
        tnfs: An (n_FASTA_entries x 103) matrix of tetranucleotide freq.
        contignames: A list of contig headers
        lengths: A Numpy array of contig lengths
    """

    if minlength < 4:
        raise ValueError('Minlength must be at least 4, not {}'.format(minlength))

    raw = _vambtools.PushArray(_np.float32)
    projected = _vambtools.PushArray(_np.float32)
    lengths = _vambtools.PushArray(_np.int)
    contignames = list()

    entries = _vambtools.byte_iterfasta(filehandle)

    for entry in entries:
        if len(entry) < minlength:
            continue

        raw.extend(entry.kmercounts(4))

        if len(raw) > 256000: # extend for 1000 times
            _convert(raw, projected)

        lengths.append(len(entry))
        contignames.append(entry.header)

    # Convert rest of contigs
    _convert(raw, projected)
    tnfs_arr = projected.take()

    # Don't use reshape since it creates a new array object with shared memory
    tnfs_arr.shape = (len(tnfs_arr)//103, 103)
    lengths_arr = lengths.take()

    return tnfs_arr, contignames, lengths_arr

def read_contigs_augmentation(filehandle, minlength=100, store_dir="./", backup_iteration=100, augmode=[-1,-1], augdatashuffle=False, usecuda=False):
    """Parses a FASTA file open in binary reading mode.

    Input:
        filehandle: Filehandle open in binary mode of a FASTA file
        minlength: Ignore any references shorter than N bases [100]
        backup_iteration: numbers of generation for training
        store_dir: the dir to store the augmentation data

    Outputs:
        tnfs: An (n_FASTA_entries x 103) matrix of tetranucleotide freq.
        contignames: A list of contig headers
        lengths: A Numpy array of contig lengths
    """

    if minlength < 4:
        raise ValueError('Minlength must be at least 4, not {}'.format(minlength))

    k = 4

    norm = _vambtools.PushArray(_np.float32)
    gaussian = _vambtools.PushArray(_np.float32)
    trans = _vambtools.PushArray(_np.float32)
    traver = _vambtools.PushArray(_np.float32)
    mutated = _vambtools.PushArray(_np.float32)
    counts_kmer = _np.zeros(1 << (2*k), dtype=_np.int32)

    lengths = _vambtools.PushArray(_np.int)
    contignames = list()

    # We do not generate the number due to time cost. We just find the minimum augmentation we need for all iteration (count)
    count = 0
    while count * count < backup_iteration:
        count += 1
    # Create backup augmentation pools
    pools = 2
    gaussian_count, trans_count, traver_count, mutated_count = [0,0,0], [0,0,0], [0,0,0], [0,0,0]
    if augdatashuffle:
        pools = 3
        augmode.append(-1)
    # Pool 1
    for i in range(pools):
        if augmode[i] == -1:
            # Constraits: transition frequency = 2 * transversion frequency = 4 * gaussian noise frequency
            gaussian_count[i], trans_count[i], traver_count[i], mutated_count[i] = count - count*4//14 - count*2//14 - count//2, count*4//14, count*2//14, count//2
        elif augmode[i] == 0:
            gaussian_count[i], trans_count[i], traver_count[i], mutated_count[i] = count, 0, 0, 0
        elif augmode[i] == 1:
            gaussian_count[i], trans_count[i], traver_count[i], mutated_count[i] = 0, count, 0, 0
        elif augmode[i] == 2:
            gaussian_count[i], trans_count[i], traver_count[i], mutated_count[i] = 0, 0, count, 0
        elif augmode[i] == 3:
            gaussian_count[i], trans_count[i], traver_count[i], mutated_count[i] = 0, 0, 0, count

        filehandle.filehandle.seek(0, 0)
        entries = _vambtools.byte_iterfasta(filehandle)
        for entry in entries:
            if len(entry) < minlength:
                continue
            
            t = entry.kmercounts(k)
            t_norm = t / _np.sum(t)
            _np.add(t_norm, - 1/(2*4**k), out=t_norm)
            #print(i,pools-1)
            if i == 0:
                norm.extend(t_norm)

            for j in range(gaussian_count[i]):
                t_gaussian = mimics.add_noise(t_norm)
                gaussian.extend(t_gaussian)
                #print(sum(t_gaussian),end=' ')

            mutations = mimics.transition(entry.sequence, 1 - 0.021, trans_count[i])
            for j in range(trans_count[i]):
                _kmercounts(bytearray(mutations[j]), k, counts_kmer)
                t_trans = counts_kmer / _np.sum(counts_kmer)
                _np.add(t_trans, - 1/(2*4**k), out=t_trans)
                trans.extend(t_trans)
                #print(sum(t_trans),end=' ')

            mutations = mimics.transversion(entry.sequence, 1 - 0.0105, traver_count[i])
            for j in range(traver_count[i]):
                _kmercounts(bytearray(mutations[j]), k, counts_kmer)
                t_traver = counts_kmer / _np.sum(counts_kmer)
                _np.add(t_traver, - 1/(2*4**k), out=t_traver)
                traver.extend(t_traver)
                #print(sum(t_traver),end=' ')

            mutations = mimics.transition_transversion(entry.sequence, 1 - 0.014, 1 - 0.007, mutated_count[i])
            for j in range(mutated_count[i]):
                _kmercounts(bytearray(mutations[j]), k, counts_kmer)
                t_mutated = counts_kmer / _np.sum(counts_kmer)
                _np.add(t_mutated, - 1/(2*4**k), out=t_mutated)
                mutated.extend(t_mutated)
                #print(sum(t_mutated),end=' ')

            if i == 0:
                lengths.append(len(entry))
                contignames.append(entry.header)

        # Don't use reshape since it creates a new array object with shared memory
        gaussian_arr = gaussian.take()
        gaussian_arr.shape = (-1, gaussian_count[i], 4**k)
        trans_arr = trans.take()
        trans_arr.shape = (-1, trans_count[i], 4**k)
        traver_arr = traver.take()
        traver_arr.shape = (-1, traver_count[i], 4**k)
        mutated_arr = mutated.take()
        mutated_arr.shape = (-1, mutated_count[i], 4**k)

        index = 0
        index_list = list(range(count))
        random.shuffle(index_list)
        for j2 in range(gaussian_count[i]):
            gaussian_save = gaussian_arr[:,j2,:]
            gaussian_save.shape = (-1, 4**k)
            _np.savez(f"{store_dir}/pool{i}_gaussian_{j2}_index{index_list[index]}.npz", gaussian_save)
            index += 1

        for j2 in range(trans_count[i]):
            trans_save = trans_arr[:,j2,:]
            trans_save.shape = (-1, 4**k)
            _np.savez(f"{store_dir}/pool{i}_trans_{j2}_index{index_list[index]}.npz", trans_save)
            index += 1

        for j2 in range(traver_count[i]):
            traver_save = traver_arr[:,j2,:]
            traver_save.shape = (-1, 4**k)
            _np.savez(f"{store_dir}/pool{i}_traver_{j2}_index{index_list[index]}.npz", traver_save)
            index += 1

        for j2 in range(mutated_count[i]):
            mutated_save = mutated_arr[:,j2,:]
            mutated_save.shape = (-1, 4**k)
            _np.savez(f"{store_dir}/pool{i}_mutated_{j2}_index{index_list[index]}.npz", mutated_save)
            index += 1

        gaussian.clear()
        trans.clear()
        traver.clear()
        mutated.clear()
        print(time())

    lengths_arr = lengths.take()
    norm_arr = norm.take()
    norm_arr.shape = (len(norm_arr)//(4**k), 4**k)

    return norm_arr, contignames, lengths_arr
