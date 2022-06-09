__doc__ = """Calculate tetranucleotide frequency from a FASTA file.

Usage:
>>> with open('/path/to/contigs.fna', 'rb') as filehandle
...     tnfs, contignames, lengths = read_contigs(filehandle)
"""

import sys as _sys
import os as _os
import numpy as _np
from itertools import product
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

def read_contigs_augmentation(filehandle, minlength=100, store_dir="./", backup_iteration=100, usecuda=False):
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

    entries = _vambtools.byte_iterfasta(filehandle)
    count = 0
    for entry in entries:
        if len(entry) < minlength:
            continue
        count += 1
        t = entry.kmercounts(k)
        t_norm = t / _np.sum(t)
        _np.add(t_norm, - 1/(2*4**k), out=t_norm)
        norm.extend(t_norm)

        for i in range(backup_iteration):
            t_gaussian = mimics.add_noise(t_norm)
            gaussian.extend(t_gaussian)
            print(sum(t_gaussian),end=' ')

        mutations = mimics.transition(entry.sequence, 1 - 0.021, backup_iteration)
        for i in range(backup_iteration):
            _kmercounts(bytearray(mutations[i]), k, counts_kmer)
            t_trans = counts_kmer / _np.sum(counts_kmer)
            _np.add(t_trans, - 1/(2*4**k), out=t_trans)
            trans.extend(t_trans)
            print(sum(t_trans),end=' ')

        mutations = mimics.transversion(entry.sequence, 1 - 0.0105, backup_iteration)
        for i in range(backup_iteration):
            _kmercounts(bytearray(mutations[i]), k, counts_kmer)
            t_traver = counts_kmer / _np.sum(counts_kmer)
            _np.add(t_traver, - 1/(2*4**k), out=t_traver)
            traver.extend(t_traver)
            print(sum(t_traver),end=' ')

        mutations = mimics.transition_transversion(entry.sequence, 1 - 0.014, 1 - 0.007, backup_iteration)
        for i in range(backup_iteration):
            _kmercounts(bytearray(mutations[i]), k, counts_kmer)
            t_mutated = counts_kmer / _np.sum(counts_kmer)
            _np.add(t_mutated, - 1/(2*4**k), out=t_mutated)
            mutated.extend(t_mutated)
            print(sum(t_mutated),end=' ')

        lengths.append(len(entry))
        contignames.append(entry.header)
        print(count, time())
    # Don't use reshape since it creates a new array object with shared memory
    norm_arr = norm.take()
    norm_arr.shape = (len(norm_arr)//(4**k), 4**k)
    gaussian_arr = gaussian.take()
    gaussian_arr.shape = (-1, backup_iteration, 4**k)
    trans_arr = trans.take()
    trans_arr.shape = (-1, backup_iteration, 4**k)
    traver_arr = traver.take()
    traver_arr.shape = (-1, backup_iteration, 4**k)
    mutated_arr = mutated.take()
    mutated_arr.shape = (-1, backup_iteration, 4**k)

    for i in range(backup_iteration):
        gaussian_save = gaussian_arr[:,i,:]
        gaussian_save.shape = (-1, 4**k)
        trans_save = trans_arr[:,i,:]
        trans_save.shape = (-1, 4**k)
        traver_save = traver_arr[:,i,:]
        traver_save.shape = (-1, 4**k)
        mutated_save = mutated_arr[:,i,:]
        mutated_save.shape = (-1, 4**k)
        _np.savez(f"{store_dir}/backup_arr_iter{i}.npz", gaussian_save, trans_save, traver_save, mutated_save)

    lengths_arr = lengths.take()

    return norm_arr, contignames, lengths_arr