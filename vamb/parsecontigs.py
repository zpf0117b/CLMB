__doc__ = """Calculate tetranucleotide frequency from a FASTA file.

Usage:
>>> with open('/path/to/contigs.fna', 'rb') as filehandle
...     tnfs, contignames, lengths = read_contigs(filehandle)
"""

import math as _math
import sys as _sys
import os as _os
import numpy as _np
import random as _random
import vamb.vambtools as _vambtools
from vamb._vambtools import _kmercounts
from . import mimics
# for debug
from time import time

# This kernel is created in src/create_kernel.py. See that file for explanation
_KERNEL = _vambtools.read_npz(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                              "kernel/kernel4.npz"))

def _project(fourmers, kernel=_KERNEL, k=4):
    "Project fourmers(4-mer) down in dimensionality"
    s = fourmers.sum(axis=1).reshape(-1, 1)
    s[s == 0] = 1.0
    fourmers *= 1/s
    fourmers += -(1/(4**k))
    return _np.dot(fourmers, kernel)

def _convert(raw, projected, k=4):
    "Move data from raw PushArray to projected PushArray, converting it."
    raw_mat = raw.take().reshape(-1, 4**k)
    projected_mat = _project(raw_mat)
    projected.extend(projected_mat.ravel())
    raw.clear()

def _convert_and_project_mat(raw_mat, kernel=_KERNEL, k=4):
    s = raw_mat.sum(axis=1).reshape(-1, 1)
    s[s == 0] = 1.0
    raw_mat *= 1/s
    raw_mat += -(1/(4**k))
    return _np.dot(raw_mat, kernel)

def read_contigs(filehandle, minlength=100, k=4):
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

def read_contigs_augmentation(filehandle, minlength=100, k=4, store_dir="./", backup_iteration=18, augmode=[-1,-1]):
    """Parses a FASTA file open in binary reading mode.

    Input:
        filehandle: Filehandle open in binary mode of a FASTA file
        minlength: Ignore any references shorter than N bases [100]
        backup_iteration: numbers of generation for training
        store_dir: the dir to store the augmentation data
        augmode: the augmentation method. 0 for gaussian noise, 1 for transition, 2 for transversion, 3 for mutation, -1 for all.
        augdatashuffle: whether to shuffle the data from another pool [False, not implemented] 

    Outputs:
        tnfs: An (n_FASTA_entries x 103) matrix of tetranucleotide freq.
        contignames: A list of contig headers
        lengths: A Numpy array of contig lengths
    
    Stores:
        augmentation data (gaussian_noise, transition, transversion, mutation)
    """

    if minlength < 4:
        raise ValueError('Minlength must be at least 4, not {}'.format(minlength))

    norm = _vambtools.PushArray(_np.float32)
    gaussian = _vambtools.PushArray(_np.float32)
    trans = _vambtools.PushArray(_np.float32)
    traver = _vambtools.PushArray(_np.float32)
    mutated = _vambtools.PushArray(_np.float32)

    lengths = _vambtools.PushArray(_np.int)
    contignames = list()
    '''
    # We do not generate the iteration number due to time cost. We just find the minimum augmentation we need for all iteration (backup_iteration)
    # Create backup augmentation pools
    '''
    pool = 2
    gaussian_count, trans_count, traver_count, mutated_count = [0,0], [0,0], [0,0], [0,0]
    # aug_all_method = ['AllAugmentation','GaussianNoise','Transition','Transversion','Mutation']

    # Create projection kernel
    _KERNEL_PROJ = _vambtools.read_npz(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                              f"kernel/kernel{k}.npz"))

    # Count the number of entries
    filehandle.filehandle.seek(0, 0)
    entry_count = _vambtools.count_entry(filehandle)
    print(f'{entry_count} sequences are used for this binning')

    '''If the number of sequences is too large, we might decrease the number of generated augmentation to avoid CLMB being killed.'''
    if entry_count * backup_iteration > 70000000:
        backup_iteration_2 = 70000000 // entry_count
    else:
        backup_iteration_2 = backup_iteration

    # Pool 1
    for i in range(pool):
        
        if augmode[i] == -1:
            # Constraits: transition frequency = 2 * transversion frequency = 4 * gaussian noise frequency
            gaussian_count[i], trans_count[i], traver_count[i], mutated_count[i] = backup_iteration_2 - backup_iteration_2*4//14 - backup_iteration_2*2//14 - backup_iteration_2//2, backup_iteration_2*4//14, backup_iteration_2*2//14, backup_iteration_2//2
        elif augmode[i] == 0:
            gaussian_count[i], trans_count[i], traver_count[i], mutated_count[i] = backup_iteration_2, 0, 0, 0
        elif augmode[i] == 1:
            gaussian_count[i], trans_count[i], traver_count[i], mutated_count[i] = 0, backup_iteration_2, 0, 0
        elif augmode[i] == 2:
            gaussian_count[i], trans_count[i], traver_count[i], mutated_count[i] = 0, 0, backup_iteration_2, 0
        elif augmode[i] == 3:
            gaussian_count[i], trans_count[i], traver_count[i], mutated_count[i] = 0, 0, 0, backup_iteration_2

        '''Generate augmented data for several batches'''
        pool2 = _math.ceil(backup_iteration / backup_iteration_2)
        # Put index for each augmented data
        index = 0
        index_list = list(range(backup_iteration))
        _random.shuffle(index_list)

        for i2 in range(pool2):
            backup_iteration_3 = backup_iteration % backup_iteration_2
            if i2 == pool2 - 1 and backup_iteration_3 != 0:
                # Last batch for generating remaining augmented data
                if augmode[i] == -1:
                    gaussian_count[i], trans_count[i], traver_count[i], mutated_count[i] = backup_iteration_3 - backup_iteration_3*4//14 - backup_iteration_3*2//14 - backup_iteration_3//2, backup_iteration_3*4//14, backup_iteration_3*2//14, backup_iteration_3//2
                elif augmode[i] == 0:
                    gaussian_count[i], trans_count[i], traver_count[i], mutated_count[i] = backup_iteration_3, 0, 0, 0
                elif augmode[i] == 1:
                    gaussian_count[i], trans_count[i], traver_count[i], mutated_count[i] = 0, backup_iteration_3, 0, 0
                elif augmode[i] == 2:
                    gaussian_count[i], trans_count[i], traver_count[i], mutated_count[i] = 0, 0, backup_iteration_3, 0
                elif augmode[i] == 3:
                    gaussian_count[i], trans_count[i], traver_count[i], mutated_count[i] = 0, 0, 0, backup_iteration_3

            '''Reset the file and generator for next reading'''
            filehandle.filehandle.seek(0, 0)
            entries = _vambtools.byte_iterfasta(filehandle)

            for entry in entries:
                if len(entry) < minlength:
                    continue
                
                t = entry.kmercounts(k)
                # t_norm = t / _np.sum(t)
                # _np.add(t_norm, - 1/(2*4**k), out=t_norm)
                # print(t_norm)
                #print(t)
                if i == 0 and i2 == 0:
                    norm.extend(t)

                for j in range(gaussian_count[i]):
                    t_gaussian = mimics.add_noise(t)
                    gaussian.extend(t_gaussian)
                    # print('gaussian',_np.sum(t_gaussian-t_norm))

                # mutations = mimics.transition(entry.sequence, 1 - 0.021, trans_count[i])
                if trans_count[i] != 0:
                    mutations = mimics.transition(entry.sequence, 1 - 0.065, trans_count[i])
                for j in range(trans_count[i]):
                    '''
                    As the function _kmercounts changes the input array at storage, we should reset counts_kmer's storage when using it. 
                    '''
                    counts_kmer = _np.zeros(1 << (2*k), dtype=_np.int32)
                    _kmercounts(bytearray(mutations[j]), k, counts_kmer)
                    # t_trans = counts_kmer / _np.sum(counts_kmer)
                    # _np.add(t_trans, - 1/(2*4**k), out=t_trans)
                    '''
                    Use deepcopy to avoid storage confusing
                    '''
                    trans.extend(counts_kmer.copy())
                    # print('trans',_np.sum(counts_kmer-t))

                # mutations = mimics.transversion(entry.sequence, 1 - 0.0105, traver_count[i])
                if traver_count[i] != 0:
                    mutations = mimics.transversion(entry.sequence, 1 - 0.003, traver_count[i])
                for j in range(traver_count[i]):
                    counts_kmer = _np.zeros(1 << (2*k), dtype=_np.int32)
                    _kmercounts(bytearray(mutations[j]), k, counts_kmer)
                    # t_traver = counts_kmer / _np.sum(counts_kmer)
                    # _np.add(t_traver, - 1/(2*4**k), out=t_traver)
                    traver.extend(counts_kmer.copy())
                    # print('traver',_np.sum(counts_kmer-t))

                # mutations = mimics.transition_transversion(entry.sequence, 1 - 0.014, 1 - 0.007, mutated_count[i])
                if mutated_count[i] != 0:
                    mutations = mimics.transition_transversion(entry.sequence, 1 - 0.065, 1 - 0.003, mutated_count[i])
                for j in range(mutated_count[i]):
                    counts_kmer = _np.zeros(1 << (2*k), dtype=_np.int32)
                    _kmercounts(bytearray(mutations[j]), k, counts_kmer)
                    # t_mutated = counts_kmer / _np.sum(counts_kmer)
                    # _np.add(t_mutated, - 1/(2*4**k), out=t_mutated)
                    mutated.extend(counts_kmer.copy())
                    # print('mutated', len(entry), _np.sum(counts_kmer-t))

                if i == 0 and i2 == 0:
                    lengths.append(len(entry))
                    contignames.append(entry.header)

            # Don't use reshape since it creates a new array object with shared memory
            gaussian_arr = gaussian.take()
            if gaussian_count[i] != 0:
                gaussian_arr.shape = (-1, gaussian_count[i], 4**k)
            trans_arr = trans.take()
            if trans_count[i] != 0:
                trans_arr.shape = (-1, trans_count[i], 4**k)
            traver_arr = traver.take()
            if traver_count[i] != 0:
                traver_arr.shape = (-1, traver_count[i], 4**k)
            mutated_arr = mutated.take()
            if mutated_count[i] != 0:
                mutated_arr.shape = (-1, mutated_count[i], 4**k)
        # AllAugmentation','GaussianNoise','Transition','Transversion','Mutation'
            for j2 in range(gaussian_count[i]):
                gaussian_save = gaussian_arr[:,j2,:]
                gaussian_save.shape = (-1, 4**k)
                _np.savez(f"{store_dir+_os.sep}pool{i}_k{k}_index{index_list[index]}_GaussianNoise_{j2}.npz", _convert_and_project_mat(gaussian_save, _KERNEL_PROJ, k))
                index += 1

            for j2 in range(trans_count[i]):
                trans_save = trans_arr[:,j2,:]
                trans_save.shape = (-1, 4**k)
                _np.savez(f"{store_dir+_os.sep}pool{i}_k{k}_index{index_list[index]}_Transition_{j2}.npz", _convert_and_project_mat(trans_save, _KERNEL_PROJ, k))
                index += 1

            for j2 in range(traver_count[i]):
                traver_save = traver_arr[:,j2,:]
                traver_save.shape = (-1, 4**k)
                _np.savez(f"{store_dir+_os.sep}pool{i}_k{k}_index{index_list[index]}_Transversion_{j2}.npz", _convert_and_project_mat(traver_save, _KERNEL_PROJ, k))
                index += 1

            for j2 in range(mutated_count[i]):
                mutated_save = mutated_arr[:,j2,:]
                mutated_save.shape = (-1, 4**k)
                _np.savez(f"{store_dir+_os.sep}pool{i}_k{k}_index{index_list[index]}_Mutation_{j2}.npz", _convert_and_project_mat(mutated_save, _KERNEL_PROJ, k))
                index += 1

            gaussian.clear()
            trans.clear()
            traver.clear()
            mutated.clear()

            print(time(), backup_iteration, backup_iteration_2, backup_iteration_3)

    lengths_arr = lengths.take()
    norm_arr = norm.take()
    norm_arr.shape = (-1, 4**k)

    return _convert_and_project_mat(norm_arr, _KERNEL_PROJ, k), contignames, lengths_arr
