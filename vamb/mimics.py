import numpy as np
import random


def add_noise(x_train):
    """
    Add artificial Gaussian noise to a training sample.
    :param x_train: ndarray with normalized kmers.
    :return: ndarray with gaussian noise.
    """
    n_features = x_train.shape[0]
    index = (np.random.random(n_features) < 0.25).astype('float32')
    mean = np.mean(x_train)
    noise = np.random.normal(-0.05*abs(mean) if abs(mean) < 1e-422 else -0.05, 0.05*abs(mean) if abs(mean) < 1e-4 else 0.05, n_features)
    gaussian_train = x_train + noise * index
    return gaussian_train

# def mutate_kmers(seq, k_dict, k_count, k, positions, mutations):
#     """
#     Deprecated due to low speed
#     Compute the k-mer counts based on mutations.

#     :param seq: Original Sequence to be mutated.
#     :param k_dict: Dictionary with kmers.
#     :param k_count: Array with k-mer counts in the original seq.
#     :param k: word length in the k-mer counts
#     :param positions: Array with the mutated positions.
#     :param mutations: Array with the correspondent mutations.
#     :return: Array with kmer counts of the mutated version
#     """

#     new_count = k_count.copy()

#     for (i, new_bp) in zip(positions, mutations):
#         max_j = min(k, len(seq)-i, i)
#         min_j = min(k, i)
#         for j in range(1, max_j + 1):
#             idx = i - min_j + j
#             kmer = seq[idx: idx + k]
#             new_kmer = list(kmer)
#             new_kmer[-j] = new_bp
#             new_kmer = ''.join(new_kmer)

#             if 'N' in kmer or 'N' in new_kmer:
#                 pass
#             else:
#                 new_count[k_dict[kmer]] -= 1
#                 new_count[k_dict[new_kmer]] += 1

#     return new_count

def base_transition(nucleotide):
    ### >>> list(b'AGCT')
    ### [65, 71, 67, 84]
    base_probability = random.random()
    if nucleotide == 65:
        return 71 if base_probability < 0.027/0.065 else 65
    elif nucleotide == 71:
        return 65
    elif nucleotide == 84:
        return 67 if base_probability < 0.064/0.065 else 84
    elif nucleotide == 67:
        return 84 if base_probability < 0.025/0.065 else 67
    else:
        return 78

def transition(seq, threshold, iter_num=100):
    """
    Mutate Genomic sequence using transitions only.
    :param seq: Original Genomic Sequence.
    :param threshold: probability of NO Transition.
    :return: Mutated Sequence.
    """

    x = np.random.random((iter_num, len(seq)))
    index = np.where(x > threshold)
    """
    Suppose that index in not empty array. I have not find a method to check this.
    """
    seq_array = np.array([seq]*iter_num)
    transition_func = np.frompyfunc(base_transition, 1, 1)
    seq_array[index] = transition_func(seq_array[index])
    return seq_array

def base_transversion(nucleotide):
    ### >>> list(b'AGCT')
    ### [65, 71, 67, 84]
    base_probability = random.random()
    if nucleotide == 65:
        if base_probability < 0.01/0.03:
            return 67
        else:
            return 84
    elif nucleotide == 71:
        return 71
    elif nucleotide == 84:
        if base_probability < 0.01/0.03:
            return 65
        elif 0.01/0.03 < base_probability < 0.02/0.03:
            return 71
        else:
            return 84
    elif nucleotide == 67:
        if base_probability < 0.01/0.03:
            return 65
        else:
            return 67
    else:
        return 78

def transversion(seq, threshold, iter_num=100):
    """
    Mutate Genomic sequence using transversions only.
    :param seq: Original Genomic Sequence.
    :param threshold: Probability of NO Transversion.
    :return: Mutated Sequence.
    """
    ### >>> list(b'AGCT')
    ### [65, 71, 67, 84]

    x = np.random.random((iter_num, len(seq)))
    index = np.where(x > threshold)
    """
    Suppose that index in not empty array. I have not find a method to check this.
    """
    seq_array = np.array([seq]*iter_num)
    transversion_func = np.frompyfunc(base_transversion, 1, 1)
    seq_array[index] = transversion_func(seq_array[index])
    return seq_array

def transition_transversion(seq, threshold_1, threshold_2, iter_num=100):
    """
    Mutate Genomic sequence using transitions and transversions
    :param seq: Original Sequence.
    :param threshold_1: Probability of NO transition.
    :param threshold_2: Probability of NO transversion.
    :return:
    """
    # First transitions Then transversions. Do not call the funcs transition and transversion to save time
    x1 = np.random.random((iter_num, len(seq)))
    index = np.where(x1 > threshold_1)
    """
    Suppose that index in not empty array. I have not find a method to check this.
    """
    seq_array = np.array([seq]*iter_num)
    transition_func = np.frompyfunc(base_transition, 1, 1)
    seq_array[index] = transition_func(seq_array[index])

    x2 = np.random.random((iter_num, len(seq)))
    index = np.where(x2 > threshold_2)
    """
    Suppose that index in not empty array. I have not find a method to check this.
    """
    transversion_func = np.frompyfunc(base_transversion, 1, 1)
    seq_array[index] = transversion_func(seq_array[index])
    return seq_array
