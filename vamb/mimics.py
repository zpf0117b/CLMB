import numpy as np


def add_noise(x_train):
    """
    Add artificial Gaussian noise to a training sample.
    :param x_train: ndarray with normalized kmers.
    :return: ndarray with gaussian noise.
    """
    n_features = x_train.shape[0]
    index = (np.random.random(n_features) < 0.25).astype('float32')
    mean = np.mean(x_train)
    noise = np.random.normal(-0.005*mean, 0.005*mean, n_features)
    gaussian_train = x_train + noise * index
    return gaussian_train

def mutate_kmers(seq, k_dict, k_count, k, positions, mutations):
    """
    Deprecated due to low speed
    Compute the k-mer counts based on mutations.

    :param seq: Original Sequence to be mutated.
    :param k_dict: Dictionary with kmers.
    :param k_count: Array with k-mer counts in the original seq.
    :param k: word length in the k-mer counts
    :param positions: Array with the mutated positions.
    :param mutations: Array with the correspondent mutations.
    :return: Array with kmer counts of the mutated version
    """

    new_count = k_count.copy()

    for (i, new_bp) in zip(positions, mutations):
        max_j = min(k, len(seq)-i, i)
        min_j = min(k, i)
        for j in range(1, max_j + 1):
            idx = i - min_j + j
            kmer = seq[idx: idx + k]
            new_kmer = list(kmer)
            new_kmer[-j] = new_bp
            new_kmer = ''.join(new_kmer)

            if 'N' in kmer or 'N' in new_kmer:
                pass
            else:
                new_count[k_dict[kmer]] -= 1
                new_count[k_dict[new_kmer]] += 1

    return new_count


def transition(seq, threshold):
    """
    Mutate Genomic sequence using transitions only.
    :param seq: Original Genomic Sequence.
    :param threshold: probability of NO Transition.
    :return: Mutated Sequence.
    """
    ### >>> list(b'AGCT')
    ### [65, 71, 67, 84]
    def base_transition(nucleotide):
        if nucleotide == 65:
            return 71
        elif nucleotide == 71:
            return 65
        elif nucleotide == 84:
            return 67
        elif nucleotide == 67:
            return 84

    x = np.random.random(len(seq))
    index = np.where(x > threshold)[0]
    seq_array = np.array(seq)
    transition_func = np.frompyfunc(base_transition, 1, 1)
    seq_array[index] = transition_func(seq_array[index])
    return bytearray(seq_array)


def transversion(seq, threshold):
    """
    Mutate Genomic sequence using transversions only.
    :param seq: Original Genomic Sequence.
    :param threshold: Probability of NO Transversion.
    :return: Mutated Sequence.
    """
    ### >>> list(b'AGCT')
    ### [65, 71, 67, 84]
    def base_transversion(nucleotide):
        if nucleotide == 65 or nucleotide == 71:
            random_number = np.random.uniform()
            if random_number > 0.5:
                return 84
            else:
                return 67
        elif nucleotide == 84 or nucleotide == 67:
            random_number = np.random.uniform()
            if random_number > 0.5:
                return 65
            else:
                return 71

    x = np.random.random(len(seq))
    index = np.where(x > threshold)[0]
    seq_array = np.array(seq)
    transversion_func = np.frompyfunc(base_transversion, 1, 1)
    seq_array[index] = transversion_func(seq_array[index])
    return bytearray(seq_array)



def transition_transversion(seq, threshold_1, threshold_2):
    """
    Mutate Genomic sequence using transitions and transversions
    :param seq: Original Sequence.
    :param threshold_1: Probability of NO transition.
    :param threshold_2: Probability of NO transversion.
    :return:
    """
    # First transitions Then transversions.

    return transversion(transition(seq, threshold_1), threshold_2)
