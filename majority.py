import numpy as np
import collections

def majority(val_spk):
    """
    :return: counts, majority
    """
    unique, counts = np.unique(val_spk, return_counts=True)
    counts = np.transpose(np.asarray((unique, counts)))
    # Problem was here. Labels were shifted!
    # counts = np.sort(counts, axis=0)
    counts = counts[counts[:,0].argsort()]

    # Summing of frequencies of all speakers should be equal to the size of the speaker dataset
    total_occurrences = counts[:, 1].sum()
    assert total_occurrences == val_spk.shape[0]

    majority_speaker = counts[counts.shape[0] - 1]

    return counts, (majority_speaker[1] / total_occurrences)

def majority_places(val_spk):
    """
    :param val_spk:
    :return:
    """
    return collections.Counter(val_spk)


