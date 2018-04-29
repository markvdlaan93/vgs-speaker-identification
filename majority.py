import numpy as np

def majority(val_spk):
    """
    :return: counts, majority
    """
    unique, counts = np.unique(val_spk, return_counts=True)
    counts = np.transpose(np.asarray((unique, counts))).astype(int)
    counts = np.sort(counts, axis=0)

    # Summing of frequencies of all speakers should be equal to the size of the speaker dataset
    total_occurrences = counts[:, 1].sum()
    assert total_occurrences == val_spk.shape[0]

    majority_speaker = counts[counts.shape[0] - 1]

    return counts, (majority_speaker[1] / total_occurrences)