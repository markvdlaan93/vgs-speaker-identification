import numpy as np
import collections

def majority(val_spk):
    """
    :return: counts, majority
    """
    unique, counts = np.unique(val_spk, return_counts=True)
    counts = np.transpose(np.asarray((unique, counts)))

    frequencies = counts[:,1]

    count_single_instances = 0
    for frequency in frequencies:
        if frequency == '1':
            count_single_instances += 1

    # 11 speakers that have only a single entry, stratification needs at least two instances of each speaker
    #print(count_single_instances)

    # Problem was here. Labels were shifted!
    # counts = np.sort(counts, axis=0)
    counts = counts[counts[:,0].argsort()]

    # Summing of frequencies of all speakers should be equal to the size of the speaker dataset
    total_occurrences = counts[:, 1].sum()
    assert total_occurrences == val_spk.shape[0]

    majority_speaker = counts[counts.shape[0] - 1]

    # Majority baseline is 360/5000 = 0.072
    return counts, (majority_speaker[1] / total_occurrences)

def majority_places(val_spk):
    """
    :param val_spk:
    :return:
    """

    counts = collections.Counter(val_spk)

    count_single_instances = 0
    for key, value in counts.items():
        if value == 1:
            count_single_instances += 1
    # 40 speakers with only a single instance, stratification needs at least two instances of each speaker
    #print(count_single_instances)

    # Majority baseline is 0.362
    print("Majority baseline is {}".format(max(counts.values()) / val_spk.shape[0]))

    return counts

#majority(val_spk)
#majority_places(val_spk_int)
