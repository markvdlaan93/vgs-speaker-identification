import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('../')

import load_data
import majority

val_conv, val_emb, val_rec, val_spk, val_spk_int, val_text, val_mfcc = load_data.dataset_places()

val_gender = np.load('../data/places_val_gender.npy')

def distribution():
    """
    Show distribution of speakers
    :return:
    """
    total = 0
    counts = majority.majority_places(val_spk)

    y_labels = []
    for key, value in counts.items():
        total += value
        y_labels.append(value)

    y_labels = np.array(y_labels)
    y_labels.sort()

    x_labels = list(range(1, len(counts) + 1))

    plt.bar(x_labels, y_labels, align='center')
    plt.xlabel('Speaker')
    plt.ylabel('Occurrences')
    plt.show()

def distribution_gender():
    """
    Display distribution between males and females in the places dataset based on manual labeling.
    :return:
    """
    male    = 0
    female  = 0
    for gender in val_gender:
        if gender:
            male += 1
        else:
            female += 1

    plt.bar([0, 1], [male, female], align='center')
    plt.xticks([0, 1], ['Male', 'Female'])
    plt.savefig('../img/places/male_female_distribution.png')
    plt.show()


