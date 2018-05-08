import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('../')

import load_data
import majority

val_conv, val_emb, val_rec, val_spk, val_text, val_mfcc = load_data.dataset_places()

def distribution():
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

distribution()