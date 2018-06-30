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
    _, _, _, _, val_gender = load_data.dataset_places_gender()

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


def gender_classification_scores():
    """
    Plot male and female accuracy scores for the Flickr8K and Places dataset
    :return:
    """

    # Flickr8K speaker classification F1-scores @todo add real values of Flickr8K speaker identification
    x = [1, 2, 3, 4, 5, 6, 7]
    y = [0.8049, 0.8143, 0.9313, 0.9200, 0.9100, 0.9000, 0.8989]
    my_xticks = ['MFCC', 'Conv.', 'Rec. 1', 'Rec. 2', 'Rec. 3', 'Rec. 4', 'Emb.']
    plt.xticks(x, my_xticks)
    plt.plot(x, y)

    # Flickr8K gender classification F1-scores
    y = [0.7477, 0.7520, 0.9552, 0.9564, 0.9418, 0.9339, 0.9054]
    plt.plot(x, y)

    # Places gender classification F1-scores
    y = [0.8647, 0.8746, 0.9750, 0.9600, 0.9675, 0.9575, 0.9249]
    plt.plot(x, y)

    # Places speaker classification F1-score
    y = [0.7695, 0.8026, 0.8544, 0.7836, 0.7979, 0.7814, 0.7329]
    plt.plot(x, y)

    plt.axis([0, 8, 0.7, 1])
    plt.savefig('../img/gender-classification.png')

gender_classification_scores()