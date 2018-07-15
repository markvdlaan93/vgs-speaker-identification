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
    plt.xlabel('Speaker ID')
    plt.ylabel('Occurrences')
    plt.savefig('../img/places/distribution.png')

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


def gender_speaker_classification_scores():
    """
    Plot male and female accuracy scores for the Flickr8K and Places dataset
    :return:
    """

    # Flickr8K speaker classification F1-scores
    x = [1, 2, 3, 4, 5, 6, 7]
    y = [0.8049, 0.8143, 0.9313, 0.8756, 0.8396, 0.8042, 0.6055]
    my_xticks = ['MFCC', 'Conv.', 'Rec. 1', 'Rec. 2', 'Rec. 3', 'Rec. 4', 'Emb.']
    plt.xticks(x, my_xticks)
    plt.plot(x, y, label='Flickr8K speaker classification (baseline = 7.2%)', linestyle='dashed')

    # Places speaker classification F1-score
    y = [0.7695, 0.8026, 0.8544, 0.7836, 0.7979, 0.7814, 0.7329]
    plt.plot(x, y, label='Places speaker classification (baseline = 36.2%)', linestyle='dashed')

    # Flickr8K gender classification F1-scores
    y = [0.7477, 0.7520, 0.9552, 0.9564, 0.9418, 0.9339, 0.9054]
    plt.plot(x, y, label='Flickr8K gender classification (baseline = 53.95%)')

    # Places gender classification F1-scores
    y = [0.8647, 0.8746, 0.9750, 0.9600, 0.9675, 0.9575, 0.9249]
    plt.plot(x, y, label='Places gender classification (baseline = 50%)')

    build_plot('../img/gender-speaker-classification.png', 0.6)

def gender_speaker_accuracy_classification_scores():
    x = [1, 2, 3, 4, 5, 6, 7]
    y = [0.8300, 0.8340, 0.9406, 0.8939, 0.8630, 0.8291, 0.6461]
    my_xticks = ['MFCC', 'Conv.', 'Rec. 1', 'Rec. 2', 'Rec. 3', 'Rec. 4', 'Emb.']
    plt.xticks(x, my_xticks)
    plt.plot(x, y, label='Flickr8K speaker classification (baseline = 7.2%)', linestyle='dashed')

    # Places speaker classification F1-score
    y = [0.8200, 0.8300, 0.8850, 0.8350, 0.8375, 0.8325, 0.7825]
    plt.plot(x, y, label='Places speaker classification (baseline = 36.2%)', linestyle='dashed')

    # Flickr8K gender classification F1-scores
    y = [0.7491, 0.7521, 0.9552, 0.9564, 0.9418, 0.9339, 0.9055]
    plt.plot(x, y, label='Flickr8K gender classification (baseline = 53.95%)')

    # Places gender classification F1-scores
    y = [0.8650, 0.8750, 0.9750, 0.9600, 0.9675, 0.9575, 0.9250]
    plt.plot(x, y, label='Places gender classification (baseline = 50%)')

    build_plot('../img/gender-speaker-classification-accuracy.png', 0.6)

def gender_bias_scores():
    """
    Gender scores per dataset
    :return:
    """
    # F1-score male Flickr8K
    x = [1, 2, 3, 4, 5, 6, 7]
    y = [0.7767, 0.7719, 0.9584, 0.9594, 0.9459, 0.9388, 0.9131]
    my_xticks = ['MFCC', 'Conv.', 'Rec. 1', 'Rec. 2', 'Rec. 3', 'Rec. 4', 'Emb.']
    plt.xticks(x, my_xticks)
    plt.plot(x, y, label='Flickr8K male', linestyle='dashed')

    # F1-score female Flickr8K
    y = [0.7137, 0.7286, 0.9513, 0.9529, 0.9371, 0.9282, 0.8963]
    plt.plot(x, y, label='Flickr8K female', linestyle='dashed')

    # F1-score male Places
    y = [0.8466, 0.8571, 0.9722, 0.9560, 0.9638, 0.9524, 0.9153]
    plt.plot(x, y, label='Places male')

    # F1-score female Places
    y = [0.8795, 0.8889, 0.9773, 0.9633, 0.9705, 0.9616, 0.9327]
    plt.plot(x, y, label='Places female')

    build_plot('../img/gender-bias.png', 0.7)

def gender_bias_accuracy_scores():
    """
    :return:
    """
    x = [1, 2, 3, 4, 5, 6, 7]
    y = [0.8090, 0.7775, 0.9584, 0.9551, 0.9427, 0.9393, 0.9213]
    my_xticks = ['MFCC', 'Conv.', 'Rec. 1', 'Rec. 2', 'Rec. 3', 'Rec. 4', 'Emb.']
    plt.xticks(x, my_xticks)
    plt.plot(x, y, label='Flickr8K male', linestyle='dashed')

    # Accuracy female Flickr8K
    y = [0.6789, 0.7224, 0.9513, 0.9579, 0.9408, 0.9371, 0.8868]
    plt.plot(x, y, label='Flickr8K female', linestyle='dashed')

    # Accuracy male Places
    y = [0.8278, 0.8333, 0.9722, 0.9667, 0.9611, 0.9444, 0.9000]
    plt.plot(x, y, label='Places male')

    # Accuracy female Places
    y = [0.8955, 0.9091, 0.9773, 0.9545, 0.9727, 0.9682, 0.9455]
    plt.plot(x, y, label='Places female')

    build_plot('../img/gender-bias-accuracy.png', 0.65)


def build_plot(file_name, x_start):
    """
    :param file_name:
    :return:
    """
    plt.axis([0, 8, x_start, 1])
    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    handles, labels = ax.get_legend_handles_labels()
    lgd = ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.1))
    ax.grid('on')
    plt.savefig(file_name, bbox_extra_artists=(lgd,), bbox_inches='tight')


#gender_speaker_classification_scores()
gender_speaker_accuracy_classification_scores()
#gender_bias_scores()
#gender_bias_accuracy_scores()