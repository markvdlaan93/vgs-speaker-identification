import matplotlib.pyplot as plt
import numpy as np
import load_data
import label_gender

val_conv, val_emb, val_rec, val_spk, val_text, val_mfcc = load_data.dataset()

def shapes(datasets):
    for dataset in datasets:
        print(dataset.shape)
        print(dataset[0])

def majority():
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

def speaker_occurrences(dataset):
    x = dataset[:,0]
    y = dataset[:,1]
    plt.bar(x,y,align='center')
    plt.xlabel('Speaker ID')
    plt.ylabel('Occurrences')
    plt.show()

def hypothesis():
    x = [-2, -1, 0, 1, 2, 3, 4]
    y = [0.45, 0.35, 0.1, 0.2, 0.35, 0.45, 0.6]
    plt.bar(x,y,align='center')
    plt.xlabel('Learning directly from MFCC (-2), Convolutional layer(-1), Recurrent layer (0 till 4)')
    plt.ylabel('Error rate')
    plt.show()

def results_no_tuning_f1_score():
    avg_f1 = np.array([
        0.80033, 0.80062, 0.9472302716385617, 0.9165205223061058, 0.8791941433923919, 0.8460119386623735, 0.5614487580601043
    ])

    x = [-2, -1, 0, 1, 2, 3, 4]
    plt.bar(x,avg_f1,align='center')
    plt.xlabel('Learning directly from MFCC (-2), Convolutional layer(-1), Recurrent layer (0 till 4)')
    plt.ylabel('F1-score')
    plt.savefig('./img/result_no_tuning_f1_score.png')

def results_no_tuning_acc():
    x = [-2, -1, 0, 1, 2, 3, 4]
    avg_acc = np.array([0.75025, 0.75775, 0.9295, 0.89375, 0.85275, 0.82075, 0.5175])

    plt.bar(x, avg_acc, align='center')
    plt.xlabel('Learning directly from MFCC (-2), Convolutional layer(-1), Recurrent layer (0 till 4)')
    plt.ylabel('Accuracy')
    plt.savefig('./img/result_no_tuning_accuracy.png')

def plot_male_female_dist(val_spk):
    val_spk = val_spk.astype(int)
    male = 0
    female = 0

    counts, maj = majority()
    labels_gender = label_gender.filter_speakers(counts)
    # Two arrays should now have the same size and the same ID's
    assert labels_gender.shape == counts.shape
    assert np.all(labels_gender[:,0] == counts[:,0])



plot_male_female_dist(val_spk)