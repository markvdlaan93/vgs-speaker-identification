import matplotlib.pyplot as plt
import numpy as np
import load_data
import label_gender
import majority

val_conv, val_emb, val_rec, val_spk, val_text, val_mfcc = load_data.dataset()

def shapes(datasets):
    for dataset in datasets:
        print(dataset.shape)
        print(dataset[0])

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
    counts, maj = majority.majority(val_spk)
    labels_gender = label_gender.filter_speakers(counts)
    # Two arrays should now have the same size and the same ID's
    assert labels_gender.shape == counts.shape
    assert np.all(labels_gender[:,0] == counts[:,0])

    male = 0
    female = 0
    for i in range(labels_gender.shape[0]):
        if not labels_gender[i][1]:
            male += counts[i][1]
        else:
            female += counts[i][1]

    assert male + female == val_spk.shape[0]

    plt.bar([0,1], [male,female], align='center')
    plt.xticks([0,1], ['Male','Female'])
    plt.savefig('./img/male_female_distribution.png')
    plt.show()


def count_occurences_male_female(val_spk):
    labels_gender = label_gender.labels_final()

    # Count males and females in labels_gender final count (79 males, 104 males)
    male = 0
    female = 0
    for label in labels_gender:
        if not label[1]:
            male += 1
        else:
            female += 1

    counts, _ = majority.majority(val_spk)

    # Turn counts into dictionary
    counts_dict = {}
    for count in counts:
        counts_dict[count[0]] = count[1]

    # Count occurrences in validation set based on the counts of occurrences (1941 speech fragments with males,
    # 3059 speech fragments with females)
    male = 0
    female = 0
    for label in labels_gender:
        if label[0] in counts_dict:
            count = counts_dict[label[0]]
            print("For {} the count is {} with gender {}".format(label[0], count, label[1]))
            if not label[1]:
                male += count
            else:
                female += count

    return male, female

#count_occurences_male_female(val_spk)

