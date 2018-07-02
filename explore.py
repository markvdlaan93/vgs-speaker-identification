import matplotlib.pyplot as plt
import numpy as np
import load_data
import label_gender
import majority

val_conv, val_emb, val_rec, val_spk, val_text, val_mfcc = load_data.dataset()

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

def plot_male_female_dist(val_spk):
    male, female = label_gender.count_occurences_male_female(val_spk)

    plt.bar([0,1], [male,female], align='center')
    plt.xticks([0,1], ['Male','Female'])
    plt.savefig('./img/male_female_distribution.png')
    plt.show()

majority.majority(val_spk)
