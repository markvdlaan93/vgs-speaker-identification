import numpy as np
import matplotlib.pyplot as plt

# Values of the activation functions (64 values per speech signal) => (5000,64)
val_conv = np.load('data/flickr8k_val_conv.npy')
# Output layer?? (5000,1024)
val_emb = np.load('data/flickr8k_val_emb.npy')
# For each recurrent layer, the values of the activation functions (1024 values per layer) for each speech signal
# => (5000, 4, 1024)
val_rec = np.load('data/flickr8k_val_rec.npy')
# Per speech signal, the ID of the speaker => (5000,)
val_spk = np.load('data/flickr8k_val_spk.npy')
# Caption of each speech signal => (5000,)
val_text = np.load('data/flickr8k_val_text.npy')
# MFCC vector per speech signal with 37 coefficients => (5000,37)
val_mfcc = np.load('data/flickr8k_val_mfcc.npy')

def shapes(datasets):
    for dataset in datasets:
        print(dataset.shape)
        print(dataset[0])


def majority():
    unique, counts = np.unique(val_spk, return_counts=True)
    counts = np.transpose(np.asarray((unique, counts))).astype(int)
    counts = np.sort(counts, axis=0)

    # Summing of frequencies of all speakers should be equal to the size of the speaker dataset
    total_occurrences = counts[:, 1].sum()
    assert total_occurrences == val_spk.shape[0]

    majority_speaker = counts[counts.shape[0] - 1]

    return counts, (majority_speaker[1] / total_occurrences)

def bar(dataset):
    x = dataset[0,:]
    y = dataset[1,:]
    plt.bar(x,y,align='center')
    plt.xlabel('Speaker ID')
    plt.ylabel('Occurrences')
    for i in dataset:
        plt.hlines(i[0],0,i[1])
    plt.show()

# shapes([val_conv, val_emb, val_rec, val_spk, val_text, val_mfcc])
counts, majority = majority()
bar(counts)