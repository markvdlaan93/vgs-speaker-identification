import numpy as np

val_conv = np.load('data/flickr8k_val_conv.npy')
val_emb = np.load('data/flickr8k_val_emb.npy')
val_rec = np.load('data/flickr8k_val_rec.npy')
val_spk = np.load('data/flickr8k_val_spk.npy')
val_text = np.load('data/flickr8k_val_text.npy')
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

    return majority_speaker[1] / total_occurrences


#shapes([val_conv, val_emb, val_rec, val_spk, val_text, val_mfcc])
print(majority())