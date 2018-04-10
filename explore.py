import load_data
import matplotlib.pyplot as plt

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

def train_test_split(dataset):
    #80/20 proportion
    return None

# shapes([val_conv, val_emb, val_rec, val_spk, val_text, val_mfcc])
# counts, majority = majority()
# speaker_occurrences(counts)
hypothesis()