import numpy as np

ABS_PATH = '/home/mark/Documents/htdocs/master-thesis/'

def dataset():
    # Values of the activation functions (64 values per speech signal) => (5000,64)
    val_conv = np.load(ABS_PATH + 'data/flickr8k_val_conv.npy')
    # Output layer (5000,1024)
    val_emb = np.load(ABS_PATH + 'data/flickr8k_val_emb.npy')
    # For each recurrent layer, the values of the activation functions (1024 values per layer) for each speech signal
    # => (5000, 4, 1024)
    val_rec = np.load(ABS_PATH + 'data/flickr8k_val_rec.npy')
    # Per speech signal, the ID of the speaker => (5000,)
    val_spk = np.load(ABS_PATH + 'data/flickr8k_val_spk.npy')
    # Caption of each speech signal => (5000,)
    val_text = np.load(ABS_PATH + 'data/flickr8k_val_text.npy')
    # MFCC vector per speech signal with 37 coefficients => (5000,37)
    val_mfcc = np.load(ABS_PATH + 'data/flickr8k_val_mfcc.npy')

    return val_conv, val_emb, val_rec, val_spk, val_text, val_mfcc


def dataset_places():
    # Values of the activation functions (64 values per speech signal) => (1000,64)
    val_conv = np.load(ABS_PATH + 'data/places_val_conv.npy')
    # Output layer (1000,1024)
    val_emb = np.load(ABS_PATH + 'data/places_val_emb.npy')
    # For each recurrent layer, the values of the activation functions (1024 values per layer) for each speech signal
    # => (1000, 4, 1024)
    val_rec = np.load(ABS_PATH + 'data/places_val_rec.npy')
    # Per speech signal, the ID of the speaker => (1000,)
    val_spk = np.load(ABS_PATH + 'data/places_val_spk.npy')
    # Caption of each speech signal => (1000,)
    val_text = np.load(ABS_PATH + 'data/places_val_text.npy')
    # MFCC vector per speech signal with 37 coefficients => (1000,13) @todo normalize
    val_mfcc = np.load(ABS_PATH + 'data/places_val_mfcc.npy')

    # Z-score MFCC vectors
    val_mfcc = (val_mfcc - val_mfcc.mean(axis=0)) / val_mfcc.std(axis=0)

    ## Turn speaker set into numerical features instead of e.g. places_A1HWI4N1RJGKYY

    # Construct dictionary for unique referencing
    count = 0
    val_spk_dict = {} # Note: length should be 85 like explore_places.distribution()
    for speaker in val_spk:
        if speaker not in val_spk_dict:
            val_spk_dict[speaker] = count
            count += 1

    # Replace val_spk string values with built dictionary
    count = 0
    val_spk_int = np.zeros(val_spk.shape)
    for speaker in val_spk:
        val_spk_int[count] = val_spk_dict[speaker]
        count += 1

    val_spk_int = val_spk_int.astype(int)

    return val_conv, val_emb, val_rec, val_spk, val_spk_int, val_text, val_mfcc

