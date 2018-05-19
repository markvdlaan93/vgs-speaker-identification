import numpy as np

ABS_PATH = '/Applications/MAMP/htdocs/master-thesis/'

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
    # MFCC vector per speech signal with 37 coefficients => (1000,13)
    val_mfcc = np.load(ABS_PATH + 'data/places_val_mfcc.npy')

    # Z-score MFCC vectors
    val_mfcc = zscore_mfcc(val_mfcc)

    val_spk_int = speaker_string_to_int(val_spk)

    return val_conv, val_emb, val_rec, val_spk, val_spk_int, val_text, val_mfcc

def dataset_places_gender():
    """
    For classifying gender, it is important to remove rows which aren't labeled (it is just one row)
    :return:
    """
    val_conv   = np.load(ABS_PATH + 'data/places_val_conv.npy')
    val_emb    = np.load(ABS_PATH + 'data/places_val_emb.npy')
    val_rec    = np.load(ABS_PATH + 'data/places_val_rec.npy')
    val_mfcc   = np.load(ABS_PATH + 'data/places_val_mfcc.npy')
    val_spk    = np.load(ABS_PATH + 'data/places_val_spk.npy')
    val_gender = np.load(ABS_PATH + 'data/places_val_gender.npy')

    val_mfcc = zscore_mfcc(val_mfcc)

    ## Filter rows. It is only a single speaker that needs to be removed. So it is no problem to hardcode the speaker
    speaker = 'A1FZB94LK9HWBM'
    indices = []
    count   = 0
    for val_speaker in val_spk:
        if val_speaker.split('_')[1] == speaker:
            indices.append(count)
        count += 1

    ## Now that we have the indices, remove them from all relevant datasets
    val_conv   = np.delete(val_conv, indices, 0)
    val_emb    = np.delete(val_emb, indices, 0)
    val_mfcc   = np.delete(val_mfcc, indices, 0)
    val_gender = np.delete(val_gender, indices, 0)

    # Make sure recurrent layers are also formatted
    amount_layers  = val_rec.shape[1]
    val_rec_result = np.zeros((val_rec.shape[0] - len(indices), val_rec.shape[1], val_rec.shape[2]))
    for i in range(amount_layers):
        layer = val_rec[:, i, :]
        layer = np.delete(layer, indices, 0)
        val_rec_result[:, i, :] = layer

    # Make sure all datasets are of equal length
    assert val_conv.shape[0] == val_emb.shape[0] == val_mfcc.shape[0] == val_gender.shape[0] == val_rec_result.shape[0]

    return val_conv, val_emb, val_rec, val_mfcc, val_gender

def zscore_mfcc(val_mfcc):
    """
    A function which z-scores the raw MFCC vectors
    :param val_mfcc:
    :return:
    """
    return (val_mfcc - val_mfcc.mean(axis=0)) / val_mfcc.std(axis=0)

def speaker_string_to_int(val_spk):
    """
    Turn speaker set into numerical features instead of e.g. places_A1HWI4N1RJGKYY
    :return:
    """
    # Construct dictionary for unique referencing
    count = 0
    val_spk_dict = {}  # Note: length should be 85 like explore_places.distribution()
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

    return val_spk_int.astype(int)

dataset_places_gender()