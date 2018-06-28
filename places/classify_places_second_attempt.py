import sys
sys.path.append('../')
import tune
import load_data

val_conv, val_emb, val_rec, _, val_spk_int, val_text, val_mfcc = load_data.dataset_places()

def classify():
    """
    Method which calls for every layer the tune method in order to find the right parameters
    :return:
    """
    file = '../data/tuning/places-speaker.txt'
    tune.tune(val_mfcc, val_spk_int, file)
    tune.tune(val_conv, val_spk_int, file)
    amount_layers = val_rec.shape[1]
    for i in range(amount_layers):
        layer = val_rec[:, i, :]
        tune.tune(layer, val_spk_int, file)
    tune.tune(val_emb, val_spk_int, file)

classify()