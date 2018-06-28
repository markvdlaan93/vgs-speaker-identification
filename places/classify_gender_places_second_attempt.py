import sys
sys.path.append('../')
import tune
import load_data

val_conv, val_emb, val_rec, val_mfcc, val_gender = load_data.dataset_places_gender()

def classify():
    """
    Method which calls for every layer the tune method in order to find the right parameters
    :return:
    """
    file = '../data/tuning/places-gender.txt'
    tune.tune(val_mfcc, val_gender, file)
    tune.tune(val_conv, val_gender, file)
    amount_layers = val_rec.shape[1]
    for i in range(amount_layers):
        layer = val_rec[:, i, :]
        tune.tune(layer, val_gender, file)
    tune.tune(val_emb, val_gender, file)

classify()