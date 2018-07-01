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
    test_size = 0.4
    tune.tune(val_mfcc, val_gender, file, test_size, True, True, True)
    tune.tune(val_conv, val_gender, file, test_size, True, True, True)
    amount_layers = val_rec.shape[1]
    for i in range(amount_layers):
        layer = val_rec[:, i, :]
        tune.tune(layer, val_gender, file, test_size, True, True, True)
    tune.tune(val_emb, val_gender, file, test_size, True, True, True)

classify()