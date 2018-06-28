import load_data
import tune

val_conv, val_emb, val_rec, val_spk, val_text, val_mfcc = load_data.dataset()


def classify():
    """
    Method which calls for every layer the tune method in order to find the right parameters
    :return:
    """
    file = './data/tuning/flickr8k-speaker-2.txt'
    tune.tune(val_mfcc, val_spk, file)
    tune.tune(val_conv, val_spk, file)
    amount_layers = val_rec.shape[1]
    for i in range(amount_layers):
        layer = val_rec[:, i, :]
        tune.tune(layer, val_spk, file)
    tune.tune(val_emb, val_spk, file)

classify()