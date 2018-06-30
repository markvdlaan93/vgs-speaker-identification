import load_data
import tune
import label_gender

val_conv, val_emb, val_rec, val_spk, val_text, val_mfcc = load_data.dataset()
val_gender = label_gender.create_y_train(val_spk)

def classify():
    """
    Method which calls for every layer the tune method in order to find the right parameters
    :return:
    """
    file = './data/tuning/flickr8k-gender-3.txt'
    test_size = 0.33
    tune.tune(val_mfcc, val_gender, file, test_size, True, True, True)
    tune.tune(val_conv, val_gender, file, test_size, True, True, True)
    amount_layers = val_rec.shape[1]
    for i in range(amount_layers):
        layer = val_rec[:, i, :]
        tune.tune(layer, val_gender, file, test_size, True, True, True)
    tune.tune(val_emb, val_gender, file, test_size, True, True, True)

classify()