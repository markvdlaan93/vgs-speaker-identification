import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('../')

import load_data
import majority

val_conv, val_emb, val_rec, val_spk, val_text, val_mfcc = load_data.dataset_places()

def distribution():
    print(majority.majority(val_spk))

distribution()