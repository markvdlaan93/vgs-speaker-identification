import load_data
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score, make_scorer
import numpy as np
import pandas as pd
from pandas import DataFrame

val_conv, val_emb, val_rec, val_spk, val_text, val_mfcc = load_data.dataset()
N_SPLITS = 5

def split_set(data_set):
    """
    Generic method for splitting a dataset into train, validation and test set.
    :param data_set:
    :return:
    """

    df = DataFrame(val_mfcc)
    train_set, validation_set, test_set = np.split(df.sample(frac=1), [int(.6 * len(df)), int(.8 * len(df))])
    return train_set, validation_set, test_set

def mfcc():
    """
    Classification on the MFCC layer
    :return:
    """
    train_set, validation_set, test_set = split_set()


mfcc()