import load_data
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score, make_scorer
import numpy as np

val_conv, val_emb, val_rec, val_spk, val_text, val_mfcc = load_data.dataset()
N_SPLITS = 5