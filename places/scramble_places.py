import sys
sys.path.append('../')

import load_data
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

N_SPLITS = 5

val_conv, val_emb, val_rec, val_spk, val_text, val_mfcc = load_data.dataset_places()

np.random.shuffle(val_spk)

def mfcc():
    cross_val(val_mfcc, val_spk)

def conv():
    cross_val(val_conv, val_spk)

def rec_layers():
    amount_layers = val_rec.shape[1]
    for i in range(amount_layers):
        layer = val_rec[:, i, :]
        X_train, _, y_train, _ = train_test_split(layer, val_spk, test_size=0.2, random_state=123)

        print("CV results for recurrent layer {}".format(i + 1))
        cross_val(X_train, y_train)

def emb():
    cross_val(val_emb, val_spk)

def cross_val(X_train, y_train):
    kf = KFold(n_splits=N_SPLITS, random_state=123)

    count = 1
    avg_acc = 0
    avg_f1 = 0
    for train_index, test_index in kf.split(X_train):
        fold_x_train, fold_x_test = X_train[train_index], X_train[test_index]
        fold_y_train, fold_y_test = y_train[train_index], y_train[test_index]

        model = SGDClassifier(loss='log', random_state=123, max_iter=1000)
        model.fit(fold_x_train, fold_y_train)
        y_pred = model.predict(fold_x_test)
        f1 = f1_score(fold_y_test, y_pred, average='weighted', labels=np.unique(y_pred))
        acc = accuracy_score(fold_y_test, y_pred)
        print("F1-score for fold {} is {}".format(count, f1))
        print("Accuracy score for fold {} is {}".format(count, acc))

        avg_acc += acc
        avg_f1 += f1
        count += 1

    print("Average accuracy over all folds is thus {}".format(avg_acc / N_SPLITS))
    print("Average F1-score over all folds is thus {}".format(avg_f1 / N_SPLITS))

mfcc()
conv()
rec_layers()
emb()