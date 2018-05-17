import sys
sys.path.append('../')

import load_data
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

N_SPLITS = 5

val_conv, val_emb, val_rec, _, val_spk_int, val_text, val_mfcc = load_data.dataset_places()

def mfcc():
    """
    F1-score for fold 1 is 0.8850769494142677
    Accuracy score for fold 1 is 0.835
    F1-score for fold 2 is 0.8638076880496226
    Accuracy score for fold 2 is 0.79
    F1-score for fold 3 is 0.9024598930481283
    Accuracy score for fold 3 is 0.855
    F1-score for fold 4 is 0.8570857044571555
    Accuracy score for fold 4 is 0.78
    F1-score for fold 5 is 0.8880510094145707
    Accuracy score for fold 5 is 0.815
    Average accuracy over all folds is thus 0.8149999999999998
    Average F1-score over all folds is thus 0.879296248876749
    :return:
    """
    cross_val(val_mfcc, val_spk_int)

def conv():
    """
    F1-score for fold 1 is 0.8732839367829226
    Accuracy score for fold 1 is 0.81
    F1-score for fold 2 is 0.8576185275188343
    Accuracy score for fold 2 is 0.8
    F1-score for fold 3 is 0.8788778018457363
    Accuracy score for fold 3 is 0.845
    F1-score for fold 4 is 0.8652758041530276
    Accuracy score for fold 4 is 0.78
    F1-score for fold 5 is 0.8926426089838463
    Accuracy score for fold 5 is 0.82
    Average accuracy over all folds is thus 0.8110000000000002
    Average F1-score over all folds is thus 0.8735397358568735
    :return:
    """
    cross_val(val_conv, val_spk_int)

def rec_layers():
    """
    CV results for recurrent layer 1
    F1-score for fold 1 is 0.9454299293610807
    Accuracy score for fold 1 is 0.90625
    F1-score for fold 2 is 0.9191555478109316
    Accuracy score for fold 2 is 0.8625
    F1-score for fold 3 is 0.9448442223007258
    Accuracy score for fold 3 is 0.9
    F1-score for fold 4 is 0.9271457502578785
    Accuracy score for fold 4 is 0.86875
    F1-score for fold 5 is 0.9387337925419355
    Accuracy score for fold 5 is 0.85
    Average accuracy over all folds is thus 0.8775000000000001
    Average F1-score over all folds is thus 0.9350618484545103

    CV results for recurrent layer 2
    F1-score for fold 1 is 0.9093488474094518
    Accuracy score for fold 1 is 0.86875
    F1-score for fold 2 is 0.8973818652292469
    Accuracy score for fold 2 is 0.83125
    F1-score for fold 3 is 0.9296088005909434
    Accuracy score for fold 3 is 0.86875
    F1-score for fold 4 is 0.9062488079631682
    Accuracy score for fold 4 is 0.83125
    F1-score for fold 5 is 0.8953931717661016
    Accuracy score for fold 5 is 0.80625
    Average accuracy over all folds is thus 0.8412500000000002
    Average F1-score over all folds is thus 0.9075962985917823

    CV results for recurrent layer 3
    F1-score for fold 1 is 0.901866722253507
    Accuracy score for fold 1 is 0.85
    F1-score for fold 2 is 0.9084694672852277
    Accuracy score for fold 2 is 0.84375
    F1-score for fold 3 is 0.907759724107561
    Accuracy score for fold 3 is 0.85625
    F1-score for fold 4 is 0.9100361865957329
    Accuracy score for fold 4 is 0.83125
    F1-score for fold 5 is 0.8943067564208435
    Accuracy score for fold 5 is 0.79375
    Average accuracy over all folds is thus 0.835
    Average F1-score over all folds is thus 0.9044877713325743

    CV results for recurrent layer 4
    F1-score for fold 1 is 0.8910747212859736
    Accuracy score for fold 1 is 0.85
    F1-score for fold 2 is 0.8819588802825482
    Accuracy score for fold 2 is 0.80625
    F1-score for fold 3 is 0.900963718333986
    Accuracy score for fold 3 is 0.85625
    F1-score for fold 4 is 0.9009929268783623
    Accuracy score for fold 4 is 0.81875
    F1-score for fold 5 is 0.8713366131715562
    Accuracy score for fold 5 is 0.7875
    Average accuracy over all folds is thus 0.8237500000000001
    Average F1-score over all folds is thus 0.8892653719904853
    :return:
    """
    amount_layers = val_rec.shape[1]
    for i in range(amount_layers):
        layer = val_rec[:, i, :]
        X_train, _, y_train, _ = train_test_split(layer, val_spk_int, test_size=0.2, random_state=123)

        print("CV results for recurrent layer {}".format(i + 1))
        cross_val(X_train, y_train)

def emb():
    """
    F1-score for fold 1 is 0.8431884343480981
    Accuracy score for fold 1 is 0.795
    F1-score for fold 2 is 0.7760798852278445
    Accuracy score for fold 2 is 0.72
    F1-score for fold 3 is 0.8254740941967751
    Accuracy score for fold 3 is 0.805
    F1-score for fold 4 is 0.7533061905190364
    Accuracy score for fold 4 is 0.7
    F1-score for fold 5 is 0.7786728019329235
    Accuracy score for fold 5 is 0.745
    Average accuracy over all folds is thus 0.7530000000000001
    Average F1-score over all folds is thus 0.7953442812449355
    :return:
    """
    cross_val(val_emb, val_spk_int)

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