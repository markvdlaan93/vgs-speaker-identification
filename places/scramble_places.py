import sys
sys.path.append('../')

import load_data
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

N_SPLITS = 5

val_conv, val_emb, val_rec, _, val_spk_int, val_text, val_mfcc = load_data.dataset_places()

np.random.shuffle(val_spk_int)

def mfcc():
    """
    F1-score for fold 1 is 0.3494982718108282
    Accuracy score for fold 1 is 0.33
    F1-score for fold 2 is 0.29124842216694313
    Accuracy score for fold 2 is 0.31
    F1-score for fold 3 is 0.35437310924369747
    Accuracy score for fold 3 is 0.325
    F1-score for fold 4 is 0.2813431215834863
    Accuracy score for fold 4 is 0.28
    F1-score for fold 5 is 0.3177570093457944
    Accuracy score for fold 5 is 0.3

    Average accuracy over all folds is thus 0.30900000000000005
    Average F1-score over all folds is thus 0.3188439868301499
    :return:
    """
    cross_val(val_mfcc, val_spk_int)

def conv():
    """
    F1-score for fold 1 is 0.2631600757755151
    Accuracy score for fold 1 is 0.285
    F1-score for fold 2 is 0.1776746252787101
    Accuracy score for fold 2 is 0.145
    F1-score for fold 3 is 0.27310521444970454
    Accuracy score for fold 3 is 0.225
    F1-score for fold 4 is 0.23024512655556698
    Accuracy score for fold 4 is 0.245
    F1-score for fold 5 is 0.22796921540713538
    Accuracy score for fold 5 is 0.21
    Average accuracy over all folds is thus 0.22199999999999998
    Average F1-score over all folds is thus 0.23443085149332638

    :return:
    """
    cross_val(val_conv, val_spk_int)

def rec_layers():
    """
    CV results for recurrent layer 1
    F1-score for fold 1 is 0.25806598125438707
    Accuracy score for fold 1 is 0.2625
    F1-score for fold 2 is 0.2775198993184382
    Accuracy score for fold 2 is 0.25
    F1-score for fold 3 is 0.26822659224273654
    Accuracy score for fold 3 is 0.25625
    F1-score for fold 4 is 0.2628137329084004
    Accuracy score for fold 4 is 0.25625
    F1-score for fold 5 is 0.18196924914636226
    Accuracy score for fold 5 is 0.1875
    Average accuracy over all folds is thus 0.2425
    Average F1-score over all folds is thus 0.2497190909740649

    CV results for recurrent layer 2
    F1-score for fold 1 is 0.19196413994430378
    Accuracy score for fold 1 is 0.19375
    F1-score for fold 2 is 0.2517809979190704
    Accuracy score for fold 2 is 0.23125
    F1-score for fold 3 is 0.2333329081528236
    Accuracy score for fold 3 is 0.2375
    F1-score for fold 4 is 0.22104462616912843
    Accuracy score for fold 4 is 0.21875
    F1-score for fold 5 is 0.22796992481203007
    Accuracy score for fold 5 is 0.225
    Average accuracy over all folds is thus 0.22125000000000003
    Average F1-score over all folds is thus 0.22521851939947127


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
    F1-score for fold 1 is 0.29975259494957285
    Accuracy score for fold 1 is 0.305
    F1-score for fold 2 is 0.26232808994892987
    Accuracy score for fold 2 is 0.28
    F1-score for fold 3 is 0.30435692554806126
    Accuracy score for fold 3 is 0.315
    F1-score for fold 4 is 0.25389164531060654
    Accuracy score for fold 4 is 0.265
    F1-score for fold 5 is 0.30328520089058875
    Accuracy score for fold 5 is 0.315
    Average accuracy over all folds is thus 0.296
    Average F1-score over all folds is thus 0.28472289132955186

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

# mfcc()
# conv()
# rec_layers()
emb()