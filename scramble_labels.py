import load_data
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score, make_scorer
import numpy as np
import label_gender

val_conv, val_emb, val_rec, val_spk, val_text, val_mfcc = load_data.dataset()
val_gender = label_gender.create_y_train(val_spk)
N_SPLITS = 5

def mfcc():
    """
    First run without further hyperparameter tuning (i.e. loss='log', max_iter=1000, n_jobs=1, learning_rate='optimal')
    but data is scrambled
    F1-score for fold 1 is 0.03595230875542767
    Accuracy score for fold 1 is 0.0275
    F1-score for fold 2 is 0.04123560694906998
    Accuracy score for fold 2 is 0.025
    F1-score for fold 3 is 0.06812128238041879
    Accuracy score for fold 3 is 0.0425
    F1-score for fold 4 is 0.0349006015830018
    Accuracy score for fold 4 is 0.025
    F1-score for fold 5 is 0.03295605434296893
    Accuracy score for fold 5 is 0.035
    Average accuracy over all folds is thus 0.031
    Average F1-score over all folds is thus 0.042633170802177434

    :return:
    """
    X_train, _, y_train, _ = train_test_split(val_mfcc, val_spk, test_size=0.2, random_state=123)
    np.random.shuffle(y_train)

    cross_val(X_train, y_train)

def mfcc_gender():
    """
    F1-score for fold 1 is 0.42870627626748264
    Accuracy score for fold 1 is 0.467

    F1-score for fold 2 is 0.46561584643034676
    Accuracy score for fold 2 is 0.49

    F1-score for fold 3 is 0.5070520252591092
    Accuracy score for fold 3 is 0.517

    F1-score for fold 4 is 0.40030279737903224
    Accuracy score for fold 4 is 0.526

    F1-score for fold 5 is 0.3975579250852344
    Accuracy score for fold 5 is 0.533

    Average accuracy over all folds is thus 0.5065999999999999
    Average F1-score over all folds is thus 0.4398469740842411
    :return:
    """
    np.random.shuffle(val_gender)

    cross_val(val_mfcc, val_gender)

def conv_gender():
    """
    F1-score for fold 1 is 0.5269497813392654
    Accuracy score for fold 1 is 0.527

    F1-score for fold 2 is 0.4601970917459985
    Accuracy score for fold 2 is 0.49

    F1-score for fold 3 is 0.42217014457174035
    Accuracy score for fold 3 is 0.468

    F1-score for fold 4 is 0.44451371069182394
    Accuracy score for fold 4 is 0.513

    F1-score for fold 5 is 0.411133082999001
    Accuracy score for fold 5 is 0.463
    
    Average accuracy over all folds is thus 0.49219999999999997
    Average F1-score over all folds is thus 0.45299276226956586
    :return:
    """
    np.random.shuffle(val_gender)

    cross_val(val_conv, val_gender)

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

conv_gender()