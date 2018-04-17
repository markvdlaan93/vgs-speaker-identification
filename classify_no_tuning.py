import load_data
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score, make_scorer
import numpy as np

val_conv, val_emb, val_rec, val_spk, val_text, val_mfcc = load_data.dataset()
N_SPLITS = 5

def mfcc():
    """
    First run without further hyperparameter tuning (i.e. loss='log', max_iter=1000, n_jobs=1, learning_rate='optimal'):

    F1-score for fold 1 is 0.8106473829790354
    Accuracy score for fold 1 is 0.7625

    F1-score for fold 2 is 0.8087015604365517
    Accuracy score for fold 2 is 0.77625

    F1-score for fold 3 is 0.7892556020302715
    Accuracy score for fold 3 is 0.72875

    F1-score for fold 4 is 0.7903686634189722
    Accuracy score for fold 4 is 0.71875

    F1-score for fold 5 is 0.8026816159176354
    Accuracy score for fold 5 is 0.765

    Average accuracy over all folds is thus 0.75025
    Average F1-score over all folds is thus 0.80033
    @todo scramble data, randomize the data, the score should then be lower
    :return:
    """
    X_train, _, y_train, _ = train_test_split(val_mfcc, val_spk, test_size=0.2, random_state=123)

    cross_val(X_train, y_train)


def conv():
    X_train, _, y_train, _ = train_test_split(val_conv, val_spk, test_size=0.2, random_state=123)

    cross_val(X_train, y_train)


def rec_layers():
    amount_layers = val_rec.shape[1]
    for i in range(amount_layers):
        layer = val_rec[:,i,:]
        X_train, _, y_train, _ = train_test_split(layer, val_spk, test_size=0.2, random_state=123)

        print("CV results for recurrent layer {}".format(i + 1))
        cross_val(X_train, y_train)


def emb():
    X_train, _, y_train, _ = train_test_split(val_emb, val_spk, test_size=0.2, random_state=123)

    cross_val(X_train, y_train)


def cross_val(X_train, y_train):
    kf = KFold(n_splits=N_SPLITS, random_state=123)

    count = 1
    avg_acc = 0
    avg_f1 = 0
    for train_index, test_index in kf.split(X_train):
        fold_x_train, fold_x_test = X_train[train_index], X_train[test_index]
        fold_y_train, fold_y_test = y_train[train_index], y_train[test_index]

        model = SGDClassifier(loss='log', random_state=123, max_iter=1000, n_jobs=-1)
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

print(mfcc())
# print(conv())