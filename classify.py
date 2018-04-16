import load_data
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import pprint

val_conv, val_emb, val_rec, val_spk, val_text, val_mfcc = load_data.dataset()

def mfcc():
    """
    First run without further hyperparameter tuning (i.e. loss='log', max_iter=1000, n_jobs=1, learning_rate='optimal')
    Score for fold 1 is 0.8106473829790354
    Score for fold 2 is 0.8087015604365517
    Score for fold 3 is 0.7892556020302715
    Score for fold 4 is 0.7903686634189722
    Score for fold 5 is 0.8026816159176354
    :return:
    """
    X_train, X_test, y_train, y_test = train_test_split(val_mfcc, val_spk, test_size=0.2, random_state=123)

    kf = KFold(n_splits=5, random_state=123)
    result = {}
    count = 1
    
    for train_index, test_index in kf.split(X_train):
        fold_x_train, fold_x_test = X_train[train_index], X_train[test_index]
        fold_y_train, fold_y_test = y_train[train_index], y_train[test_index]

        model = SGDClassifier(loss='log', random_state=123, max_iter=1000, n_jobs=-1)
        model.fit(fold_x_train, fold_y_train)
        y_pred = model.predict(fold_x_test)
        score = f1_score(fold_y_test, y_pred, average='weighted', labels=np.unique(y_pred))
        print("Score for fold {} is {}".format(count, score))
        result[count] = score

        count += 1

    return result


def conv():
    X_train, X_val, y_train, y_val = train_test_split(val_conv, val_spk, test_size=0.2, random_state=123)

    model = SGDClassifier(loss='log', random_state=123, max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    return f1_score(y_val, y_pred, average='weighted', labels=np.unique(y_pred))

print(mfcc())
# print(conv())