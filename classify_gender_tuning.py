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

    :return:
    """
    print("CV results for MFCC layer")
    cross_val(val_mfcc, val_gender)

def conv():
    """

    :return:
    """
    print("CV results for convolutional layer")
    cross_val(val_conv, val_gender)


def rec_layers():
    """

    :return:
    """
    amount_layers = val_rec.shape[1]
    for i in range(amount_layers):
        layer = val_rec[:, i, :]

        print("CV results for recurrent layer {}".format(i + 1))
        cross_val(layer, val_gender)


def emb():
    """

    :return:
    """
    print("CV results for embeddings layer")
    cross_val(val_emb, val_gender)


def cross_val(X_train, y_train):
    kf = KFold(n_splits=N_SPLITS, random_state=123)

    count = 1
    avg_acc = 0
    avg_f1 = 0
    for train_index, test_index in kf.split(X_train):
        for loss in ['hinge', 'log', 'perceptron', 'squared_hinge', 'modified_huber']:
            for learning_rate in ['constant', 'optimal', 'invscaling']:
                fold_x_train, fold_x_test = X_train[train_index], X_train[test_index]
                fold_y_train, fold_y_test = y_train[train_index], y_train[test_index]

                model = SGDClassifier(loss=loss, random_state=123, max_iter=1000, learning_rate=learning_rate)
                model.fit(fold_x_train, fold_y_train)
                y_pred = model.predict(fold_x_test)
                f1 = f1_score(fold_y_test, y_pred, average='weighted', labels=np.unique(y_pred))
                acc = accuracy_score(fold_y_test, y_pred)
                print("F1-score for fold {} is {} with loss function {} and learning rate {}".format(count, f1, loss, learning_rate))
                print("Accuracy score for fold {} is {} with loss function {} and learning rate {}".format(count, acc, loss, learning_rate))

                avg_acc += acc
                avg_f1 += f1
                count += 1

    print("Average accuracy over all folds is thus {}".format(avg_acc / N_SPLITS))
    print("Average F1-score over all folds is thus {}".format(avg_f1 / N_SPLITS))

print(mfcc())
print(conv())
print(rec_layers())
print(emb())