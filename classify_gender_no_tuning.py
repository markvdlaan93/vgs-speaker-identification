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
    First run without further hyperparameter tuning (i.e. loss='log', max_iter=1000, n_jobs=1, learning_rate='optimal'):

    F1-score for fold 1 is 0.7116523391751348
    Accuracy score for fold 1 is 0.713

    F1-score for fold 2 is 0.7489329686466248
    Accuracy score for fold 2 is 0.75

    F1-score for fold 3 is 0.7071663106499848
    Accuracy score for fold 3 is 0.706

    F1-score for fold 4 is 0.6506206395348836
    Accuracy score for fold 4 is 0.678

    F1-score for fold 5 is 0.7297462061136246
    Accuracy score for fold 5 is 0.738

    Average accuracy over all folds is thus 0.717
    Average F1-score over all folds is thus 0.7096236928240506

    :return:
    """
    print("CV results for MFCC layer")
    cross_val(val_mfcc, val_gender)

def conv():
    """
    F1-score for fold 1 is 0.7153148730857981
    Accuracy score for fold 1 is 0.716

    F1-score for fold 2 is 0.7020664517312784
    Accuracy score for fold 2 is 0.713

    F1-score for fold 3 is 0.7272686688311687
    Accuracy score for fold 3 is 0.732

    F1-score for fold 4 is 0.6737588916252031
    Accuracy score for fold 4 is 0.684

    F1-score for fold 5 is 0.7362115185937139
    Accuracy score for fold 5 is 0.741

    Average accuracy over all folds is thus 0.7172
    Average F1-score over all folds is thus 0.7109240807734324
    :return:
    """
    print("CV results for convolutional layer")
    cross_val(val_conv, val_gender)


def rec_layers():
    """
    CV results for recurrent layer 1
    --------------------------------
    F1-score for fold 1 is 0.9579749990950445
    Accuracy score for fold 1 is 0.958
    F1-score for fold 2 is 0.952
    Accuracy score for fold 2 is 0.952
    F1-score for fold 3 is 0.949163774890928
    Accuracy score for fold 3 is 0.949
    F1-score for fold 4 is 0.9460195102368527
    Accuracy score for fold 4 is 0.946
    F1-score for fold 5 is 0.9510274417397768
    Accuracy score for fold 5 is 0.951
    Average accuracy over all folds is thus 0.9511999999999998
    Average F1-score over all folds is thus 0.9512371451925205

    CV results for recurrent layer 2
    --------------------------------
    F1-score for fold 1 is 0.943957192745297
    Accuracy score for fold 1 is 0.944
    F1-score for fold 2 is 0.9369907167108227
    Accuracy score for fold 2 is 0.937
    F1-score for fold 3 is 0.9291940672510175
    Accuracy score for fold 3 is 0.929
    F1-score for fold 4 is 0.9380079686395475
    Accuracy score for fold 4 is 0.938
    F1-score for fold 5 is 0.9390097263212362
    Accuracy score for fold 5 is 0.939
    Average accuracy over all folds is thus 0.9374
    Average F1-score over all folds is thus 0.9374319343335842

    CV results for recurrent layer 3
    --------------------------------
    F1-score for fold 1 is 0.9238996129977972
    Accuracy score for fold 1 is 0.924
    F1-score for fold 2 is 0.9100273995649718
    Accuracy score for fold 2 is 0.91
    F1-score for fold 3 is 0.9191357733031021
    Accuracy score for fold 3 is 0.919
    F1-score for fold 4 is 0.9179511887093724
    Accuracy score for fold 4 is 0.918
    F1-score for fold 5 is 0.9240402058385034
    Accuracy score for fold 5 is 0.924
    Average accuracy over all folds is thus 0.9190000000000002
    Average F1-score over all folds is thus 0.9190108360827495

    CV results for recurrent layer 4
    --------------------------------
    F1-score for fold 1 is 0.9156144792863965
    Accuracy score for fold 1 is 0.916
    F1-score for fold 2 is 0.9029857066817428
    Accuracy score for fold 2 is 0.903
    F1-score for fold 3 is 0.9090439441800952
    Accuracy score for fold 3 is 0.909
    F1-score for fold 4 is 0.8878945262979971
    Accuracy score for fold 4 is 0.888
    F1-score for fold 5 is 0.9150360346004325
    Accuracy score for fold 5 is 0.915
    Average accuracy over all folds is thus 0.9061999999999999
    Average F1-score over all folds is thus 0.9061149382093328

    :return:
    """
    amount_layers = val_rec.shape[1]
    for i in range(amount_layers):
        layer = val_rec[:, i, :]

        print("CV results for recurrent layer {}".format(i + 1))
        cross_val(layer, val_gender)


def emb():
    """
    F1-score for fold 1 is 0.8999404740358202
    Accuracy score for fold 1 is 0.9

    F1-score for fold 2 is 0.8948515988271352
    Accuracy score for fold 2 is 0.895

    F1-score for fold 3 is 0.9131699289956673
    Accuracy score for fold 3 is 0.913

    F1-score for fold 4 is 0.8939701738518743
    Accuracy score for fold 4 is 0.894

    F1-score for fold 5 is 0.9120264661654136
    Accuracy score for fold 5 is 0.912

    Average accuracy over all folds is thus 0.9028
    Average F1-score over all folds is thus 0.9027917283751821

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

print(conv())
print(rec_layers())
print(emb())