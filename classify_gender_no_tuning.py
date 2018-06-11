import load_data
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
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

    count           = 1
    avg_f1          = 0
    avg_f1_male     = 0
    avg_f1_female   = 0
    avg_acc         = 0
    avg_acc_male    = 0
    avg_acc_female  = 0
    for train_index, test_index in kf.split(X_train):
        fold_x_train, fold_x_test = X_train[train_index], X_train[test_index]
        fold_y_train, fold_y_test = y_train[train_index], y_train[test_index]

        model = SGDClassifier(loss='log', random_state=123, max_iter=1000)
        model.fit(fold_x_train, fold_y_train)
        y_pred = model.predict(fold_x_test)
        f1 = f1_score(fold_y_test, y_pred, average='weighted', labels=np.unique(y_pred))
        acc = accuracy_score(fold_y_test, y_pred)
        # print("F1-score for fold {} is {}".format(count, f1))
        # print("Accuracy score for fold {} is {}".format(count, acc))

        # Calculate accuracy per class
        avg_acc_male   += calculate_accuracy_per_class(fold_y_test, y_pred, False)
        avg_acc_female += calculate_accuracy_per_class(fold_y_test, y_pred, True)

        gender_f1 = f1_score(fold_y_test, y_pred, average=None)
        avg_f1_male     += gender_f1[0]
        avg_f1_female   += gender_f1[1]

        avg_acc += acc
        avg_f1 += f1
        count += 1

    # print("Average accuracy over all folds is thus {}".format(avg_acc / N_SPLITS))
    # print("Average F1-score over all folds is thus {}".format(avg_f1 / N_SPLITS))

    print("Average accuracy male: {}".format(avg_acc_male / N_SPLITS))
    print("Average accuracy female: {}".format(avg_acc_female / N_SPLITS))
    print("Average f1-score male: {}".format(avg_f1_male / N_SPLITS))
    print("Average f1-score female: {}".format(avg_f1_female / N_SPLITS))
    print("")

def calculate_accuracy_per_class(y_true, y_pred, gender):
    """
    In order to verify whether there is any gender bias in data, calculate the accuracy for male and female. The
    classification_report function of sklearn only has precision, recall and F1-score.

    Accuracy = items classified correctly in class / all items in class
    :param y_true:
    :param y_pred:
    :param gender: bool

    Results gender bias research:
    CV results for MFCC layer
    Average accuracy male: 0.7949859387139739
    Average accuracy female: 0.6284809023404451
    Average f1-score male: 0.74959099408591
    Average f1-score female: 0.6628803598698803

    CV results for convolutional layer
    Average accuracy male: 0.8349354765339767
    Average accuracy female: 0.5801735405706209
    Average f1-score male: 0.760627322290188
    Average f1-score female: 0.6522932743167661

    CV results for recurrent layer 1
    Average accuracy male: 0.9486851122861628
    Average accuracy female: 0.9546583109008665
    Average f1-score male: 0.9544453220290906
    Average f1-score female: 0.9473169013138337

    CV results for recurrent layer 2
    Average accuracy male: 0.9402513724781905
    Average accuracy female: 0.934513705987014
    Average f1-score male: 0.9418943348549638
    Average f1-score female: 0.9319680756982873

    CV results for recurrent layer 3
    Average accuracy male: 0.9235925017090321
    Average accuracy female: 0.9136110728771939
    Average f1-score male: 0.9246822780825127
    Average f1-score female: 0.9120846546088025

    CV results for recurrent layer 4
    Average accuracy male: 0.9213751907557979
    Average accuracy female: 0.8884483729672041
    Average f1-score male: 0.9135973164062511
    Average f1-score female: 0.897043362918342

    CV results for embeddings layer
    Average accuracy male: 0.9124966563253031
    Average accuracy female: 0.8919120345012435
    Average f1-score male: 0.9099516789000568
    Average f1-score female: 0.8941730813618882


    :return:
    """
    all_items = 0
    for y in y_true:
        if y == gender:
            all_items += 1

    correctly_classified_items = 0
    for i in range(y_true.shape[0]):
        if y_true[i] == gender and y_true[i] == y_pred[i]:
            correctly_classified_items += 1

    return correctly_classified_items / all_items

mfcc()
conv()
rec_layers()
emb()