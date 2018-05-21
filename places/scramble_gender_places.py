import sys
sys.path.append('../')

import load_data
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

N_SPLITS = 5

val_conv, val_emb, val_rec, val_mfcc, val_gender = load_data.dataset_places_gender()

np.random.shuffle(val_gender)

def mfcc():
    """
    F1-score for fold 1 is 0.5313833992094862
    Accuracy score for fold 1 is 0.545
    F1-score for fold 2 is 0.44524305555555554
    Accuracy score for fold 2 is 0.48
    F1-score for fold 3 is 0.46384982638888883
    Accuracy score for fold 3 is 0.51
    F1-score for fold 4 is 0.518604462474645
    Accuracy score for fold 4 is 0.545
    F1-score for fold 5 is 0.5346624393319082
    Accuracy score for fold 5 is 0.542713567839196
    Average accuracy over all folds is thus 0.5245427135678392
    Average F1-score over all folds is thus 0.4987486365920968


    :return:
    """
    cross_val(val_mfcc, val_gender)

def conv():
    """
    F1-score for fold 1 is 0.5035004730368969
    Accuracy score for fold 1 is 0.6
    F1-score for fold 2 is 0.4539047619047619
    Accuracy score for fold 2 is 0.53
    F1-score for fold 3 is 0.5157242635894321
    Accuracy score for fold 3 is 0.52
    F1-score for fold 4 is 0.4755698529411764
    Accuracy score for fold 4 is 0.53
    F1-score for fold 5 is 0.6090783753851717
    Accuracy score for fold 5 is 0.6130653266331658
    Average accuracy over all folds is thus 0.558613065326633
    Average F1-score over all folds is thus 0.5115555453714877

    :return:
    """
    cross_val(val_conv, val_gender)

def rec_layers():
    """
    CV results for recurrent layer 1
    F1-score for fold 1 is 0.5246274509803921
    Accuracy score for fold 1 is 0.525
    F1-score for fold 2 is 0.4970010171114037
    Accuracy score for fold 2 is 0.49375
    F1-score for fold 3 is 0.505585247439925
    Accuracy score for fold 3 is 0.50625
    F1-score for fold 4 is 0.6162709359605911
    Accuracy score for fold 4 is 0.61875
    F1-score for fold 5 is 0.5351076600343778
    Accuracy score for fold 5 is 0.5345911949685535
    Average accuracy over all folds is thus 0.5356682389937106
    Average F1-score over all folds is thus 0.5357184623053379

    CV results for recurrent layer 2
    F1-score for fold 1 is 0.4906898186559204
    Accuracy score for fold 1 is 0.49375
    F1-score for fold 2 is 0.5715193849467513
    Accuracy score for fold 2 is 0.56875
    F1-score for fold 3 is 0.5414901960784314
    Accuracy score for fold 3 is 0.5375
    F1-score for fold 4 is 0.5133157018899323
    Accuracy score for fold 4 is 0.51875
    F1-score for fold 5 is 0.516259971347739
    Accuracy score for fold 5 is 0.5157232704402516
    Average accuracy over all folds is thus 0.5268946540880504
    Average F1-score over all folds is thus 0.5266550145837549

    CV results for recurrent layer 3
    F1-score for fold 1 is 0.46694373401534534
    Accuracy score for fold 1 is 0.475
    F1-score for fold 2 is 0.5401614450127876
    Accuracy score for fold 2 is 0.5375
    F1-score for fold 3 is 0.5161654135338346
    Accuracy score for fold 3 is 0.5125
    F1-score for fold 4 is 0.5245985532016376
    Accuracy score for fold 4 is 0.53125
    F1-score for fold 5 is 0.560062893081761
    Accuracy score for fold 5 is 0.559748427672956
    Average accuracy over all folds is thus 0.5231996855345912
    Average F1-score over all folds is thus 0.5215864077690732

    CV results for recurrent layer 4
    F1-score for fold 1 is 0.47411152327390216
    Accuracy score for fold 1 is 0.48125
    F1-score for fold 2 is 0.5715193849467513
    Accuracy score for fold 2 is 0.56875
    F1-score for fold 3 is 0.49883743842364525
    Accuracy score for fold 3 is 0.49375
    F1-score for fold 4 is 0.5268857326271867
    Accuracy score for fold 4 is 0.5375
    F1-score for fold 5 is 0.5471698113207547
    Accuracy score for fold 5 is 0.5471698113207547
    Average accuracy over all folds is thus 0.525683962264151
    Average F1-score over all folds is thus 0.5237047781184481


    :return:
    """
    amount_layers = val_rec.shape[1]
    for i in range(amount_layers):
        layer = val_rec[:, i, :]
        X_train, _, y_train, _ = train_test_split(layer, val_gender, test_size=0.2, random_state=123)

        print("CV results for recurrent layer {}".format(i + 1))
        cross_val(X_train, y_train)

def emb():
    """
    F1-score for fold 1 is 0.550690537084399
    Accuracy score for fold 1 is 0.55
    F1-score for fold 2 is 0.5246307629538964
    Accuracy score for fold 2 is 0.53
    F1-score for fold 3 is 0.5302675211067033
    Accuracy score for fold 3 is 0.54
    F1-score for fold 4 is 0.5371966473243069
    Accuracy score for fold 4 is 0.545
    F1-score for fold 5 is 0.5275483277239874
    Accuracy score for fold 5 is 0.5226130653266332
    Average accuracy over all folds is thus 0.5375226130653267
    Average F1-score over all folds is thus 0.5340667592386585

    :return:
    """
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

mfcc()
conv()
rec_layers()
emb()