import sys
sys.path.append('../')

import load_data
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from sklearn.metrics import confusion_matrix

N_SPLITS = 5

val_conv, val_emb, val_rec, val_mfcc, val_gender = load_data.dataset_places_gender()

def mfcc():
    """
    F1-score for fold 1 is 0.9000100010001
    Accuracy score for fold 1 is 0.9
    F1-score for fold 2 is 0.8894223046030275
    Accuracy score for fold 2 is 0.89
    F1-score for fold 3 is 0.8999699729756782
    Accuracy score for fold 3 is 0.9
    F1-score for fold 4 is 0.8807096171802055
    Accuracy score for fold 4 is 0.88
    F1-score for fold 5 is 0.9246615721909464
    Accuracy score for fold 5 is 0.9246231155778895
    Average accuracy over all folds is thus 0.8989246231155779
    Average F1-score over all folds is thus 0.8989546935899915

    :return:
    """
    cross_val(val_mfcc, val_gender)

def conv():
    """
    F1-score for fold 1 is 0.8596634615384616
    Accuracy score for fold 1 is 0.86
    F1-score for fold 2 is 0.9
    Accuracy score for fold 2 is 0.9
    F1-score for fold 3 is 0.8396153846153847
    Accuracy score for fold 3 is 0.84
    F1-score for fold 4 is 0.8955321084227761
    Accuracy score for fold 4 is 0.895
    F1-score for fold 5 is 0.8593675447946805
    Accuracy score for fold 5 is 0.8592964824120602
    Average accuracy over all folds is thus 0.870859296482412
    Average F1-score over all folds is thus 0.8708356998742606

    :return:
    """
    cross_val(val_conv, val_gender)

def rec_layers():
    """
    CV results for recurrent layer 1
    F1-score for fold 1 is 0.9749435847840104
    Accuracy score for fold 1 is 0.975
    F1-score for fold 2 is 0.9687512207508104
    Accuracy score for fold 2 is 0.96875
    F1-score for fold 3 is 0.9688205753001308
    Accuracy score for fold 3 is 0.96875
    F1-score for fold 4 is 0.9686794246998692
    Accuracy score for fold 4 is 0.96875
    F1-score for fold 5 is 0.9748068283917339
    Accuracy score for fold 5 is 0.9748427672955975
    Average accuracy over all folds is thus 0.9712185534591196
    Average F1-score over all folds is thus 0.971200326785311

    CV results for recurrent layer 2
    F1-score for fold 1 is 0.9812296449680351
    Accuracy score for fold 1 is 0.98125
    F1-score for fold 2 is 0.9562448712437966
    Accuracy score for fold 2 is 0.95625
    F1-score for fold 3 is 0.956348805420183
    Accuracy score for fold 3 is 0.95625
    F1-score for fold 4 is 0.949928786200348
    Accuracy score for fold 4 is 0.95
    F1-score for fold 5 is 0.9559329423123015
    Accuracy score for fold 5 is 0.9559748427672956
    Average accuracy over all folds is thus 0.9599449685534591
    Average F1-score over all folds is thus 0.959937010028933

    CV results for recurrent layer 3
    F1-score for fold 1 is 0.9749435847840104
    Accuracy score for fold 1 is 0.975
    F1-score for fold 2 is 0.9562517090511349
    Accuracy score for fold 2 is 0.95625
    F1-score for fold 3 is 0.9625595238095238
    Accuracy score for fold 3 is 0.9625
    F1-score for fold 4 is 0.9437122199834065
    Accuracy score for fold 4 is 0.94375
    F1-score for fold 5 is 0.9496136567834682
    Accuracy score for fold 5 is 0.949685534591195
    Average accuracy over all folds is thus 0.957437106918239
    Average F1-score over all folds is thus 0.9574161388823088

    CV results for recurrent layer 4
    F1-score for fold 1 is 0.9874717923920052
    Accuracy score for fold 1 is 0.9875
    F1-score for fold 2 is 0.943717008797654
    Accuracy score for fold 2 is 0.94375
    F1-score for fold 3 is 0.9438770355402353
    Accuracy score for fold 3 is 0.94375
    F1-score for fold 4 is 0.9374109827504352
    Accuracy score for fold 4 is 0.9375
    F1-score for fold 5 is 0.9307517664907597
    Accuracy score for fold 5 is 0.9308176100628931
    Average accuracy over all folds is thus 0.9486635220125785
    Average F1-score over all folds is thus 0.948645717194218
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
    F1-score for fold 1 is 0.9299439102564102
    Accuracy score for fold 1 is 0.93
    F1-score for fold 2 is 0.925051564767895
    Accuracy score for fold 2 is 0.925
    F1-score for fold 3 is 0.9499849864878391
    Accuracy score for fold 3 is 0.95
    F1-score for fold 4 is 0.9450619905113891
    Accuracy score for fold 4 is 0.945
    F1-score for fold 5 is 0.9347066958988202
    Accuracy score for fold 5 is 0.9346733668341709
    Average accuracy over all folds is thus 0.936934673366834
    Average F1-score over all folds is thus 0.9369498295844707

    :return:
    """
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
