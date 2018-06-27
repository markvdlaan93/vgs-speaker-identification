import load_data
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.linear_model import SGDClassifier

val_conv, val_emb, val_rec, val_spk, val_text, val_mfcc = load_data.dataset()

def tune(x, y):
    """
    Generic method which performs grid search in conjunctin with k-fold cross validation in order to find the optimal
    parameters
    :param x:
    :param y:
    :return:
    """
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=123)
    parameters = {
        'loss': ('log', 'hinge'),
        'penalty': ['l1', 'l2', 'elasticnet'],
        'alpha': [0.001, 0.0001, 0.00001, 0.000001]
    }

    # Weighted because see http://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
    clf = GridSearchCV(SGDClassifier(random_state=123, max_iter=1000), parameters, scoring='f1_weighted', verbose=1)
    clf.fit(X_train, y_train)

    print("Best params: {}".format(clf.best_params_))

    y_true, y_pred = y_test, clf.predict(X_test)
    print("Accuracy: {}".format(accuracy_score(y_true, y_pred)))
    print("F1-score: {}".format(f1_score(y_true, y_pred)))
    print()

def classify():
    """
    Method which calls for every layer the tune method in order to find the right parameters
    :return:
    """
    tune(val_mfcc[0:100], val_spk[0:100])
    exit()
    tune(val_conv[0:100], val_spk[0:100])
    amount_layers = val_rec.shape[1]
    for i in range(amount_layers):
        layer = val_rec[:, i, :]
        tune(layer, val_spk)
    tune(val_emb[0:100], val_spk[0:100])

classify()