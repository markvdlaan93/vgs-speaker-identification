import load_data
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import SGDClassifier

val_conv, val_emb, val_rec, val_spk, val_text, val_mfcc = load_data.dataset()
N_SPLITS = 5

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

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))
    print()

def classify():
    """
    Method which calls for every layer the tune method in order to find the right parameters
    :return:
    """
    tune(val_mfcc[0:100], val_spk[0:100])
    tune(val_conv[0:100], val_spk[0:100])
    amount_layers = val_rec.shape[1]
    for i in range(amount_layers):
        layer = val_rec[:, i, :]
        tune(layer, val_spk)
    tune(val_emb[0:100], val_spk[0:100])