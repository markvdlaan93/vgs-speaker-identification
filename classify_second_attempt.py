import load_data
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.linear_model import SGDClassifier

val_conv, val_emb, val_rec, val_spk, val_text, val_mfcc = load_data.dataset()

def tune(x, y, file):
    """
    Generic method which performs grid search in conjunctin with k-fold cross validation in order to find the optimal
    parameters
    :param x:
    :param y:
    :param file: e.g. ./data/tuning/flickr8k-speaker.txt
    :return:
    """
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=123)
    parameters = {
        'loss': ('log', 'hinge'),
        'penalty': ['l1', 'l2', 'elasticnet'],
        'alpha': [0.01, 0.001, 0.0001, 0.00001]
    }

    # Array used to write results to a file
    result = []

    # Weighted because see http://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
    clf = GridSearchCV(SGDClassifier(random_state=123, max_iter=1000), parameters, scoring='f1_weighted', verbose=1)
    clf.fit(X_train, y_train)

    print("Best params: {}".format(clf.best_params_))
    result.append(clf.best_params_)

    # Check grid scores
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        str = "%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params)
        print(str)
        result.append(str)

    y_true, y_pred = y_test, clf.predict(X_test)
    acc = "Accuracy: {}".format(accuracy_score(y_true, y_pred))
    f1 = "F1-score: {}".format(f1_score(y_true, y_pred, average='weighted'))
    print(acc)
    print(f1)
    result.append(acc)
    result.append(f1)

    with open(file, 'a') as file:
        for row in result:
            file.write("{}\n".format(row))
        file.write("\n")


def classify():
    """
    Method which calls for every layer the tune method in order to find the right parameters
    :return:
    """
    file = './data/tuning/flickr8k-speaker.txt'
    tune(val_mfcc, val_spk, file)
    tune(val_conv, val_spk, file)
    amount_layers = val_rec.shape[1]
    for i in range(amount_layers):
        layer = val_rec[:, i, :]
        tune(layer, val_spk, file)
    tune(val_emb, val_spk, file)

classify()