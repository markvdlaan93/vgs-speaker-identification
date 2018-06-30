from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import SGDClassifier

def tune(x, y, file, test_size = 0.33, stratification = False, gender_accuracy = False, predict_test = True):
    """
    Generic method which performs grid search in conjunctin with k-fold cross validation in order to find the optimal
    parameters
    :param x:
    :param y:
    :param file: e.g. ./data/tuning/flickr8k-speaker.txt
    :param test_size:
    :param stratification:
    :param gender_accuracy: if True the accuracy score per class will be given. This is relevant for the gender bias
    research
    :param predict_test:
    :return:
    """
    if stratification:
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=123, stratify=y)
    else:
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=123)

    parameters = {
        'loss': ('log', 'hinge'),
        'penalty': ['l1', 'l2', 'elasticnet'],
        'alpha': [0.01, 0.001, 0.0001, 0.00001]
    }

    # Array used to write results to a file
    result = []

    # Weighted because see http://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
    clf = GridSearchCV(SGDClassifier(random_state=123, max_iter=1000), parameters, scoring='f1_weighted', verbose=1, n_jobs=-1)
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

    if predict_test:
        y_true, y_pred = y_test, clf.predict(X_test)
        acc = "Accuracy: {:.4f}".format(accuracy_score(y_true, y_pred))
        f1 = "F1-score: {:.4f}".format(f1_score(y_true, y_pred, average='weighted'))
        print(acc)
        print(f1)
        result.append(acc)
        result.append(f1)

        # Prints accuracy and F1-score per gender
        if gender_accuracy:
            male_acc = calculate_accuracy_per_class(y_true, y_pred, False)
            female_acc = calculate_accuracy_per_class(y_true, y_pred, True)
            with open(file, 'a') as f:
                f.write("{}\n".format("Male accuracy: ".format(male_acc)))
                f.write("{}\n".format("Female accuracy: ").format(female_acc))

                f1_score(y_true, y_pred, average=None)

    with open(file, 'a') as f:
        for row in result:
            f.write("{}\n".format(row))
        f.write("\n")




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