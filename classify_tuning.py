import load_data
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score, make_scorer
import numpy as np

NUM_TRIALS = 30
val_conv, val_emb, val_rec, val_spk, val_text, val_mfcc = load_data.dataset()


def mfcc_cv_tuning():

    X_train, _, y_train, _ = train_test_split(val_mfcc, val_spk, test_size=0.2, random_state=123)

    p_grid = {
        'loss': ['hinge', 'log', 'perceptron'],
        'learning_rate': ['optimal', 'constant', 'invscaling']
    }
    sgd = SGDClassifier(max_iter=1000, n_jobs=-1)
    non_nested_scores = np.zeros(NUM_TRIALS)
    nested_scores = np.zeros(NUM_TRIALS)

    for i in range(NUM_TRIALS):
        print("BEGIN TRIAL {}".format(i))

        inner_fold = KFold(n_splits=5, random_state=i, shuffle=True)
        outer_fold = KFold(n_splits=5, random_state=i, shuffle=True)

        scorer = make_scorer(f1_score, average='weighted')
        classifier = GridSearchCV(estimator=sgd, param_grid=p_grid, cv=inner_fold, scoring=scorer)
        classifier.fit(X_train, y_train)
        non_nested_scores[i] = classifier.best_score_

        print("BEST SCORE FOR NON NESTED FOLD: {}".format(classifier.best_score_))
        print("BEST PARAMS FOR NON NESTED FOLD: {}".format(classifier.best_params_))

        nested_score = cross_val_score(classifier, X=X_train, y=y_train, cv=outer_fold, scoring=scorer)
        nested_scores[i] = nested_score.mean()

        print("BEST SCORE FOR NESTED FOLD: {}".format(nested_score.mean()))

        print("END TRAIL {}".format(i))

    return non_nested_scores, nested_scores