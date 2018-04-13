import load_data
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

val_conv, val_emb, val_rec, val_spk, val_text, val_mfcc = load_data.dataset()

def mfcc():
    # @todo split X_train further into CV with grid search
    X_train, X_val, y_train, y_val = train_test_split(val_mfcc, val_spk, test_size=0.2, random_state=123)

    model = SGDClassifier(loss='log', random_state=123, max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    return f1_score(y_val, y_pred, average='weighted', labels=np.unique(y_pred))

def conv():
    X_train, X_val, y_train, y_val = train_test_split(val_conv, val_spk, test_size=0.2, random_state=123)

    model = SGDClassifier(loss='log', random_state=123, max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    return f1_score(y_val, y_pred, average='weighted', labels=np.unique(y_pred))

print(conv())