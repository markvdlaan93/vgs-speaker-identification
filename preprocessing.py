from sklearn.preprocessing import OneHotEncoder

def one_hot_encoding(y_train, y_val):
    encoder = OneHotEncoder(sparse=False)
    y_train = encoder.fit_transform(y_train)
    y_val = encoder.transform(y_val)
    return y_train, y_val
