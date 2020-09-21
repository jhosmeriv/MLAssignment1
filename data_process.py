import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def get_titanic():
    train_path = os.path.join('.data', 'titanic', 'train.csv')
    pdf_train = pd.read_csv(train_path, index_col='PassengerId')
    pdf_train = titanic_transform(pdf_train)

    titanic_datasets = iceberg(pdf_train, impute_cols=['Age'], one_hot_cols=['Pclass', 'Sex', 'Embarked',
                                                                             'Cabin_letter'])

    return titanic_datasets


def titanic_transform(frame):
    # Make features from name and cabin
    frame['Name_len'] = frame['Name'].apply(lambda x: len(x))
    frame['Cabin_letter'] = frame['Cabin'].apply(lambda x: cabin_function(x))

    # Drop text columns
    frame = frame.drop(['Name', 'Cabin', 'Ticket'], axis=1)

    return frame


# This splits the titanic. If you don't like this joke you don't have a good sense of humor
def iceberg(frame, pred_col='Survived', train_val_test=(.8, .1, .1), impute_cols=None, one_hot_cols=None):
    # It should be safe to one_hot_encode before splitting since we won't know effect of new values
    frame = pd.get_dummies(frame, columns=one_hot_cols)

    X = frame.drop(pred_col, 1)
    y = frame[pred_col]

    # Split before we impute or min_max scale
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=train_val_test[2], random_state=0)

    # Impute here so we aren't looking at test for our mean
    if impute_cols is not None:
        for col in impute_cols:
            average = X_train[col].mean()
            X_train[col] = X_train[col].fillna(average)
            X_test[col] = X_test[col].fillna(average)

    # Scale
    min_max_scaler = MinMaxScaler()
    min_max_scaler.fit(X_train)

    X_train = pd.DataFrame(data=min_max_scaler.transform(X_train), index=X_train.index, columns=X_train.columns)
    X_test = pd.DataFrame(data=min_max_scaler.transform(X_test), index=X_test.index, columns=X_test.columns)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                      test_size=train_val_test[1]/(train_val_test[0]+train_val_test[1]),
                                                      random_state=0)

    return X_train, X_val, X_test, y_train, y_val, y_test


def cabin_function(cabin):
    if pd.isna(cabin):
        return 'UNK'
    else:
        return cabin[0]


def get_diabetes(test_size=.2):
    data_path = os.path.join('.data', 'diabetes', 'diabetes_data_upload.csv')
    frame = pd.read_csv(data_path)

    frame = diabetes_transform(frame)

    X_train, X_test, y_train, y_test = iceberg(frame, pred_col='class', test_size=test_size, impute_cols=['Age'])

    return X_train, X_test, y_train, y_test


def diabetes_transform(frame):
    # One hot encode almost everything
    frame = pd.get_dummies(frame, columns=['Gender', 'Polyuria', 'Polydipsia', 'sudden weight loss', 'weakness',
                                           'Polyphagia', 'Genital thrush', 'visual blurring', 'Itching', 'Irritability',
                                           'delayed healing', 'partial paresis', 'muscle stiffness', 'Alopecia',
                                           'Obesity'])

    # Convert class data to 0, 1
    frame['class'] = frame['class'].apply(lambda x: int(x == 'Positive'))

    # Scale
    min_max_scaler = MinMaxScaler()
    frame = pd.DataFrame(data=min_max_scaler.fit_transform(frame), index=frame.index, columns=frame.columns)

    return frame


def get_wine(test_size=.2):
    data_path = os.path.join('.data', 'wine', 'winequality-red.csv')
    frame = pd.read_csv(data_path, sep=';')

    frame = wine_transform(frame)

    X_train, X_test, y_train, y_test = iceberg(frame, pred_col='quality', test_size=test_size)

    return X_train, X_test, y_train, y_test


def wine_transform(frame):
    # Scale
    cols = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide',
            'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']
    min_max_scaler = MinMaxScaler()

    data = min_max_scaler.fit_transform(frame.loc[:, cols])
    labels = frame['quality'].values
    all_data = np.hstack([data, labels.reshape((-1, 1))])

    frame = pd.DataFrame(data=all_data, index=frame.index, columns=frame.columns)

    return frame


def get_credit():
    data_path = os.path.join('.data', 'credit', 'crx.data')
    frame = pd.read_csv(data_path, header=None, na_values='?')
    frame.columns = ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "A10", "A11", "A12", "A13", "A14", "A15",
                     "class"]

    frame = credit_transform(frame)

    credit_datasets = iceberg(frame, pred_col='class', impute_cols=['A2', 'A14'],
                                               one_hot_cols=["A1", "A4", "A5", "A6", "A7", "A9", "A10", "A12", "A13"])

    return credit_datasets


def credit_transform(frame):
    """
    A1:	b, a.
    A2:	continuous.
    A3:	continuous.
    A4:	u, y, l, t.
    A5:	g, p, gg.
    A6:	c, d, cc, i, j, k, m, r, q, w, x, e, aa, ff.
    A7:	v, h, bb, j, n, z, dd, ff, o.
    A8:	continuous.
    A9:	t, f.
    A10:	t, f.
    A11:	continuous.
    A12:	t, f.
    A13:	g, p, s.
    A14:	continuous.
    A15:	continuous.
    """
    # Convert class data to 0, 1
    frame['class'] = frame['class'].apply(lambda x: int(x == '+'))

    return frame


