from sklearn.model_selection import KFold
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score

def train_the_model(X, y):
    X, y = np.array(X), np.array(y)
    kf = KFold(n_splits=5, shuffle=True, random_state=10)

    f1_values, recall_values, precision_values, accuracy_values = [], [], [], []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        clf = MultinomialNB()
        y_pred = clf.fit(X_train, y_train).predict(X_test)

        this_acc = accuracy_score(y_test, y_pred)
        this_f1 = f1_score(y_test, y_pred, average='weighted')
        this_pre = precision_score(y_test, y_pred, average='weighted')
        this_recall = recall_score(y_test, y_pred, average='weighted')

        f1_values.append(this_f1)
        recall_values.append(this_recall)
        precision_values.append(this_pre)
        accuracy_values.append(this_acc)

    avg_acc = np.mean(accuracy_values)
    avg_recall = np.mean(recall_values)
    avg_pre = np.mean(precision_values)
    avg_f1 = np.mean(f1_values)

    print('Accuracy = {}, Recall = {}, Precision = {}, F1 = {}'.format(avg_acc, avg_recall, avg_pre, avg_f1))
