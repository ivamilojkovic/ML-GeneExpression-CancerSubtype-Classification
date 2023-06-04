from skmultilearn.problem_transform import LabelPowerset, BinaryRelevance
from skmultilearn.adapt import MLkNN
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.multioutput import ClassifierChain
from sklearn.preprocessing import FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.metrics import label_ranking_loss, label_ranking_average_precision_score
from sklearn.ensemble import RandomForestClassifier
import datetime
import pickle as pkl
from sklearn.model_selection import train_test_split, GridSearchCV
from utils import log_transform
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
import xgboost as xgb

class MultiLabelClassification():
    def __init__(self, X_train, y_train, X_test, y_test):
        
        self.X_train = X_train
        self.X_test = X_test

        #self.labels = np.unique(y_train)
        self.labels = y_train.columns

#        self.y_train = pd.get_dummies(y_train)
#        self.y_test = pd.get_dummies(y_test)
        self.y_train = y_train
        self.y_test = y_test

class MultiLabel_OnevsRest(MultiLabelClassification):

    def __init__(self, X_train, y_train, X_test, y_test):
        super().__init__(X_train, y_train, X_test, y_test)

    def train_test(self, model):

        preds = {}

        # If model XGBoost
        xgb_param = model.get_xgb_params()
        extra = {'objective': 'binary:logistic'}
        xgb_param.update(extra)

        model = xgb.XGBClassifier(**xgb_param)
        
        class_pipe = Pipeline([('clf', OneVsRestClassifier(model))])
        for label in self.labels:
            print('Progress for predicting label {}...'.format(label))

            class_pipe.fit(self.X_train, self.y_train[label])
            pred = class_pipe.predict(self.X_test)
            preds[label] = pred
            
            print('Test accuracy is {}\n'.format(accuracy_score(self.y_test[label], pred)))
            print('Test recall is {}\n'.format(recall_score(self.y_test[label], pred)))
            print('Test precision is {}\n'.format(precision_score(self.y_test[label], pred)))
            print('Test f1 score is {}\n'.format(f1_score(self.y_test[label], pred)))

        return pd.DataFrame(preds, columns=self.labels)
    
class MultiLabel_OnevsOne(MultiLabelClassification):

    def __init__(self, X_train, y_train, X_test, y_test):
        super().__init__(X_train, y_train, X_test, y_test)

    def train_test(self, model):

        # If model XGBoost
        xgb_param = model.get_xgb_params()
        extra = {'objective': 'binary:logistic'}
        xgb_param.update(extra)

        model = xgb.XGBClassifier(**xgb_param)
        class_pipe = Pipeline([('clf', OneVsOneClassifier(model))])

        preds = {}
        for label in self.labels:
            print('Progress for predicting label {}...'.format(label))

            class_pipe.fit(self.X_train, self.y_train[label])
            pred = class_pipe.predict(self.X_test)

            preds[label] = pred
            
            print('Test accuracy is {}\n'.format(accuracy_score(self.y_test[label], pred)))
            print('Test recall is {}\n'.format(recall_score(self.y_test[label], pred)))
            print('Test precision is {}\n'.format(precision_score(self.y_test[label], pred)))
            print('Test f1 score is {}\n'.format(f1_score(self.y_test[label], pred)))

        return pd.DataFrame(preds, columns=self.labels)
    
class MultiLabel_BinaryRelevance(MultiLabelClassification):

    def __init__(self, X_train, y_train, X_test, y_test):
        super().__init__(X_train, y_train, X_test, y_test)

    def train_test(self, model):

        preds = []

        # If model XGBoost
        xgb_param = model.get_xgb_params()
        extra = {'objective': 'binary:logistic'}
        xgb_param.update(extra)

        base = xgb.XGBClassifier(**xgb_param)
        clf = OneVsOneClassifier(base)

        for label in self.labels:
            print('Progress for predicting label {}...'.format(label))

            clf.fit(self.X_train, self.y_train[label])
            pred = clf.predict(self.X_test)

            preds.append(pred)
            
            print('Test accuracy is {}\n'.format(accuracy_score(self.y_test[label], pred)))
            print('Test recall is {}\n'.format(recall_score(self.y_test[label], pred)))
            print('Test precision is {}\n'.format(precision_score(self.y_test[label], pred)))
            print('Test f1 score is {}\n'.format(f1_score(self.y_test[label], pred)))

        return pd.DataFrame(np.transpose(preds), columns=self.labels)
    
    
class MultiLabel_Chains(MultiLabelClassification):

    def __init__(self, X_train, y_train, X_test, y_test):
        super().__init__(X_train, y_train, X_test, y_test)

    def train_test(self, clf = LogisticRegression(solver='saga')):

        classifier = clf
        chain = ClassifierChain(classifier) 

        chain.fit(self.X_train, self.y_train)
        pred = chain.predict(self.X_test)

        print('\nTest accuracy: {}'.format(accuracy_score(self.y_test, pred)))
        print('Test recall: {}'.format(recall_score(self.y_test, pred, average='weighted', zero_division=1)))
        print('Test precision: {}'.format(precision_score(self.y_test, pred, average='weighted', zero_division=1)))
        print('Test f1 score: {}'.format(f1_score(self.y_test, pred, average='weighted', zero_division=1)))

        print('Test label-rank average precision score: {}\n'. 
              format(label_ranking_average_precision_score(self.y_test, pred)))

        return pd.DataFrame(pred, columns=self.y_test.columns)


class MultiLabel_Adapted(MultiLabelClassification):

    def __init__(self, X_train, y_train, X_test, y_test):
        super().__init__(X_train, y_train, X_test, y_test)

    def train_test(self):

        x_train = lil_matrix(self.X_train.values).toarray()
        y_train = lil_matrix(self.y_train.values).toarray()

        classifier = MLkNN(k=20)
        classifier.fit(x_train, y_train)
        pred = classifier.predict(self.X_test.values)

        print('Test label-rank average precision score: {}\n'. 
              format(label_ranking_average_precision_score(self.y_test, pred.toarray())))

        return pd.DataFrame(pred.todense(), columns=self.y_test.columns)


class MultiLabel_PowerSet(MultiLabelClassification):

    def __init__(self, X_train, y_train, X_test, y_test):
        super().__init__(X_train, y_train, X_test, y_test)

    def train_test(self):

        parameters = [
            {
            'classifier': [MultinomialNB()],
            'classifier__alpha': [0.7, 1.0],
            },
            {
            'classifier': [RandomForestClassifier()],
            'classifier__criterion': ['gini', 'entropy'],
            'classifier__n_estimators': [10, 20, 50],
            },
            {
            'classifier': [LogisticRegression()],
            'classifier__tol': [1e-2, 1e-3],
            'classifier__solver': ['lbfgs', 'sag'],
            'classifier__C': [0.01, 0.1, 0.5, 1]
            },

        ]   

        clf = GridSearchCV(LabelPowerset(), parameters, scoring='f1_weighted')
        clf.fit(X=self.X_train.values, y=self.y_train)
        pred = clf.predict(self.X_test.values)

        print('\nTest accuracy: {}'.format(accuracy_score(self.y_test, pred)))
        print('Test recall: {}'.format(recall_score(self.y_test, pred, average='weighted', zero_division=1)))
        print('Test precision: {}'.format(precision_score(self.y_test, pred, average='weighted', zero_division=1)))
        print('Test f1 score: {}'.format(f1_score(self.y_test, pred, average='weighted', zero_division=1)))

        print('Test label-rank average precision score: {}\n'. 
              format(label_ranking_average_precision_score(self.y_test, pred.toarray())))

        return pd.DataFrame(pred.todense(), columns=self.y_test.columns)

