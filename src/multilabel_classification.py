from skmultilearn.problem_transform import LabelPowerset, BinaryRelevance
from skmultilearn.adapt import MLkNN
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.multioutput import ClassifierChain
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, \
    average_precision_score, label_ranking_average_precision_score, label_ranking_loss
from tqdm import tqdm
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

        base = xgb.XGBClassifier(**xgb_param)
        clf = OneVsOneClassifier(base)

        preds = {}
        for label in self.labels:
            print('Progress for predicting label {}...'.format(label))

            clf.fit(self.X_train, self.y_train[label])
            pred = clf.predict(self.X_test)

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

        preds, prob_preds = [], []

        # If model XGBoost
        if isinstance(model, xgb.XGBClassifier):
            xgb_param = model.get_xgb_params()
            extra = {'objective': 'binary:logistic'}
            xgb_param.update(extra)
            base = xgb.XGBClassifier(**xgb_param)

        clf = BinaryRelevance(base)

        for label in self.labels:
            print('Progress for predicting label {}...'.format(label))

            clf.fit(self.X_train, self.y_train[label].values)
            pred = clf.predict(self.X_test).toarray()
            pred_prob = clf.predict_proba(self.X_test).toarray()

            preds.append(pred)
            prob_preds.append(pred_prob)
            
            print('Test accuracy is {}\n'.format(accuracy_score(self.y_test[label].values, pred)))
            print('Test recall is {}\n'.format(recall_score(self.y_test[label], pred)))
            print('Test precision is {}\n'.format(precision_score(self.y_test[label], pred)))
            print('Test f1 score is {}\n'.format(f1_score(self.y_test[label], pred)))

        # Put preds and prob_preds into suitable shape
        preds = pd.DataFrame(np.transpose(preds)[0], columns=self.labels)
        prob_preds = pd.DataFrame(np.transpose(prob_preds)[0], columns=self.labels)

        # Compute ranking average precision
        rank_avg_prec = label_ranking_average_precision_score(self.y_test, prob_preds)
        print('Ranking average precision: ', rank_avg_prec)

        # Compute average precision score
        avg_prec = average_precision_score(self.y_test, preds)
        print('Average precision: ', avg_prec)

        return preds
    
    
class MultiLabel_Chains(MultiLabelClassification):

    def __init__(self, X_train, y_train, X_test, y_test):
        super().__init__(X_train, y_train, X_test, y_test)

    def train_test(self, model, N=50):

        # Check if model is XGBoost classifier
        if isinstance(model, xgb.XGBClassifier):
            xgb_param = model.get_xgb_params()
            extra = {'objective': 'binary:logistic'}
            xgb_param.update(extra)
            base = xgb.XGBClassifier(**xgb_param)

        # Randomly select the order for N times 
        chains = [ClassifierChain(
            base, 
            order="random", 
            random_state=i) for i in range(N)]
        
        # Train each chain
        for chain in tqdm(chains, desc='Chain: '):
            chain.fit(self.X_train, self.y_train)

        # Compute probability predictions for each chain and average them
        y_pred_chains = np.array([chain.predict_proba(self.X_test) for chain in chains])
        y_pred_ensemble = y_pred_chains.mean(axis=0)

        # Compute ranking average precision
        rank_avg_prec = label_ranking_average_precision_score(self.y_test, y_pred_ensemble)
        print('Ranking average precision: ', rank_avg_prec)

        # Take average probability predictions and compute average predictions
        y_pred = (y_pred_ensemble > 0.5).astype('int')

        # Compute average precision score
        avg_prec = average_precision_score(self.y_test, y_pred)
        print('Average precision: ', avg_prec)

        return pd.DataFrame(y_pred, columns=self.y_test.columns)


class MultiLabel_Adapted(MultiLabelClassification):

    def __init__(self, X_train, y_train, X_test, y_test):
        super().__init__(X_train, y_train, X_test, y_test)

    def train_test(self, model=None):

        x_train = lil_matrix(self.X_train.values).toarray()
        y_train = lil_matrix(self.y_train.values).toarray()

        classifier = MLkNN(k=20)
        classifier.fit(x_train, y_train)

        # Compute predictions and their probabilities
        pred = classifier.predict(self.X_test.values)
        pred_prob = classifier.predict_proba(self.X_test.values)

        # Compute ranking average precision
        rank_avg_prec = label_ranking_average_precision_score(self.y_test, pred_prob.todense())
        print('Ranking average precision: ', rank_avg_prec)

        # Compute average precision score
        avg_prec = average_precision_score(self.y_test, pred.todense())
        print('Average precision: ', avg_prec)

        return pd.DataFrame(pred.todense(), columns=self.y_test.columns)


class MultiLabel_PowerSet(MultiLabelClassification):

    def __init__(self, X_train, y_train, X_test, y_test):
        super().__init__(X_train, y_train, X_test, y_test)

    def train_test(self, model):

        # Create XGBoost instance with previously obtained optimal hyper-parameters
        if isinstance(model, xgb.XGBClassifier):
            xgb_param = model.get_xgb_params()
            extra = {'objective': 'binary:logistic'}
            xgb_param.update(extra)
            model = xgb.XGBClassifier(**xgb_param)

        # create MultiOutputClassifier instance with XGBoost model inside
        multilabel_model = LabelPowerset(model)
        multilabel_model.fit(X=self.X_train.values, y=self.y_train)

        # Compute predictions and their probabilities
        pred = multilabel_model.predict(self.X_test.values)
        pred_prob = multilabel_model.predict_proba(self.X_test.values)

        # Compute ranking average precision
        rank_avg_prec = label_ranking_average_precision_score(self.y_test, pred_prob.todense())
        print('Ranking average precision: ', rank_avg_prec)

        # Compute average precision score
        avg_prec = average_precision_score(self.y_test, pred.todense())
        print('Average precision: ', avg_prec)

        return pd.DataFrame(pred.todense(), columns=self.y_test.columns)

