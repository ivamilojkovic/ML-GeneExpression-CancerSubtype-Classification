from skmultilearn.problem_transform import LabelPowerset, BinaryRelevance, ClassifierChain
from skmultilearn.adapt import MLkNN, MLTSVM, MLARAM
from skmultilearn.ensemble import RakelD, RakelO
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.multioutput import ClassifierChain as CC
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, \
    average_precision_score, label_ranking_average_precision_score, label_ranking_loss
from tqdm import tqdm
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
import xgboost as xgb
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import itertools
from skmultilearn.ensemble import LabelSpacePartitioningClassifier
from config_model import MULTILABEL_MODEL_PARAMS
import pickle, time, datetime

class MultiLabelClassification():
    def __init__(self, X_train, y_train, X_test, y_test):
        
        self.X_train = X_train
        self.X_test = X_test

        self.labels = y_train.columns
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

    def train_test(self, model, optimize_model: bool = False) -> pd.DataFrame:

        preds, prob_preds = [], []

        # If model XGBoost
        if isinstance(model, xgb.XGBClassifier):
            xgb_param = model.get_xgb_params()
            extra = {'objective': 'binary:logistic'}
            xgb_param.update(extra)
            base = xgb.XGBClassifier(**xgb_param)
        elif isinstance(model, SVC):
            model.probability = True
            base = model
        else:
            base = model

        clf = BinaryRelevance(base)

        if optimize_model:
            if isinstance(model, LogisticRegression):
                model_name = 'Logistic Regression'
            elif isinstance(model, RandomForestClassifier):
                model_name = 'Random Forest'
            elif isinstance(model, xgb.XGBClassifier):
                model_name = 'XGBoost'
            elif isinstance(model, SVC):
                model_name = 'SVC'

            additional_params = {}
            for key, value in MULTILABEL_MODEL_PARAMS[model_name].items():
                new_key = 'classifier__' + key
                additional_params[new_key] = value
        
            # Create the grid search object
            grid_search = GridSearchCV(clf, param_grid=additional_params, scoring='f1_weighted', verbose=3, cv=5)

            # Fit the grid search to the data
            grid_search.fit(self.X_train, self.y_train)

            # Get the best model
            clf = grid_search.best_estimator_
            print(grid_search.best_params_)

            now = datetime.datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
            with open('ml_best_model_lr' + now + '.pkl', 'wb') as file:
                pickle.dump(clf, file)

            pred = clf.predict(self.X_test).toarray()
            pred_prob = clf.predict_proba(self.X_test).toarray()

            # Put preds and prob_preds into suitable shape
            preds = pd.DataFrame(pred, columns=self.labels)
            prob_preds = pd.DataFrame(pred_prob, columns=self.labels)

        else:
            for label in self.labels:

                clf.fit(self.X_train, self.y_train[label].values)
                pred = clf.predict(self.X_test).toarray()
                pred_prob = clf.predict_proba(self.X_test).toarray()

                preds.append(pred)
                prob_preds.append(pred_prob)

            # Put preds and prob_preds into suitable shape
            preds = pd.DataFrame(np.transpose(preds)[0], columns=self.labels)
            prob_preds = pd.DataFrame(np.transpose(prob_preds)[0], columns=self.labels)

        return preds, prob_preds

class MultiLabel_Chains(MultiLabelClassification):

    def __init__(self, X_train, y_train, X_test, y_test):
        super().__init__(X_train, y_train, X_test, y_test)

    def train_test(self, model, optimize: bool = False, optimize_model: bool = False):

        # Check if model is XGBoost classifier
        if isinstance(model, xgb.XGBClassifier):
            xgb_param = model.get_xgb_params()
            extra = {'objective': 'binary:logistic'}
            xgb_param.update(extra)
            base = xgb.XGBClassifier(**xgb_param)
        elif isinstance(model, SVC):
            model.probability = True
            base = model
        else:
            base = model

        # Create classifier chain 
        chain = ClassifierChain(classifier=base, order=[0, 1, 2, 3, 4])

        if optimize:
            # Define the labels
            labels = self.y_train.columns

            # Generate all possible permutations of label orders
            all_orders = list(itertools.permutations([0, 1, 2, 3, 4]))
            all_orders = [list(order) for order in all_orders]

            # Define the parameter grid for hyperparameter tuning
            param_grid = {
                'order': all_orders # Order in which labels are chained
                }
            
            # Create the grid search object
            grid_search = GridSearchCV(chain, param_grid=param_grid, scoring='f1_weighted', verbose=3)

            # Fit the grid search to the data
            grid_search.fit(self.X_train, self.y_train)

            # Get best model
            chain = grid_search.best_estimator_
        else:
            chain.fit(self.X_train, self.y_train)

        if optimize_model:
            if isinstance(model, LogisticRegression):
                model_name = 'Logistic Regression'
            elif isinstance(model, RandomForestClassifier):
                model_name = 'Random Forest'
            elif isinstance(model, xgb.XGBClassifier):
                model_name = 'XGBoost'
            elif isinstance(model, SVC):
                model_name = 'SVC'

            additional_params = {}
            for key, value in MULTILABEL_MODEL_PARAMS[model_name].items():
                new_key = 'classifier__' + key
                additional_params[new_key] = value
        
            # Create the grid search object
            grid_search = GridSearchCV(chain, param_grid=additional_params, scoring='f1_weighted', cv=5, verbose=3)

            # Fit the grid search to the data
            grid_search.fit(self.X_train, self.y_train)

            # Get best model
            chain = grid_search.best_estimator_

            print(grid_search.best_params_)

            now = datetime.datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
            with open('ml_chain_best_model_lr' + now + '.pkl', 'wb') as file:
                pickle.dump(chain, file)

        # Compute probability predictions and predictions
        y_pred = chain.predict(self.X_test).toarray()
        y_pred_prob = chain.predict_proba(self.X_test).toarray()
        
        return pd.DataFrame(y_pred, columns=self.y_test.columns, dtype='int'),\
              pd.DataFrame(y_pred_prob, columns=self.y_test.columns)
        

class MultiLabel_PowerSet(MultiLabelClassification):

    def __init__(self, X_train, y_train, X_test, y_test):
        super().__init__(X_train, y_train, X_test, y_test)

    def train_test(self, model, optimize_model: bool = False):

        # Create XGBoost instance with previously obtained optimal hyper-parameters
        if isinstance(model, xgb.XGBClassifier):
            xgb_param = model.get_xgb_params()
            extra = {'objective': 'binary:logistic'}
            xgb_param.update(extra)
            model = xgb.XGBClassifier(**xgb_param)
        elif isinstance(model, SVC):
            model.probability = True

        # create MultiOutputClassifier instance with XGBoost model inside
        clf = LabelPowerset(model)

        if optimize_model:
            if isinstance(model, LogisticRegression):
                model_name = 'Logistic Regression'
            elif isinstance(model, RandomForestClassifier):
                model_name = 'Random Forest'
            elif isinstance(model, xgb.XGBClassifier):
                model_name = 'XGBoost'
            elif isinstance(model, SVC):
                model_name = 'SVC'

            additional_params = {}
            for key, value in MULTILABEL_MODEL_PARAMS[model_name].items():
                new_key = 'classifier__' + key
                additional_params[new_key] = value

            # Create the grid search object
            grid_search = GridSearchCV(clf, param_grid=additional_params, scoring='f1_weighted', verbose=3, cv=5)

            # Fit the grid search to the data
            grid_search.fit(self.X_train, self.y_train)

            # Get the best model
            clf = grid_search.best_estimator_
            print(grid_search.best_params_)

            now = datetime.datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
            with open('ml_best_model_lr' + now + '.pkl', 'wb') as file:
                pickle.dump(clf, file)

            pred = clf.predict(self.X_test).toarray()
            pred_prob = clf.predict_proba(self.X_test).toarray()
        else:
            clf.fit(X=self.X_train.values, y=self.y_train)

            # Compute predictions and their probabilities
            pred = clf.predict(self.X_test.values)
            pred_prob = clf.predict_proba(self.X_test.values)

        return pd.DataFrame(pred, columns=self.y_test.columns),\
              pd.DataFrame(pred_prob, columns=self.y_test.columns)

class MultiLabel_Adapted(MultiLabelClassification):

    def __init__(self, X_train, y_train, X_test, y_test):
        super().__init__(X_train, y_train, X_test, y_test)

    def train_test(self, type: str='MLkNN', optimize: bool=False):

        if type == 'MLkNN':

            classifier = MLkNN(k=20)
            params = {
                'k': range(10, 100, 10),
                's': [0.1, 0.3, 0.5, 1]
            }

            x_train = lil_matrix(self.X_train.values).toarray()
            y_train = lil_matrix(self.y_train.values).toarray()

        elif type == 'MLTSVM':
            classifier = MLTSVM(c_k=0.1, threshold=0.5)

            x_train = lil_matrix(self.X_train.values)
            y_train = lil_matrix(self.y_train.values)

        elif type == 'MLARAM':

            classifier = MLARAM(threshold=0.05, vigilance=0.9)

            params = {
                'threshold': [i/10 for i in range(11)],
                'vigilance': [0.7, 0.8, 0.85, 0.9, 0.95],
                # 'neurons': [[10, 10, 20], [5, 5, 5], [30, 20, 10]]
            }
            x_train = lil_matrix(self.X_train.values).toarray()
            y_train = lil_matrix(self.y_train.values).toarray()

        if optimize:
            gs = GridSearchCV(
                estimator=classifier, 
                param_grid=params, 
                scoring='f1_weighted', 
                cv=5, verbose=3)
            
            gs.fit(x_train, y_train)
            print(gs.best_params_)
            classifier = gs.best_estimator_
        else:
            classifier.fit(x_train, y_train)

        # Compute predictions and their probabilities
        pred = classifier.predict(self.X_test.values)
        pred_prob = classifier.predict_proba(self.X_test.values)

        if type != 'MLARAM':
            pred_prob = pred_prob.todense()
            pred = pred.todense()

        return pd.DataFrame(pred, columns=self.y_test.columns, dtype='int'), \
            pd.DataFrame(pred_prob, columns=self.y_test.columns)
    

class MultiLabel_EnsembleChains(MultiLabelClassification):

    def __init__(self, X_train, y_train, X_test, y_test):
        super().__init__(X_train, y_train, X_test, y_test)

    def train_test(self, model, N=50):

        # Check if model is XGBoost classifier
        if isinstance(model, xgb.XGBClassifier):
            xgb_param = model.get_xgb_params()
            extra = {'objective': 'binary:logistic'}
            xgb_param.update(extra)
            base = xgb.XGBClassifier(**xgb_param)
        elif isinstance(model, SVC):
            model.probability = True
            base = model
        else:
            base = model

        # Randomly select the order for N times 
        chains = [CC(base, 
                    order="random", 
                    random_state=i) for i in range(N)]
        
        # Train each chain
        for chain in tqdm(chains, desc='Chain: '):
            chain.fit(self.X_train, self.y_train)

        # Compute probability predictions for each chain and average them
        y_pred_chains = np.array([chain.predict_proba(self.X_test) for chain in chains])
        y_pred_ensemble = y_pred_chains.mean(axis=0)

        # Take average probability predictions and compute average predictions
        y_pred = (y_pred_ensemble > 0.5).astype('int')

        return pd.DataFrame(y_pred, columns=self.y_test.columns)
    
class MultiLabel_EnsembleRakel(MultiLabelClassification):

    def __init__(self, X_train, y_train, X_test, y_test):
        super().__init__(X_train, y_train, X_test, y_test)

    def train_test(self, model, type='distinct'):

        # Check if model is XGBoost classifier
        if isinstance(model, xgb.XGBClassifier):
            xgb_param = model.get_xgb_params()
            extra = {'objective': 'binary:logistic'}
            xgb_param.update(extra)
            base = xgb.XGBClassifier(**xgb_param)
        elif isinstance(model, SVC):
            model.probability = True
            base = model
        else:
            base = model

        if type=='distinct':
            clf = RakelD(base_classifier=base, labelset_size=3)
        elif type=='overlapping':
            clf = RakelO(base_classifier=base, labelset_size=3, model_count=2*5)
        clf.fit(self.X_train, self.y_train)

        y_pred = clf.predict(self.X_test).toarray()
        y_pred_prob = clf.predict_proba(self.X_test).toarray()

        # Compute ranking average precision
        rank_avg_prec = label_ranking_average_precision_score(self.y_test,y_pred_prob)
        print('Ranking average precision: ', rank_avg_prec)

        # Compute average precision score
        avg_prec = average_precision_score(self.y_test, y_pred)
        print('Average precision: ', avg_prec)

        return pd.DataFrame(y_pred, columns=self.y_test.columns)
    