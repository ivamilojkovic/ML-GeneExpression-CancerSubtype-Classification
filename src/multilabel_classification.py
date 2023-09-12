from skmultilearn.problem_transform import LabelPowerset, BinaryRelevance, ClassifierChain
from skmultilearn.adapt import MLkNN, MLTSVM, MLARAM
from skmultilearn.ensemble import RakelD, RakelO
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.multioutput import ClassifierChain as CC, MultiOutputClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, \
    label_ranking_loss, make_scorer
from tqdm import tqdm
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
import xgboost as xgb
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, KFold
import itertools
from skmultilearn.ensemble import LabelSpacePartitioningClassifier
from config_model import MULTILABEL_MODEL_PARAMS
import pickle, time, datetime
from sklearn.preprocessing import normalize

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
            # LabelRankLoss = make_scorer(label_ranking_loss, greater_is_better=False, needs_proba=True)
            grid_search = GridSearchCV(
                estimator=clf, 
                param_grid=additional_params,
                scoring= 'f1_weighted',  # f1_weighted, accuracy...
                verbose=2, 
                cv=KFold(
                    n_splits=5, 
                    shuffle=True, 
                    random_state=123), 
                return_train_score=True, 
                refit=True) 

            # Fit the grid search to the data
            grid_search.fit(self.X_train.values, self.y_train.values)
            # grid_search.fit(self.X_train, self.y_train)

            # Get the best model
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_

            # Get cross-validation scores (averagre and std value of train/validation score)
            validation_scores = grid_search.cv_results_['mean_test_score']
            train_scores = grid_search.cv_results_['mean_train_score']

            # Calculate the average score and standard deviation
            avg_val_score, std_val_score = np.nanmean(validation_scores), np.nanstd(validation_scores)
            avg_train_score, std_train_score = np.nanmean(train_scores), np.nanstd(train_scores)

            print("Average training score:", avg_train_score)
            print("Training score standard deviation:", std_train_score)

            print("Average validation score:", avg_val_score)
            print("Validation score standard deviation:", std_val_score)

            scores = {
                'Train': {
                    'average': avg_train_score,
                    'std': std_train_score,
                },
                'Validation': {
                    'average': avg_val_score,
                    'std': std_val_score,
                }
            }

            now = datetime.datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
            with open('ml_best_model_lr' + now + '.pkl', 'wb') as file:
                pickle.dump(best_model, file)
            
            with open('best_model_cv_scores' + now + '.pkl', 'wb') as file:
                pickle.dump(scores, file)

            pred = best_model.predict(self.X_test).toarray()
            pred_prob = best_model.predict_proba(self.X_test).toarray()

            # Put preds and prob_preds into suitable shape
            preds = pd.DataFrame(pred, columns=self.labels)
            prob_preds = pd.DataFrame(pred_prob, columns=self.labels, index=self.X_test.index)

        else:
            for label in self.labels:

                clf.fit(self.X_train, self.y_train[label].values)
                pred = clf.predict(self.X_test).toarray()
                pred_prob = clf.predict_proba(self.X_test).toarray()

                preds.append(pred)
                prob_preds.append(pred_prob)

            # Put preds and prob_preds into suitable shape
            preds = pd.DataFrame(np.transpose(preds)[0], columns=self.labels)
            prob_preds = pd.DataFrame(np.transpose(prob_preds)[0], columns=self.labels, index=self.X_test.index)

        return preds, prob_preds, best_model, best_params, scores

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

        # Define the  classifier
        chain = Pipeline([
            ('classifier', chain)
        ])

        if optimize:
            # Define the labels
            labels = self.y_train.columns

            # Generate all possible permutations of label orders
            all_orders = list(itertools.permutations([0, 1, 2, 3, 4]))
            all_orders = [list(order) for order in all_orders]

            # Define the parameter grid for hyperparameter tuning
            param_grid = {
                'classifier__order': all_orders, # Order in which labels are chained
                'classifier__classifier': [base]
                }
            
            # Create the grid search object
            grid_search = GridSearchCV(
                chain, 
                param_grid=param_grid, 
                scoring='f1_weighted', 
                verbose=2,
                cv=KFold(
                    n_splits=5, 
                    shuffle=True, 
                    random_state=123), 
                return_train_score=True, 
                refit=True)

            
            # Fit the grid search to the data
            grid_search.fit(self.X_train.values, self.y_train.values)

            # if isinstance(model, RandomForestClassifier):
            #     grid_search.fit(self.X_train.values, self.y_train.values)
            # else:
            #     grid_search.fit(self.X_train, self.y_train)

            # Get best model
            chain = grid_search.best_estimator_

        if optimize_model:
            if isinstance(model, LogisticRegression):
                model_name = 'Logistic Regression'
            elif isinstance(model, RandomForestClassifier):
                model_name = 'Random Forest'
            elif isinstance(model, xgb.XGBClassifier):
                model_name = 'XGBoost'
            elif isinstance(model, SVC):
                model_name = 'SVC'

            additional_params = {
                'classifier__order': [[0, 1, 2, 3, 4]], # Order in which labels are chained [[0, 1, 2, 3, 4]]
                'classifier__classifier': [base]
                }
            for key, value in MULTILABEL_MODEL_PARAMS[model_name].items():
                new_key = 'classifier__classifier__' + key
                additional_params[new_key] = value
        
            # Create the grid search object
            grid_search = GridSearchCV(
                chain, 
                param_grid=additional_params, 
                scoring='f1_weighted', 
                verbose=2,
                cv=KFold(
                    n_splits=5, 
                    shuffle=True, 
                    random_state=123), 
                return_train_score=True, 
                refit=True)

            # Fit the grid search to the data
            grid_search.fit(self.X_train, self.y_train)

            # Get best model
            chain = grid_search.best_estimator_
            best_params = grid_search.best_params_

            # Get cross-validation scores (averagre and std value of train/validation score)
            validation_scores = grid_search.cv_results_['mean_test_score']
            train_scores = grid_search.cv_results_['mean_train_score']

            # Calculate the average score and standard deviation
            avg_val_score, std_val_score = np.nanmean(validation_scores), np.nanstd(validation_scores)
            avg_train_score, std_train_score = np.nanmean(train_scores), np.nanstd(train_scores)

            print("Average training score:", avg_train_score)
            print("Training score standard deviation:", std_train_score)

            print("Average validation score:", avg_val_score)
            print("Validation score standard deviation:", std_val_score)

            scores = {
                'Train': {
                    'average': avg_train_score,
                    'std': std_train_score,
                },
                'Validation': {
                    'average': avg_val_score,
                    'std': std_val_score,
                }
            }

            now = datetime.datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
            with open('ml_best_model_lr' + now + '.pkl', 'wb') as file:
                pickle.dump(chain, file)
            
            with open('best_model_cv_scores' + now + '.pkl', 'wb') as file:
                pickle.dump(scores, file)
        else:
            chain.fit(self.X_train, self.y_train)

            
        # Compute probability predictions and predictions
        y_pred = chain.predict(self.X_test).toarray()
        y_pred = pd.DataFrame(y_pred, columns=self.y_test.columns, dtype='int')
        y_pred_prob = chain.predict_proba(self.X_test).toarray()
        y_pred_prob = pd.DataFrame(y_pred_prob, columns=self.y_test.columns)
        
        return y_pred, y_pred_prob, chain, best_params, scores

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
            grid_search = GridSearchCV(
                estimator=clf, 
                param_grid=additional_params, 
                scoring='f1_weighted', 
                verbose=2,
                cv=KFold(
                    n_splits=5, 
                    shuffle=True, 
                    random_state=123), 
                return_train_score=True, 
                refit=True)
            
            # Fit the grid search to the data
            grid_search.fit(self.X_train.values, self.y_train.values)
            # if isinstance(model, RandomForestClassifier):
            #     grid_search.fit(self.X_train.values, self.y_train.values)
            # else:
            #     grid_search.fit(self.X_train, self.y_train)

            # Get the best model
            clf = grid_search.best_estimator_
            best_params = grid_search.best_params_

            # Get cross-validation scores (averagre and std value of train/validation score)
            validation_scores = grid_search.cv_results_['mean_test_score']
            train_scores = grid_search.cv_results_['mean_train_score']

            # Calculate the average score and standard deviation
            avg_val_score, std_val_score = np.nanmean(validation_scores), np.nanstd(validation_scores)
            avg_train_score, std_train_score = np.nanmean(train_scores), np.nanstd(train_scores)

            print("Average training score:", avg_train_score)
            print("Training score standard deviation:", std_train_score)

            print("Average validation score:", avg_val_score)
            print("Validation score standard deviation:", std_val_score)

            scores = {
                'Train': {
                    'average': avg_train_score,
                    'std': std_train_score,
                },
                'Validation': {
                    'average': avg_val_score,
                    'std': std_val_score,
                }
            }

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

        # Compute probability predictions and predictions
        y_pred = clf.predict(self.X_test).toarray()
        y_pred = pd.DataFrame(y_pred, columns=self.y_test.columns, dtype='int')
        y_pred_prob = clf.predict_proba(self.X_test).toarray()
        y_pred_prob = pd.DataFrame(y_pred_prob, columns=self.y_test.columns)

        return y_pred, y_pred_prob, clf, best_params, scores


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
                verbose=2,
                cv=KFold(
                    n_splits=5, 
                    shuffle=True, 
                    random_state=123), 
                return_train_score=True, 
                refit=True)
            
            gs.fit(x_train, y_train)

            classifier = gs.best_estimator_
            best_params = gs.best_params_

            # Get cross-validation scores (averagre and std value of train/validation score)
            validation_scores = gs.cv_results_['mean_test_score']
            train_scores = gs.cv_results_['mean_train_score']

            # Calculate the average score and standard deviation
            avg_val_score, std_val_score = np.nanmean(validation_scores), np.nanstd(validation_scores)
            avg_train_score, std_train_score = np.nanmean(train_scores), np.nanstd(train_scores)

            print("Average training score:", avg_train_score)
            print("Training score standard deviation:", std_train_score)

            print("Average validation score:", avg_val_score)
            print("Validation score standard deviation:", std_val_score)

            scores = {
                'Train': {
                    'average': avg_train_score,
                    'std': std_train_score,
                },
                'Validation': {
                    'average': avg_val_score,
                    'std': std_val_score,
                }
            }
        else:
            classifier.fit(x_train, y_train)

        # Compute predictions and their probabilities
        y_pred = classifier.predict(self.X_test.values)
        y_pred_prob = classifier.predict_proba(self.X_test.values)

        if type != 'MLARAM':
            y_pred_prob = y_pred_prob.todense()
            y_pred = y_pred.todense()

        # Compute probability predictions and predictions
        y_pred = pd.DataFrame(y_pred, columns=self.y_test.columns, 
                                index=self.y_test.index, dtype='int')
        y_pred_prob = pd.DataFrame(y_pred_prob, columns=self.y_test.columns,
                                    index=self.y_test.index, dtype='float64')

        return y_pred, y_pred_prob, classifier, best_params, scores
    

class MultiLabel_EnsembleChains(MultiLabelClassification):

    def __init__(self, X_train, y_train, X_test, y_test):
        super().__init__(X_train, y_train, X_test, y_test)

    def train_test(self, model, N: int=50, optimize: bool=False):

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
        # chains = [ClassifierChain(classifier=base) for _ in range(N)]

        classifier = MultiOutputClassifier(chains)

        # Define the pipeline with a classifier
        pipeline = Pipeline([
            ('classifier', classifier)
        ])

        if optimize:
            if isinstance(model, LogisticRegression):
                model_name = 'Logistic Regression'
            elif isinstance(model, RandomForestClassifier):
                model_name = 'Random Forest'
            elif isinstance(model, xgb.XGBClassifier):
                model_name = 'XGBoost'
            elif isinstance(model, SVC):
                model_name = 'SVC'

            additional_params = {'classifier__estimator': [base]}
            for key, value in MULTILABEL_MODEL_PARAMS[model_name].items():
                new_key = 'classifier__estimator__' + key
                additional_params[new_key] = value

            # Create the grid search object
            grid_search = GridSearchCV(
                pipeline, 
                param_grid=additional_params, 
                scoring='f1_weighted', 
                verbose=2,
                cv=KFold(
                    n_splits=5, 
                    shuffle=True, 
                    random_state=123), 
                return_train_score=True, 
                refit=True)

            # Fit the grid search to the data
            grid_search.fit(self.X_train.values, self.y_train.values)
            # if isinstance(model, RandomForestClassifier):
            #     grid_search.fit(self.X_train.values, self.y_train.values)
            # else:
            #     grid_search.fit(self.X_train, self.y_train)

            # Get best model
            classifier = grid_search.best_estimator_
            best_params = grid_search.best_params_

            # Get cross-validation scores (averagre and std value of train/validation score)
            validation_scores = grid_search.cv_results_['mean_test_score']
            train_scores = grid_search.cv_results_['mean_train_score']

            # Calculate the average score and standard deviation
            avg_val_score, std_val_score = np.nanmean(validation_scores), np.nanstd(validation_scores)
            avg_train_score, std_train_score = np.nanmean(train_scores), np.nanstd(train_scores)

            print("Average training score:", avg_train_score)
            print("Training score standard deviation:", std_train_score)

            print("Average validation score:", avg_val_score)
            print("Validation score standard deviation:", std_val_score)

            scores = {
                'Train': {
                    'average': avg_train_score,
                    'std': std_train_score,
                },
                'Validation': {
                    'average': avg_val_score,
                    'std': std_val_score,
                }
            }

            now = datetime.datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
            with open('bestmodel_ECC_' + now + '.pkl', 'wb') as file:
                pickle.dump(classifier, file)
            with open('bestparam_ECC_' + now + '.pkl', 'wb') as file:
                pickle.dump(grid_search.best_params_, file)

        else:
            # Train each chain
            # for chain in tqdm(chains, desc='Chain: '):
            #     chain.fit(self.X_train, self.y_train)
            classifier.fit(self.X_train, self.y_train)

        # Compute probability predictions and predictions
        y_pred = classifier.predict(self.X_test)
        y_pred_prob = classifier.predict_proba(self.X_test)
        y_pred_prob = normalize(np.array([y_pred_prob[i][:, 1] for i in range(5)]).transpose(), axis=1, norm='l1')
        # y_pred_prob = np.array([arr[:, 1] for arr in classifier.predict_proba(self.X_test)]).T

        # Compute probability predictions and predictions
        y_pred = pd.DataFrame(y_pred, columns=self.y_test.columns, dtype='int')
        y_pred_prob = pd.DataFrame(y_pred_prob, columns=self.y_test.columns)

        return y_pred, y_pred_prob, classifier, best_params, scores
    
class MultiLabel_EnsembleRakel(MultiLabelClassification):

    def __init__(self, X_train, y_train, X_test, y_test):
        super().__init__(X_train, y_train, X_test, y_test)

    def train_test(self, model, optimize: bool=False, type: str='distinct'):

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
            classifier = RakelD(base_classifier=base, labelset_size=3)
        elif type=='overlapping':
            raise NotImplementedError
            clf = RakelO(base_classifier=base, labelset_size=3, model_count=2*5)

        if optimize:
            if isinstance(model, LogisticRegression):
                model_name = 'Logistic Regression'
            elif isinstance(model, RandomForestClassifier):
                model_name = 'Random Forest'
            elif isinstance(model, xgb.XGBClassifier):
                model_name = 'XGBoost'
            elif isinstance(model, SVC):
                model_name = 'SVC'

            classifier = Pipeline([('classifier', classifier)])

            additional_params = {
                'classifier__labelset_size': [2, 3, 4],
                'classifier__base_classifier': [model]
            }

            for key, value in MULTILABEL_MODEL_PARAMS[model_name].items():
                new_key = 'classifier__base_classifier__' + key
                additional_params[new_key] = value

            # Create the grid search object
            grid_search = GridSearchCV(
                classifier, 
                param_grid=additional_params, 
                scoring='f1_weighted', 
                verbose=2,
                cv=KFold(
                    n_splits=5, 
                    shuffle=True, 
                    random_state=123), 
                return_train_score=True, 
                refit=True)

            # Fit the grid search to the data
            grid_search.fit(self.X_train.values, self.y_train.values)
            # if isinstance(model, RandomForestClassifier):
            #     grid_search.fit(self.X_train.values, self.y_train.values)
            # else:
            #     grid_search.fit(self.X_train, self.y_train)

            # Get best model
            classifier = grid_search.best_estimator_
            best_params = grid_search.best_params_

            # Get cross-validation scores (averagre and std value of train/validation score)
            validation_scores = grid_search.cv_results_['mean_test_score']
            train_scores = grid_search.cv_results_['mean_train_score']

            # Calculate the average score and standard deviation
            avg_val_score, std_val_score = np.nanmean(validation_scores), np.nanstd(validation_scores)
            avg_train_score, std_train_score = np.nanmean(train_scores), np.nanstd(train_scores)

            print("Average training score:", avg_train_score)
            print("Training score standard deviation:", std_train_score)

            print("Average validation score:", avg_val_score)
            print("Validation score standard deviation:", std_val_score)

            scores = {
                'Train': {
                    'average': avg_train_score,
                    'std': std_train_score,
                },
                'Validation': {
                    'average': avg_val_score,
                    'std': std_val_score,
                }
            }

            now = datetime.datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
            with open('bestmodel_rakel_' + now + '.pkl', 'wb') as file:
                pickle.dump(classifier, file)
            with open('bestparam_rakel_' + now + '.pkl', 'wb') as file:
                pickle.dump(grid_search.best_params_, file)

        else:
            classifier.fit(self.X_train, self.y_train)

        # Compute probability predictions and predictions
        y_pred = classifier.predict(self.X_test).toarray()
        y_pred = pd.DataFrame(y_pred, columns=self.y_test.columns, dtype='int')
        y_pred_prob = classifier.predict_proba(self.X_test).toarray()
        y_pred_prob = pd.DataFrame(y_pred_prob, columns=self.y_test.columns)

        return y_pred, y_pred_prob, classifier, best_params, scores