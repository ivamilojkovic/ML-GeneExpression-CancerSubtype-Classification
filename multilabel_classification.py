from skmultilearn.problem_transform import LabelPowerset, BinaryRelevance
from skmultilearn.adapt import MLkNN
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.multioutput import ClassifierChain
from sklearn.preprocessing import LabelEncoder, LabelBinarizer, FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import datetime
import pickle as pkl
from sklearn.model_selection import train_test_split
from utils import log_transform
import pandas as pd
import numpy as np

class MultiLabelClassification():
    def __init__(self, X_train, y_train, X_test, y_test):
        
        self.X_train = X_train
        self.X_test = X_test

        self.labels = y_train.unique()

        self.y_train = pd.get_dummies(y_train)
        self.y_test = pd.get_dummies(y_test)

class MultiLabel_OnevsRest(MultiLabelClassification):

    def __init__(self, X_train, y_train, X_test, y_test):
        super().__init__(X_train, y_train, X_test, y_test)

    def train_test(self):

        preds = {}
        
        class_pipe = Pipeline([('clf', OneVsRestClassifier(LogisticRegression(solver='sag', n_jobs=-1)))])
        for label in self.labels:
            print('Progress for predicting label {}...'.format(label))

            class_pipe.fit(self.X_train, self.y_train[label])
            pred = class_pipe.predict(self.X_test)

            preds[label] = pred
            
            print('Test accuracy is {}\n'.format(accuracy_score(self.y_test[label], pred)))

        return pd.DataFrame(preds)
    
class MultiLabel_Chains(MultiLabelClassification):

    def __init__(self, X_train, y_train, X_test, y_test):
        super().__init__(X_train, y_train, X_test, y_test)

    def train_test(self):

        classifier = GaussianNB()
        chain = ClassifierChain(classifier) 

        chain.fit(self.X_train, self.y_train)
        pred = chain.predict(self.X_test)

        print('Test accuracy is {}\n'.format(accuracy_score(self.y_test, pred)))

        return pred


class MultiLabel_Adapted(MultiLabelClassification):

    def __init__(self, X_train, y_train, X_test, y_test):
        super().__init__(X_train, y_train, X_test, y_test)

    def train_test(self):

        classifier = MLkNN(k=20)
        classifier.fit(X=self.X_train.values, y=self.y_train.values)
        pred = classifier.predict(self.X_test.values)

        print('Test accuracy is {}\n'.format(accuracy_score(self.y_test.values, pred)))

        return pred


if __name__ == '__main__':

    EXP_PATH = 'experiments'

    # Set the parameters
    SOLVE_IMB = False # Solve class imbalance problem
    SMOTE = True
    CROSS_VAL = False
    TEST_SIZE = 0.3
    RANDOM_STATE = 4
    OPTIM = False

    N_folds = 10
    N_feats = 500

    # Feature selection can be: Univariate, Recursive...
    FEAT_SELECT = 'Filter' 

    # Get the current date and time
    now = datetime.datetime.now()
    experiment_name =  'run_' + now.strftime("%d-%m-%Y_%H:%M:%S")
    experiment_params = {
        'solve_ibm': SOLVE_IMB,
        'use_smote': SMOTE,
        'cross_validation': CROSS_VAL,
        'multi_label_classification': True,
        'n_folds': N_folds,
        'n_features_to_select': N_feats,
        'test_size': TEST_SIZE,
        'feature_selection_method': FEAT_SELECT,
        'optimized': OPTIM
    }


    # Load the dataset
    DATASET_PATH = "tcga_brca_raw_19036_1053samples.pkl"

    with open(DATASET_PATH, 'rb') as file:
        dataset = pkl.load(file) 

    X = dataset.drop(columns=['expert_PAM50_subtype', 'tcga_id', 'sample_id', 'cancer_type'], inplace=False)
    y = dataset.expert_PAM50_subtype

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=1, 
                                                        shuffle=True, stratify=y) 
    
    # Data standardization | normalization
    scaler = FunctionTransformer(log_transform)
    X_train_scaled = scaler.fit_transform(X_train)
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_test_scaled = scaler.transform(X_test)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)

    # Train and test
    mlc = MultiLabel_Adapted(X_train=X_train_scaled,
                               X_test=X_test_scaled,
                               y_train=y_train, 
                               y_test=y_test)
    
    predictions = mlc.train_test()
    print(predictions)









