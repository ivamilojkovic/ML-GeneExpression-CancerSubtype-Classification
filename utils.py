import matplotlib.pyplot as plt
from sklearn.model_selection import cross_validate
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score
from sklearn.metrics import recall_score, matthews_corrcoef
import torch
from deeplearn_models import *
import pandas as pd
from sklearn.linear_model import LassoCV

def log_transform(x):
    return np.log2(x + 0.1)

def check_dim(model, x_check=None):
    # Check model dimensions
    if isinstance(model, AutoEncoder):
        x_check = torch.randn(10, 1, 25150)
        if model(x_check).shape==torch.Size([10,1,25150]):
            print('Model is well defined!')
    elif isinstance(model, Classifier):
        if model(x_check).shape==torch.Size([1,5]):
            print('Model is well defined!')

def cmp_metrics(pred, y_test):

    metrics = {}

    # Weighted scores
    acc = accuracy_score(pred, y_test)
    precision = precision_score(pred, y_test, average='weighted')
    recall = recall_score(pred, y_test, average='weighted')
    f1 = f1_score(pred, y_test, average='weighted')
    print('Scores (weighted) on the test set:\n ')
    print('Accuracy: {}\nPrecision: {}\nRecall: {}\nF1 score: {}\n'.format(acc, precision, recall, f1))

    metrics['Accuracy weighted'] = acc
    metrics['Precision weighted'] = precision
    metrics['Recall weighted'] = recall
    metrics['F1 score weighted'] = f1

    # Unweighted scores
    precision = precision_score(pred, y_test, average='macro')
    recall = recall_score(pred, y_test, average='macro')
    f1 = f1_score(pred, y_test, average='macro')
    print('Scores (macro) on the test set:\n ')
    print('Precision: {}\nRecall: {}\nF1 score: {}'.format(precision, recall, f1))

    metrics['Precision unweighted'] = precision
    metrics['Recall unweighted'] = recall
    metrics['F1 score unweighted'] = f1

     # Scores for each class/label
    precision = precision_score(pred, y_test, average=None)
    recall = recall_score(pred, y_test, average=None)
    f1 = f1_score(pred, y_test, average=None)
    mcc = matthews_corrcoef(pred, y_test)
    print('Scores (per class) on the test set:\n ')
    print('Precision: {}\nRecall: {}\nF1 score: {}\nMCC: {}'.format(precision, recall, f1, mcc))

    metrics['Precision per class'] = precision
    metrics['Recall per class'] = recall
    metrics['F1 score per class'] = f1
    metrics['MCC'] = mcc

    return metrics
    
def cross_validation(model, _X, _y, _cv=5):

    _scoring = ['accuracy', 'precision', 'recall', 'f1']
    results = cross_validate(estimator=model,
                             X=_X,
                             y=_y,
                             cv=_cv,
                             scoring=_scoring,
                             return_train_score=True)

    return {"Training Accuracy scores": results['train_accuracy'],
            "Mean Training Accuracy": results['train_accuracy'].mean()*100,
            "Training Precision scores": results['train_precision'],
            "Mean Training Precision": results['train_precision'].mean(),
            "Training Recall scores": results['train_recall'],
            "Mean Training Recall": results['train_recall'].mean(),
            "Training F1 scores": results['train_f1'],
            "Mean Training F1 Score": results['train_f1'].mean(),
            "Validation Accuracy scores": results['test_accuracy'],
            "Mean Validation Accuracy": results['test_accuracy'].mean()*100,
            "Validation Precision scores": results['test_precision'],
            "Mean Validation Precision": results['test_precision'].mean(),
            "Validation Recall scores": results['test_recall'],
            "Mean Validation Recall": results['test_recall'].mean(),
            "Validation F1 scores": results['test_f1'],
            "Mean Validation F1 Score": results['test_f1'].mean()
            }

def plot_result(x_label, y_label, plot_title, train_data, val_data):

    plt.figure(figsize=(12,6))

    labels = ["1st Fold", "2nd Fold", "3rd Fold", "4th Fold", "5th Fold"]
    X_axis = np.arange(len(labels))

    ax = plt.gca()
    plt.ylim(0.40000, 1)
    plt.bar(X_axis-0.2, train_data, 0.4, color='blue', label='Training')
    plt.bar(X_axis+0.2, val_data, 0.4, color='red', label='Validation')
    plt.title(plot_title, fontsize=30)
    plt.xticks(X_axis, labels)
    plt.xlabel(x_label, fontsize=14)
    plt.ylabel(y_label, fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.show()

def forward_selection(X_train, X_val, y_train, y_val):
    """ 
    Custom forward selection!
        When score starts decreasing stops the algorithm 
        and returns all features up till then.

    """
    remaining_features = X_train.columns
    best_features = []
    past_val = 0.1
    while (len(remaining_features)>0):
        model = LassoCV(eps=0.001, cv=3)
        model.fit(X_train[best_features+[remaining_features[0]]], y_train)
        preds = model.predict(X_val)
        new_val = f1_score(preds, y_val, average='weighted')
        if(np.mean(new_val)<past_val):
            break
        else:
            best_features.append(remaining_features[0])
            remaining_features.pop(0)
            past_val = new_val

    return best_features