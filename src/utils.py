import matplotlib.pyplot as plt
import matplotlib
from sklearn.model_selection import cross_validate
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, roc_auc_score
from sklearn.metrics import recall_score, matthews_corrcoef
import torch
from deeplearn_models import *
import pandas as pd
from sklearn.linear_model import LassoCV
import seaborn as sns

plt.rcParams.update({'font.size': 12})
matplotlib.rcParams['xtick.labelsize'] = 12

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

    # Accuracy and Metthew Correlation Coeficients
    acc = accuracy_score(pred, y_test)
    mcc = matthews_corrcoef(y_test, pred)
    metrics['Accuracy weighted'] = acc
    metrics['MCC'] = mcc

    print(" ------ SCORES ------")
    print('Accuracy: {}\nMCCn: {}\n'.format(acc, mcc))

    # Weighted scores
    precision = precision_score(pred, y_test, average='weighted')
    recall = recall_score(pred, y_test, average='weighted')
    f1 = f1_score(pred, y_test, average='weighted')
    metrics['Precision weighted'] = precision
    metrics['Recall weighted'] = recall
    metrics['F1 score weighted'] = f1
    
    print('Scores (weighted) on the test set:\n')
    print('Precision: {}\nRecall: {}\nF1 score: {}\n'.format(precision, recall, f1))

    # Micro scores
    precision = precision_score(pred, y_test, average='micro')
    recall = recall_score(pred, y_test, average='micro')
    f1 = f1_score(pred, y_test, average='micro')
    metrics['Precision micro'] = precision
    metrics['Recall micro'] = recall
    metrics['F1 score micro'] = f1

    print('Scores (micro) on the test set:\n ')
    print('Precision: {}\nRecall: {}\nF1 score: {}\n'.format(precision, recall, f1))

    # Macro scores
    precision = precision_score(pred, y_test, average='macro')
    recall = recall_score(pred, y_test, average='macro')
    f1 = f1_score(pred, y_test, average='macro')
    metrics['Precision unweighted'] = precision
    metrics['Recall unweighted'] = recall
    metrics['F1 score unweighted'] = f1

    print('Scores (macro) on the test set:\n ')
    print('Precision: {}\nRecall: {}\nF1 score: {}'.format(precision, recall, f1))

    # Scores for each class/label
    precision = precision_score(y_test, pred, average=None)
    recall = recall_score(y_test, pred, average=None)
    f1 = f1_score(y_test, pred, average=None)

    metrics['Precision per class'] = precision
    metrics['Recall per class'] = recall
    metrics['F1 score per class'] = f1

    print('Scores (per class) on the test set:\n ')
    print('Precision: {}\nRecall: {}\nF1 score: {}'.format(precision, recall, f1))

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

def plot_before_after_counts(counts_before, counts_after):
    df_before = pd.DataFrame({'Class': list(counts_after.index),
                            'Counts': counts_before,
                            'Type': ['Before']*5
                            })
    df_after = pd.DataFrame({'Class': list(counts_after.index),
                            'Counts': counts_after,
                            'Type': ['After']*5
                            })

    df = pd.concat([df_before, df_after], ignore_index=True)

    ax = plt.figure()
    ax = sns.barplot(
            x = "Class",
            y = "Counts",
            hue = "Type",
            data = df
            )
    for g in ax.patches:
        ax.annotate(format(g.get_height(), '.0f'),
                    (g.get_x() + g.get_width() / 2., g.get_height()),
                    ha = 'center', va = 'center',
                    xytext = (0, 5),
                    textcoords = 'offset points',
                    fontsize=8)
        
    ax.tick_params(axis='x', rotation=30)
    ax.set_title('Class balance before and after')

# Let's try to visualize the clusters from different planes
def plot_pca(df_pca, labels_assigned, new_samples, dim=2):

    fig, ax = plt.subplots()

    label_colors = ['red', 'limegreen', 'darkgreen', 'royalblue', 'gold', 'pink', 'palevioletred']
    labels = np.unique(labels_assigned).tolist()

    i_color = 0
    if dim==2:
        for _, label in enumerate(labels):
            plt.scatter(df_pca.iloc[:,0][labels_assigned==label], 
                        df_pca.iloc[:,1][labels_assigned==label],
                        marker = '.', c = label_colors[i_color], s=30, alpha=.7)
            if sum(labels_assigned[new_samples]==label)!=0:
                plt.scatter(df_pca.iloc[new_samples,0][labels_assigned[new_samples]==label], 
                            df_pca.iloc[new_samples,1][labels_assigned[new_samples]==label],
                            marker = '.', c = label_colors[i_color+1], s=50, alpha=.4, edgecolors='black')
                i_color += 1
            i_color += 1

        ax.tick_params(axis='both', which='major', labelsize=8)
        plt.xlabel("PCA component 1", fontsize=10)
        plt.ylabel("PCA component 2", fontsize=10)
        plt.title('Principal Component Analysis')

    if dim==3:

        # Creating figure
        fig = plt.figure()
        ax = plt.axes(projection ="3d")

        label_colors = ['red', 'limegreen', 'darkgreen', 'royalblue', 'gold', 'pink', 'palevioletred']
        labels = np.unique(labels_assigned).tolist()

        i_color = 0
        for _, label in enumerate(labels):
            ax.scatter3D(df_pca.iloc[:,0][labels_assigned==label], 
                        df_pca.iloc[:,1][labels_assigned==label],
                        df_pca.iloc[:,2][labels_assigned==label],
                        marker = '.', c = label_colors[i_color], s=30, alpha=.7)
            if sum(labels_assigned[new_samples]==label)!=0:
                ax.scatter3D(df_pca.iloc[new_samples,0][labels_assigned[new_samples]==label], 
                            df_pca.iloc[new_samples,1][labels_assigned[new_samples]==label],
                            df_pca.iloc[new_samples,2][labels_assigned[new_samples]==label],
                            marker = '.', c = label_colors[i_color+1], s=50, alpha=.4, edgecolors='black')
                i_color += 1
            i_color += 1

        ax.tick_params(axis='both', which='major', labelsize=8)
        ax.set_xlabel("PCA component 1", fontsize=10)
        ax.set_ylabel("PCA component 2", fontsize=10)
        ax.set_zlabel("PCA component 3", fontsize=10)
        plt.title('Principal Component Analysis')

    labels = ['Basal', 'Her2', 'Her2 - new','LumA', 'LumB',  'Normal', 'Normal - new']
    plt.legend(labels, loc="upper left", ncol=len(labels), fontsize=6)