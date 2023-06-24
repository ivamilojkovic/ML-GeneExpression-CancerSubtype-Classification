import matplotlib.pyplot as plt
import matplotlib
from sklearn.model_selection import cross_validate
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, roc_auc_score
from sklearn.metrics import recall_score, matthews_corrcoef, hamming_loss
import torch
from deeplearn_models import *
import pandas as pd
from sklearn.linear_model import LassoCV
import seaborn as sns


plt.rcParams.update({'font.size': 12})
matplotlib.rcParams['xtick.labelsize'] = 12

def discard_negative_correlations(old_dataset):

    """ Set all negative correlation values to zero.
    """

    dataset = old_dataset.copy()

    dataset['LumA'][dataset['LumA'] < 0] = 0
    dataset['LumB'][dataset['LumB'] < 0] = 0
    dataset['Basal'][dataset['Basal'] < 0] = 0
    dataset['Her2'][dataset['Her2'] < 0] = 0
    dataset['Normal'][dataset['Normal'] < 0] = 0

    return dataset

def m_cut_strategy_class_assignment(orig_data, non_neg_values=False):

    """ Given the predicted probabilities from the classifier or correlations, 
        the M-cut strategy is performed to assign each sample one or more labels.
    """

    if non_neg_values:
        data = discard_negative_correlations(orig_data)
    else:
        data = orig_data

    assigned_labels = []

    for row in range(data.shape[0]):
        sample = data.iloc[row, :]
        sample_sorted = np.sort(sample)
        sort_idx = np.argmax(sample)

        prob_diff = sample_sorted[1:] - sample_sorted[:-1]
        max_diff_idx = np.argmax(prob_diff)
        
        m_cut_thresh = (sample_sorted[max_diff_idx] + sample_sorted[max_diff_idx+1])/2
        new_row = (sample>m_cut_thresh).astype(int)
        assigned_labels.append(new_row)

    return pd.DataFrame(assigned_labels, columns=data.columns)
        

def plot_class_distribution_comparison(data, y_corr_labels, y_corr_labels_neg):

    # Class distribution (counts) comparison
    ax = plt.figure()
    df_compare = pd.DataFrame({
        'Single label assigned by PAM50 maximum correlation': data['Subtype-from Parker centroids'].value_counts(),
        'Multiple labels M-cut strategy (non-negative correlations)': y_corr_labels.sum(axis=0),
        'Multiple labels M-cut & 5th percentile strategy (non-negative correlations)': y_corr_labels_neg.sum(axis=0)
        }, 
        index=['LumA', 'LumB',	'Basal', 'Her2', 'Normal'])
    ax = df_compare.plot(kind='bar', rot=30, title='Class distribution comparison')
    for g in ax.patches:
        ax.annotate(format(g.get_height(),),
                    (g.get_x() + g.get_width() / 2., g.get_height()),
                    ha = 'center', va = 'center',
                    xytext = (0, 5),
                    textcoords = 'offset points',
                    fontsize=6)


def count_number_of_labels(predictions):
    cnts = []
    for i in range(predictions.shape[1]+1):
        cnts.append(sum(predictions.sum(axis=1)==i))
    return cnts

def plot_bar_counts_of_label_predictions(predictions_orig, 
                                         predictions_pam50, 
                                         predictions_mcut,
                                         predictions_5perc,
                                         predictions_10perc,
                                         predictions_25perc):

    # Barplots for number of labels predicted
    ax = plt.figure()
    df_lr = pd.DataFrame({'Orignial (one-hot encoded)': count_number_of_labels(predictions_orig),
                    'PAM50 (one-hot encoded)': count_number_of_labels(predictions_pam50),
                    'M-cut': count_number_of_labels(predictions_mcut), 
                    'M-cut & 5th percentile filtering': count_number_of_labels(predictions_5perc),
                    'M-cut & 10th percentile filtering': count_number_of_labels(predictions_10perc),
                    'M-cut & 25th percentile filtering': count_number_of_labels(predictions_25perc)
                    }, 
                    index=['None', 'One', 'Two', 'Three', 'Four', 'All'])
    ax = df_lr.plot(kind='bar', rot=0, width = 0.9)
    for g in ax.patches:
        ax.annotate(format(g.get_height(), '.0f'),
                    (g.get_x() + g.get_width() / 2, g.get_height()),
                    ha = 'center', va = 'center',
                    xytext = (0, 5),
                    textcoords = 'offset points',
                    fontsize=4)
    plt.legend(loc='upper right')
    return ax

def plot_bar_counts_of_label(
    y_mcut,
    y_5perc,
    y_10perc,
    y_25perc):

    # Barplots for number of labels predicted
    ax = plt.figure()
    df_lr = pd.DataFrame({
                    'M-cut': count_number_of_labels(y_mcut), 
                    'M-cut & 5th percentile filtering': count_number_of_labels(y_5perc),
                    'M-cut & 10th percentile filtering': count_number_of_labels(y_10perc),
                    'M-cut & 25th percentile filtering': count_number_of_labels(y_25perc)
                    }, 
                    index=['None', 'One', 'Two', 'Three', 'Four', 'All'])
    ax = df_lr.plot(kind='bar', rot=0, width = 0.9)
    for g in ax.patches:
        ax.annotate(format(g.get_height(), '.0f'),
                    (g.get_x() + g.get_width() / 2, g.get_height()),
                    ha = 'center', va = 'center',
                    xytext = (0, 5),
                    textcoords = 'offset points',
                    fontsize=4)
    plt.legend(loc='upper right')
    return ax
        
def plot_stacked_bars(y_new, y_corr_labels, y_5perc_labels):

    # Generate some random data for the bars
    num_bars = 5
    num_colors = 5
    
    class_names = ['LumA', 'LumB',	'Basal', 'Her2', 'Normal']
    data = np.zeros(shape=(num_bars, num_colors))
    data2 = np.zeros(shape=(num_bars, num_colors))

    for i, label_y in enumerate(class_names):
        for j, label_corr in enumerate(class_names):
            val = y_corr_labels[label_corr][y_new==label_y].sum()
            data[i][j] = val

    for i, label_y in enumerate(class_names):
        for j, label_corr in enumerate(class_names):
            val = y_5perc_labels[label_corr][y_new==label_y].sum()
            data2[i][j] = val
            
    # Define the colors for the bars
    palette = sns.color_palette("deep", num_colors)

    # Create a figure and axis
    fig, ax = plt.subplots()

    # Set the positions of the bars on the x-axis
    x = np.arange(num_bars)

    # Plot the stacked bars
    # Plot the stacked bars
    width = 0.35

    # Plot the first set of stacked bars
    bottom = np.zeros(num_bars)
    for i in range(num_colors):
        ax.bar(x - 0.02 - width/2, data[:, i], width, bottom=bottom, 
               label=class_names[i], color=palette[i])
        bottom += data[:, i]

    # Plot the second set of stacked bars
    bottom = np.zeros(num_bars)
    for i in range(num_colors):
        ax.bar(x + 0.02 + width/2, data2[:, i], width, bottom=bottom, 
               label=class_names[i], color=palette[i])
        bottom += data2[:, i]

    # Customize the plot
    ax.set_xticks(x)
    ax.set_xticklabels(class_names)
    ax.set_ylabel('Labels correlation values - counts')
    ax.set_xlabel('PAM50 label')
    ax.legend(class_names)

    return ax

def plot_stacked_bars_primary_secondary_label_assigned(y_mcut_labels, y_pam50):

    # Generate some random data for the bars
    num_bars = 5
    num_colors = 2
    
    class_names = ['LumA', 'LumB',	'Basal', 'Her2', 'Normal']

    primary, secondary = {}, {}
    for i, label_pam50 in enumerate(y_pam50):

        if y_mcut_labels.iloc[i, :][label_pam50] == 1:
            if label_pam50 not in primary:
                primary[label_pam50] = 0
            primary[label_pam50] += 1

        if y_mcut_labels.iloc[i, :].sum() > 1: # there are other labels assigned 
            sec_labels = (y_mcut_labels.columns[y_mcut_labels.iloc[i, :] == 1]).tolist() 
            for sec_label in sec_labels:
                if sec_labels != label_pam50:
                    if sec_label not in secondary:
                        secondary[sec_label] = 0
                    secondary[sec_label] += 1

    # Concatenate Dictionary string values
    res = {key: [primary[key]] + [secondary[key]] for key in primary.keys()}
    result = res.values()
    class_names = list(res.keys())
 
    # Convert object to a list
    data = list(result)
    
    # Convert list to an array
    data = np.array(data).transpose()
            
    # Define the colors for the bars
    palette = sns.color_palette("deep", num_colors)

    # Create a figure and axis
    fig, ax = plt.subplots()

    # Set the positions of the bars on the x-axis
    bars = np.arange(num_bars)

    # Plot the stacked bars
    width = 0.5

    # Plot the first set of stacked bars
    bottom = np.zeros(num_bars)
    for i in range(num_colors):
        ax.bar(bars, data[i, :], width, bottom=bottom, 
               label=class_names[i], color=palette[i])
        bottom += data[i, :]

    # Customize the plot
    ax.set_xticks(bars)
    ax.set_xticklabels(class_names)
    ax.set_ylabel('Label counts')
    ax.set_xlabel('labels')
    ax.legend(['Primary label', 'Secondary label'])

    return ax


def create_mcut_nth_percentile_labels(m_cut_labels, correlations, y, N: int = 5):

    """ If a label-specific correlation is below the 5th percentile,
        set the m-cut_label to 0 if it was 1 previously. This scenario 
        can create samples that do not have any label assigned, compared 
        to m-cut strategy, where at least one label need to be assigned.
    """

    # Create a copy of m_cut_labels that's going to be modified
    m_cut_labels_2 = m_cut_labels.copy(deep=True)

    for i, label in enumerate(correlations.columns):

        # Find indices where label is located in y (PAM50 label with max correlation)
        pam50_label_idx = y == label

        # Compute the 5th percentile
        label_thresh = np.percentile(
            correlations[label][pam50_label_idx], N)
        
        # Set 0 where the correlation is below the threshold
        lower_than_thresh_idx = correlations.loc[pam50_label_idx, label] < label_thresh

        # Use loc to modify values in m_cut_labels_2
        indices = m_cut_labels_2.loc[pam50_label_idx, label].loc[lower_than_thresh_idx].index
        m_cut_labels_2.loc[indices, label] = 0

    return m_cut_labels_2

def relaxed_accuracy(y_true, y_pred):

    num_instances = len(y_true)
    correct_instances = 0

    for i in range(num_instances):
        if y_true.shape[1] == 5:
            if any(y_pred.iloc[i][y_true.iloc[i,:]==1]==1):
                correct_instances += 1
        else:
            if y_pred.iloc[i, y_true[i]]==1:
                correct_instances += 1

    return correct_instances / num_instances


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



def print_all_scores(y_test, predictions):

    # Total scores
    print('\nTest accuracy: {}'.format(accuracy_score(y_test, predictions)))
    print('Test Hamming loss: {}\n'.format(hamming_loss(y_test, predictions)))

    print('Test precision (weighted): {}'.\
        format(precision_score(y_test, predictions, 
                                average='weighted', zero_division=1)))
    print('Test recall (weighted): {}'.\
        format(recall_score(y_test, predictions, 
                            average='weighted', zero_division=1)))
    print('Test f1 score (weighted): {}\n'.\
        format(f1_score(y_test, predictions, 
                        average='weighted', zero_division=1)))

    print('Test precision (macro): {}'.\
        format(precision_score(y_test, predictions, 
                                average='macro', zero_division=1)))
    print('Test recall (macro): {}'.\
        format(recall_score(y_test, predictions, 
                            average='macro', zero_division=1)))
    print('Test f1 score (macro): {}\n'.\
        format(f1_score(y_test, predictions, 
                        average='macro', zero_division=1)))

    print('Test precision (micro): {}'.\
        format(precision_score(y_test, predictions, 
                                average='micro', zero_division=1)))
    print('Test recall (micro): {}'.\
        format(recall_score(y_test, predictions, 
                            average='micro', zero_division=1)))
    print('Test f1 score (micro): {}\n'.\
        format(f1_score(y_test, predictions, 
                        average='micro', zero_division=1)))
