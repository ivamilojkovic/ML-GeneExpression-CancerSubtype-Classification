from multilabel_classification import *
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, matthews_corrcoef
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import LabelEncoder
import datetime
import pickle
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib
from utils import m_cut_strategy_class_assignment

# Set parameters
plt.style.use('ggplot')
matplotlib.rcParams['font.size'] = 8
plt.rcParams["figure.figsize"] = (8, 6)

exp_name = 'run_16-05-2023_21:05:39'

########################### Load the data ###########################

# with open(os.path.join('artefacts', exp_name + '.pkl'), 'rb') as file:
#     data = pickle.load(file)
#     X_train_scaled, y_train = data[0], data[1]
#     X_test_scaled, y_test = data[2], data[3]

with open('dataset_multilabel.pkl', 'rb') as file:
    data = pickle.load(file)
    X = data.drop(columns=['expert_PAM50_subtype', 'tcga_id',
                           'Subtype-from Parker centroids',	'MaxCorr',
                            'Basal', 'Her2', 'LumA', 'LumB', 'Normal'], inplace=False)
    y_old = data.expert_PAM50_subtype
    y_new = data['Subtype-from Parker centroids']

    # M-cut strategy to assign labels on whole dataset
    y_corr = data[['Basal', 'Her2', 'LumA', 'LumB',	'Normal']]
    y_corr_labels = m_cut_strategy_class_assignment(y_corr, non_neg_values=True)
    #y_corr_labels_neg = m_cut_strategy_class_assignment(y_corr, non_neg_values=True)
    

    # Class distribution (counts) comparison
    ax = plt.figure()
    df_compare = pd.DataFrame({'Single label': data['Subtype-from Parker centroids'].value_counts(),
                    'Multiple labels M-cut strategy (non-negative correlations)': y_corr_labels.sum(axis=0),
                    #'Multiple labels M-cut strategy (negative correlations)': y_corr_labels_neg.sum(axis=0)
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
    
    X_train, X_test, y_train, y_test, \
        y_train_corr_labels, y_test_corr_labels = train_test_split(X, y_new, y_corr_labels, 
                                                   test_size=0.3, random_state=1)

    # Data standardization | normalization
    X_train = X_train.divide(X_train.sum(axis=1), axis=0) * 1e6
    X_test = X_test.divide(X_test.sum(axis=1), axis=0) * 1e6
    scaler = FunctionTransformer(log_transform)
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)

    # One hot encoding of original labels
    LB = LabelEncoder() 
    y_train = pd.Series(LB.fit_transform(y_train), index=y_train.index)
    y_test = LB.transform(y_test)
    # y_test = pd.get_dummies(y_test)
    # y_train = pd.get_dummies(y_train)

    # Feature selection
    best_feat_model = SelectKBest(score_func=f_classif, k=500) # k needs to be defined
    best_feat_model.fit(X_train_scaled, y_train)
    df_scores = pd.DataFrame(best_feat_model.scores_)
    df_feats = pd.DataFrame(X.columns)

    featureScores = pd.concat([df_feats, df_scores],axis=1)
    featureScores.columns = ['Feature', 'Score'] 

    selected_feat = featureScores.sort_values(by='Score')[-500:]['Feature']

    X_train_scaled_selected = X_train_scaled[list(selected_feat)]
    X_test_scaled_selected = X_test_scaled[list(selected_feat)]

# Load model
with open(os.path.join('models', 'bestmodel_run_08-05-2023_10:32:03.pkl'), 'rb') as file:
    model = pickle.load(file)

# Train and test
ml_approach = MultiLabel_BinaryRelevance(X_train=X_train_scaled_selected,
                             X_test=X_test_scaled_selected,
                             y_train=y_train_corr_labels, 
                             y_test=y_test_corr_labels)

predictions = ml_approach.train_test(model)
print(predictions)

# If M-cut strategy
y_test = y_test_corr_labels

# Total scores
print('\nTest accuracy: {}'.format(accuracy_score(y_test, predictions)))

print('Test precision (weighted): {}'.format(precision_score(y_test, predictions, average='weighted', zero_division=1)))
print('Test recall (weighted): {}'.format(recall_score(y_test, predictions, average='weighted', zero_division=1)))
print('Test f1 score (weighted): {}\n'.format(f1_score(y_test, predictions, average='weighted', zero_division=1)))

print('Test precision (macro): {}'.format(precision_score(y_test, predictions, average='macro', zero_division=1)))
print('Test recall (macro): {}'.format(recall_score(y_test, predictions, average='macro', zero_division=1)))
print('Test f1 score (macro): {}\n'.format(f1_score(y_test, predictions, average='macro', zero_division=1)))

print('Test precision (micro): {}'.format(precision_score(y_test, predictions, average='micro', zero_division=1)))
print('Test recall (micro): {}'.format(recall_score(y_test, predictions, average='micro', zero_division=1)))
print('Test f1 score (micro): {}\n'.format(f1_score(y_test, predictions, average='micro', zero_division=1)))






