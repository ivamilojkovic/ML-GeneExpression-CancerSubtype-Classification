from multilabel_classification import *
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import FunctionTransformer
import pickle
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib
from utils import *
import mlflow

# Set parameters
plt.style.use('ggplot')
matplotlib.rcParams['font.size'] = 8
plt.rcParams["figure.figsize"] = (8, 6)

# EXPERIMENT_NAME = "multilabel_0"
# EXPERIMENT_ID = mlflow.create_experiment(EXPERIMENT_NAME)
# exp_name = 'run_16-05-2023_21:05:39'

REMOVE_MISSCLASS = False
LABEL_UNLANELED = False


########################### Load the data ###########################

with open('data/dataset_multilabel.pkl', 'rb') as file:
    data = pickle.load(file)
    X = data.drop(columns=['expert_PAM50_subtype', 'tcga_id',
                           'Subtype-from Parker centroids',	'MaxCorr',
                            'Basal', 'Her2', 'LumA', 'LumB', 'Normal'], inplace=False)
    y_orig = data.expert_PAM50_subtype
    y_pam50 = data['Subtype-from Parker centroids']

    # Take labels on whole dataset for PAM50
    y_corr = data[['Basal', 'Her2', 'LumA', 'LumB',	'Normal']]
    y_corr_non_neg = discard_negative_correlations(y_corr)

    # M-cut strategy to assign labels on whole dataset
    y_mcut_labels = m_cut_strategy_class_assignment(y_corr, non_neg_values=True)
    y_mcut_labels_neg = m_cut_strategy_class_assignment(y_corr, non_neg_values=False)
    
    # Compare class distributions for original and two cases with m-cut
    plot_class_distribution_comparison(data, y_mcut_labels, y_mcut_labels_neg)

    # Compute labels from two strategies (M-cut and 5th percentile)
    y_mcut_5perc_labels = create_mcut_nth_percentile_labels(
        m_cut_labels=y_mcut_labels,
        correlations=y_corr_non_neg,
        y=y_pam50,
        N=5
    )

    y_mcut_10perc_labels = create_mcut_nth_percentile_labels(
        m_cut_labels=y_mcut_labels,
        correlations=y_corr_non_neg,
        y=y_pam50,
        N=10
    )

    y_mcut_25perc_labels = create_mcut_nth_percentile_labels(
        m_cut_labels=y_mcut_labels,
        correlations=y_corr_non_neg,
        y=y_pam50,
        N=25
    )

    # TODO: Save all label variations

    # Check the number of misclassified 
    misclass_samples_idx = y_orig[y_pam50!=y_orig].index
    print('Number of missclassified samples compared to original labels: ', 
          sum(y_pam50!=y_orig))
    
    # Number of labels assigned
    ax_labels = plot_bar_counts_of_label(y_mcut_labels, y_mcut_5perc_labels,
                                         y_mcut_10perc_labels, y_mcut_25perc_labels)
    ax_labels.set_title('Number of labels assigned (whole dataset)')

    # TODO: Check which subtypes appeared 0, 1, 2 or 3 

    # TODO: Plot primary and secondary labels assigned
    plot_stacked_bars_primary_secondary_label_assigned(y_mcut_labels, y_pam50)
    
    # Remove non-overlapping samples
    if REMOVE_MISSCLASS:
        same_idx = y_pam50==y_orig
        y_orig = y_orig[same_idx]
        y_pam50 = y_pam50[same_idx]
        y_mcut_labels = y_mcut_labels[same_idx]
        y_mcut_5perc_labels = y_mcut_5perc_labels[same_idx]
        y_mcut_10perc_labels = y_mcut_10perc_labels[same_idx]
        y_mcut_25perc_labels = y_mcut_25perc_labels[same_idx]
        X = X.loc[same_idx, :]

    if LABEL_UNLANELED:
        no_labels_after_5th_perc = y_mcut_5perc_labels.sum(axis=1)==0
        no_labels_after_5th_perc_idx = \
            no_labels_after_5th_perc.index[no_labels_after_5th_perc]
        y_mcut_5perc_labels.loc[no_labels_after_5th_perc_idx, :] = \
            pd.get_dummies(y_pam50[no_labels_after_5th_perc_idx])
        
        no_labels_after_10th_perc = y_mcut_10perc_labels.sum(axis=1)==0
        no_labels_after_10th_perc_idx = \
            no_labels_after_10th_perc.index[no_labels_after_10th_perc]
        y_mcut_10perc_labels.loc[no_labels_after_10th_perc_idx, :] = \
            pd.get_dummies(y_pam50[no_labels_after_10th_perc_idx])
        
        no_labels_after_25th_perc = y_mcut_25perc_labels.sum(axis=1)==0
        no_labels_after_25th_perc_idx = \
            no_labels_after_25th_perc.index[no_labels_after_25th_perc]
        y_mcut_25perc_labels.loc[no_labels_after_25th_perc_idx, :] = \
            pd.get_dummies(y_pam50[no_labels_after_25th_perc_idx])

    no_labels_after_5th_perc = y_mcut_5perc_labels.sum(axis=1)==0
    misclass_no_labels_after_5th_perc = \
        set(misclass_samples_idx).intersection(
            set(no_labels_after_5th_perc.index[no_labels_after_5th_perc].tolist()))
    print('Among this missclassified samples, number of ones that \
    have zero labels assigned after 5th percentile filtering: ', len(misclass_no_labels_after_5th_perc))

    no_labels_after_10th_perc = y_mcut_10perc_labels.sum(axis=1)==0
    misclass_no_labels_after_10th_perc = \
        set(misclass_samples_idx).intersection(
            set(no_labels_after_10th_perc.index[no_labels_after_10th_perc].tolist()))
    print('Among this missclassified samples, number of ones that \
    have zero labels assigned after 10th percentile filtering: ', len(misclass_no_labels_after_10th_perc))
    no_labels_after_25th_perc = y_mcut_25perc_labels.sum(axis=1)==0
    misclass_no_labels_after_25th_perc = \
        set(misclass_samples_idx).intersection(
            set(no_labels_after_25th_perc.index[no_labels_after_25th_perc].tolist()))
    print('Among this missclassified samples, number of ones that \
    have zero labels assigned after 25th percentile filtering: ', len(misclass_no_labels_after_25th_perc))

    # Plot stacked bars for m-cut (non-negative correlations)
    ax_orig_5_perc = plot_stacked_bars(y_pam50, y_mcut_labels, y_mcut_5perc_labels)
    ax_orig_5_perc.set_title(
        'Labels (M-cut and 5th percentile strategy) count depending on PAM50 label')

    ax_10_15_perc = plot_stacked_bars(y_pam50, y_mcut_10perc_labels, y_mcut_25perc_labels)
    ax_10_15_perc.set_title(
        'Labels (M-cut and 10th and 25th percentile strategy) count depending on PAM50 label')
    
    # Compare class distributions for original and two cases with m-cut
    plot_class_distribution_comparison(data, y_mcut_labels, y_mcut_5perc_labels)
    
    X_train, X_test, \
    y_train_pam50, y_test_pam50, \
    y_train_mcut, y_test_mcut, \
    y_train_orig, y_test_orig, \
    y_train_5perc, y_test_5perc, \
    y_train_10perc, y_test_10perc, \
    y_train_25perc, y_test_25perc = \
        train_test_split(X, y_pam50, y_mcut_labels, y_orig, 
                            y_mcut_5perc_labels, y_mcut_10perc_labels, y_mcut_25perc_labels,
                            test_size=0.3, random_state=1, stratify=y_pam50)

    # Data standardization | normalization
    X_train = X_train.divide(X_train.sum(axis=1), axis=0) * 1e6
    X_test = X_test.divide(X_test.sum(axis=1), axis=0) * 1e6
    scaler = FunctionTransformer(log_transform)
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)

    # Feature selection (based on original case)
    best_feat_model = SelectKBest(score_func=f_classif, k=500) 
    best_feat_model.fit(X_train_scaled, y_train_orig)
    df_scores = pd.DataFrame(best_feat_model.scores_)
    df_feats = pd.DataFrame(X.columns)

    featureScores = pd.concat([df_feats, df_scores],axis=1)
    featureScores.columns = ['Feature', 'Score'] 

    selected_feat = featureScores.sort_values(by='Score')[-500:]['Feature']

    X_train_scaled_selected = X_train_scaled[list(selected_feat)]
    X_test_scaled_selected = X_test_scaled[list(selected_feat)]

    # One-hot encoding of original and PAM50 labels
    y_train_orig = pd.get_dummies(y_train_orig)
    y_test_orig = pd.get_dummies(y_test_orig)
    y_train_pam50 = pd.get_dummies(y_train_pam50)
    y_test_pam50 = pd.get_dummies(y_test_pam50)

################# Select one of 4 best models from the case 0 #################
# XGBoost ---------------- 'run_08-05-2023_10:32:03.pkl',
# Random Forest ---------- 'run_21-06-2023_19:09:12.pkl',
# Logistic Regression ---- 'run_21-06-2023_20:22:21.pkl',
# Support Vector Machine - 'run_21-06-2023_21:42:11.pkl'

#best_model_name = 'bestmodel_' + 'run_08-05-2023_10:32:03.pkl'
#best_model_name = 'bestmodel_' + 'run_21-06-2023_19:09:12.pkl'
#best_model_name = 'bestmodel_' + 'run_21-06-2023_20:22:21.pkl'
best_model_name = 'bestmodel_' + 'run_21-06-2023_21:42:11.pkl'

with open(os.path.join('models', best_model_name), 'rb') as file:
    model = pickle.load(file)

################################ TRAIN & TEST ####################################

# ##################### PROBLEM TRANSFORMATION ####################
# print('------------- Binary Relenace -------------')

# BR_orig = MultiLabel_BinaryRelevance(
#     X_train=X_train_scaled_selected,
#     X_test=X_test_scaled_selected,
#     y_train=y_train_orig, 
#     y_test=y_test_orig)
# predictions_orig = BR_orig.train_test(model)

# BR_pam50 = MultiLabel_BinaryRelevance(
#     X_train=X_train_scaled_selected,
#     X_test=X_test_scaled_selected,
#     y_train=y_train_pam50, 
#     y_test=y_test_pam50)
# predictions_pam50 = BR_pam50.train_test(model)

# BR_mcut = MultiLabel_BinaryRelevance(
#     X_train=X_train_scaled_selected,
#     X_test=X_test_scaled_selected,
#     y_train=y_train_mcut, 
#     y_test=y_test_mcut)
# predictions_mcut = BR_mcut.train_test(model)

# BR_5perc = MultiLabel_BinaryRelevance(
#     X_train=X_train_scaled_selected,
#     X_test=X_test_scaled_selected,
#     y_train=y_train_5perc, 
#     y_test=y_test_5perc)
# predictions_5perc = BR_5perc.train_test(model)

# BR_10perc = MultiLabel_BinaryRelevance(
#     X_train=X_train_scaled_selected,
#     X_test=X_test_scaled_selected,
#     y_train=y_train_10perc, 
#     y_test=y_test_10perc)
# predictions_10perc = BR_10perc.train_test(model)

# BR_25perc = MultiLabel_BinaryRelevance(
#     X_train=X_train_scaled_selected,
#     X_test=X_test_scaled_selected,
#     y_train=y_train_25perc, 
#     y_test=y_test_25perc)
# predictions_25perc = BR_25perc.train_test(model)

# ax_BR = plot_bar_counts_of_label_predictions(predictions_orig, predictions_pam50,
#                                             predictions_mcut, predictions_5perc,
#                                             predictions_10perc, predictions_25perc)
# ax_BR.set_title('Binary Relevance (XGBoost) number of label predictions')

# print('\n-- Original case labels:')
# print_all_scores(y_test_orig, predictions_orig)
# print('-- PAM50 case labels:')
# print_all_scores(y_test_pam50, predictions_pam50)
# print('-- PAM50 case labels after M-cut strategy:')
# print_all_scores(y_test_mcut, predictions_mcut)
# print('-- PAM50 case labels after M-cut strategy:')
# print_all_scores(y_test_mcut, predictions_mcut)
# print('-- PAM50 case labels after M-cut strategy & 5th percentile filtering:')
# print_all_scores(y_test_5perc, predictions_5perc)
# print('-- PAM50 case labels after M-cut strategy & 10th percentile filtering:')
# print_all_scores(y_test_10perc, predictions_10perc)
# print('-- PAM50 case labels after M-cut strategy & 25th percentile filtering:')
# print_all_scores(y_test_25perc, predictions_25perc)

# # print('*** NEW TESTING ***')
# # print('-- PAM50 case labels after M-cut strategy:')
# # print_all_scores(y_test_mcut, predictions_pam50)
# # print('-- PAM50 case labels after M-cut strategy & 5th percentile filtering:')
# # print_all_scores(y_test_5perc, predictions_pam50)
# # print('-- PAM50 case labels after M-cut strategy & 10th percentile filtering:')
# # print_all_scores(y_test_10perc, predictions_pam50)
# # print('-- PAM50 case labels after M-cut strategy & 25th percentile filtering:')
# # print_all_scores(y_test_25perc, predictions_pam50)

# print('\nTest relaxed accuracy (PAM50): {}'.\
#       format(relaxed_accuracy(y_test_orig, predictions_pam50)))
# print('\nTest relaxed accuracy (original): {}'.\
#       format(relaxed_accuracy(y_test_pam50, predictions_pam50)))

# # - Classifier Chain -

# print('------------- Classifier Chain -------------')

# CC_orig = MultiLabel_Chains(
#     X_train=X_train_scaled_selected,
#     X_test=X_test_scaled_selected,
#     y_train=y_train_orig, 
#     y_test=y_test_orig)
# predictions_orig = CC_orig.train_test(model)

# CC_pam50 = MultiLabel_Chains(
#     X_train=X_train_scaled_selected,
#     X_test=X_test_scaled_selected,
#     y_train=y_train_pam50, 
#     y_test=y_test_pam50)
# predictions_pam50 = CC_pam50.train_test(model)

# CC_mcut = MultiLabel_Chains(
#     X_train=X_train_scaled_selected,
#     X_test=X_test_scaled_selected,
#     y_train=y_train_mcut, 
#     y_test=y_test_mcut)
# predictions_mcut = CC_mcut.train_test(model)

# CC_5perc = MultiLabel_Chains(
#     X_train=X_train_scaled_selected,
#     X_test=X_test_scaled_selected,
#     y_train=y_train_5perc, 
#     y_test=y_test_5perc)
# predictions_5perc = CC_5perc.train_test(model)

# CC_10perc = MultiLabel_Chains(
#     X_train=X_train_scaled_selected,
#     X_test=X_test_scaled_selected,
#     y_train=y_train_10perc, 
#     y_test=y_test_10perc)
# predictions_10perc = CC_10perc.train_test(model)

# CC_25perc = MultiLabel_Chains(
#     X_train=X_train_scaled_selected,
#     X_test=X_test_scaled_selected,
#     y_train=y_train_25perc, 
#     y_test=y_test_25perc)
# predictions_25perc = CC_25perc.train_test(model)

# ax_CC = plot_bar_counts_of_label_predictions(predictions_orig, predictions_pam50,
#                                             predictions_mcut, predictions_5perc,
#                                             predictions_10perc, predictions_25perc)
# ax_CC.set_title('Classifier Chain (XGBoost) number of label predictions')


# print('\n-- Original case labels:')
# print_all_scores(y_test_orig, predictions_orig)
# print('-- PAM50 case labels:')
# print_all_scores(y_test_pam50, predictions_pam50)
# print('-- PAM50 case labels after M-cut strategy:')
# print_all_scores(y_test_mcut, predictions_mcut)
# print('-- PAM50 case labels after M-cut strategy & 5th percentile filtering:')
# print_all_scores(y_test_5perc, predictions_5perc)
# print('-- PAM50 case labels after M-cut strategy & 10th percentile filtering:')
# print_all_scores(y_test_10perc, predictions_10perc)
# print('-- PAM50 case labels after M-cut strategy & 25th percentile filtering:')
# print_all_scores(y_test_25perc, predictions_25perc)

# print('\nTest relaxed accuracy (PAM50): {}'.\
#       format(relaxed_accuracy(y_test_orig, predictions_pam50)))
# print('\nTest relaxed accuracy (original): {}'.\
#       format(relaxed_accuracy(y_test_pam50, predictions_pam50)))


# # - Label Powerset -

# print('------------- Label PowerSet -------------')

# LP_orig = MultiLabel_PowerSet(
#     X_train=X_train_scaled_selected,
#     X_test=X_test_scaled_selected,
#     y_train=y_train_orig, 
#     y_test=y_test_orig)
# predictions_orig = LP_orig.train_test(model)

# LP_pam50 = MultiLabel_PowerSet(
#     X_train=X_train_scaled_selected,
#     X_test=X_test_scaled_selected,
#     y_train=y_train_pam50, 
#     y_test=y_test_pam50)
# predictions_pam50 = LP_pam50.train_test(model)

# LP_mcut = MultiLabel_PowerSet(
#     X_train=X_train_scaled_selected,
#     X_test=X_test_scaled_selected,
#     y_train=y_train_mcut, 
#     y_test=y_test_mcut)
# predictions_mcut = LP_mcut.train_test(model)

# LP_5perc = MultiLabel_PowerSet(
#     X_train=X_train_scaled_selected,
#     X_test=X_test_scaled_selected,
#     y_train=y_train_5perc, 
#     y_test=y_test_5perc)
# predictions_5perc = LP_5perc.train_test(model)

# LP_10perc = MultiLabel_PowerSet(
#     X_train=X_train_scaled_selected,
#     X_test=X_test_scaled_selected,
#     y_train=y_train_10perc, 
#     y_test=y_test_10perc)
# predictions_10perc = LP_10perc.train_test(model)

# LP_25perc = MultiLabel_PowerSet(
#     X_train=X_train_scaled_selected,
#     X_test=X_test_scaled_selected,
#     y_train=y_train_25perc, 
#     y_test=y_test_25perc)
# predictions_25perc = LP_25perc.train_test(model)

# ax_LP = plot_bar_counts_of_label_predictions(predictions_orig, predictions_pam50,
#                                             predictions_mcut, predictions_5perc,
#                                             predictions_10perc, predictions_25perc)
# ax_LP.set_title('Label Powerset (XGBoost) number of label predictions')


# print('\n-- Original case labels:')
# print_all_scores(y_test_orig, predictions_orig)
# print('-- PAM50 case labels:')
# print_all_scores(y_test_pam50, predictions_pam50)
# print('-- PAM50 case labels after M-cut strategy:')
# print_all_scores(y_test_mcut, predictions_mcut)
# print('-- PAM50 case labels after M-cut strategy & 5th percentile filtering:')
# print_all_scores(y_test_5perc, predictions_5perc)
# print('-- PAM50 case labels after M-cut strategy & 10th percentile filtering:')
# print_all_scores(y_test_10perc, predictions_10perc)
# print('-- PAM50 case labels after M-cut strategy & 25th percentile filtering:')
# print_all_scores(y_test_25perc, predictions_25perc)

# print('\nTest relaxed accuracy (PAM50): {}'.\
#       format(relaxed_accuracy(y_test_orig, predictions_pam50)))
# print('\nTest relaxed accuracy (original): {}'.\
#       format(relaxed_accuracy(y_test_pam50, predictions_pam50)))

##################### ALGORITHM ADAPTATION ####################

# print('------------- ALGORITHM ADAPTATION -------------')

# print('------------- ML-kNN-------------')

# AA_kNN_orig = MultiLabel_Adapted(
#     X_train=X_train_scaled_selected,
#     X_test=X_test_scaled_selected,
#     y_train=y_train_orig, 
#     y_test=y_test_orig)
# predictions_orig = AA_kNN_orig.train_test(type='MLkNN', model=None)

# AA_kNN_pam50 = MultiLabel_Adapted(
#     X_train=X_train_scaled_selected,
#     X_test=X_test_scaled_selected,
#     y_train=y_train_pam50, 
#     y_test=y_test_pam50)
# predictions_pam50 = AA_kNN_pam50.train_test(type='MLkNN', model=None)

# AA_kNN_mcut = MultiLabel_Adapted(
#     X_train=X_train_scaled_selected,
#     X_test=X_test_scaled_selected,
#     y_train=y_train_mcut, 
#     y_test=y_test_mcut)
# predictions_mcut = AA_kNN_mcut.train_test(type='MLkNN', model=None)

# print('Original case labels:')
# print_all_scores(y_test_orig, predictions_orig)
# print('PAM50 case labels:')
# print_all_scores(y_test_pam50, predictions_pam50)
# print('PAM50 case labels after M-cut strategy:')
# print_all_scores(y_test_mcut, predictions_mcut)

# print('\nTest relaxed accuracy (PAM50): {}'.\
#       format(relaxed_accuracy(y_test_orig, predictions_pam50)))
# print('\nTest relaxed accuracy (original): {}'.\
#       format(relaxed_accuracy(y_test_pam50, predictions_pam50)))

# print('------------- MLARAM ------------')

# AA_ARAM_orig = MultiLabel_Adapted(
#     X_train=X_train_scaled_selected,
#     X_test=X_test_scaled_selected,
#     y_train=y_train_orig, 
#     y_test=y_test_orig)
# predictions_orig = AA_ARAM_orig.train_test(type='MLARAM', model=None)

# AA_ARAM_pam50 = MultiLabel_Adapted(
#     X_train=X_train_scaled_selected,
#     X_test=X_test_scaled_selected,
#     y_train=y_train_pam50, 
#     y_test=y_test_pam50)
# predictions_pam50 = AA_ARAM_pam50.train_test(type='MLARAM', model=None)

# AA_ARAM_mcut = MultiLabel_Adapted(
#     X_train=X_train_scaled_selected,
#     X_test=X_test_scaled_selected,
#     y_train=y_train_mcut, 
#     y_test=y_test_mcut)
# predictions_mcut = AA_ARAM_mcut.train_test(type='MLARAM', model=None)

# print('Original case labels:')
# print_all_scores(y_test_orig, predictions_orig)
# print('PAM50 case labels:')
# print_all_scores(y_test_pam50, predictions_pam50)
# print('PAM50 case labels after M-cut strategy:')
# print_all_scores(y_test_mcut, predictions_mcut)

# print('\nTest relaxed accuracy (PAM50): {}'.\
#       format(relaxed_accuracy(y_test_orig, predictions_pam50)))
# print('\nTest relaxed accuracy (original): {}'.\
#       format(relaxed_accuracy(y_test_pam50, predictions_pam50)))


##################### ALGORITHM ADAPTATION ####################

print('------------- ENSEMBLE METHODS  -------------')

print('------------- ENSEMBLE CHAINS -------------')

ECC_orig = MultiLabel_EnsembleChains(
    X_train=X_train_scaled_selected,
    X_test=X_test_scaled_selected,
    y_train=y_train_orig, 
    y_test=y_test_orig)
predictions_orig = ECC_orig.train_test(model)

ECC_pam50 = MultiLabel_EnsembleChains(
    X_train=X_train_scaled_selected,
    X_test=X_test_scaled_selected,
    y_train=y_train_pam50, 
    y_test=y_test_pam50)
predictions_pam50 = ECC_pam50.train_test(model)

ECC_mcut = MultiLabel_EnsembleChains(
    X_train=X_train_scaled_selected,
    X_test=X_test_scaled_selected,
    y_train=y_train_mcut, 
    y_test=y_test_mcut)
predictions_mcut = ECC_mcut.train_test(model)

print('Original case labels:')
print_all_scores(y_test_orig, predictions_orig)
print('PAM50 case labels:')
print_all_scores(y_test_pam50, predictions_pam50)
print('PAM50 case labels after M-cut strategy:')
print_all_scores(y_test_mcut, predictions_mcut)

print('\nTest relaxed accuracy (PAM50): {}'.\
      format(relaxed_accuracy(y_test_orig, predictions_pam50)))
print('\nTest relaxed accuracy (original): {}'.\
      format(relaxed_accuracy(y_test_pam50, predictions_pam50)))

print('------------- ENSEMBLE RAKEL - DISTINCT -------------')

ER_orig = MultiLabel_EnsembleRakel(
    X_train=X_train_scaled_selected,
    X_test=X_test_scaled_selected,
    y_train=y_train_orig, 
    y_test=y_test_orig)
predictions_orig = ER_orig.train_test(model, type='distinct')

ER_pam50 = MultiLabel_EnsembleRakel(
    X_train=X_train_scaled_selected,
    X_test=X_test_scaled_selected,
    y_train=y_train_pam50, 
    y_test=y_test_pam50)
predictions_pam50 = ER_pam50.train_test(model, type='distinct')

ER_mcut = MultiLabel_EnsembleRakel(
    X_train=X_train_scaled_selected,
    X_test=X_test_scaled_selected,
    y_train=y_train_mcut, 
    y_test=y_test_mcut)
predictions_mcut = ER_mcut.train_test(model, type='distinct')

print('Original case labels:')
print_all_scores(y_test_orig, predictions_orig)
print('PAM50 case labels:')
print_all_scores(y_test_pam50, predictions_pam50)
print('PAM50 case labels after M-cut strategy:')
print_all_scores(y_test_mcut, predictions_mcut)

print('\nTest relaxed accuracy (PAM50): {}'.\
      format(relaxed_accuracy(y_test_orig, predictions_pam50)))
print('\nTest relaxed accuracy (original): {}'.\
      format(relaxed_accuracy(y_test_pam50, predictions_pam50)))


# print('------------- ENSEMBLE RAKEL - OVERLAPPING -------------')

# ER_O_orig = MultiLabel_EnsembleRakel(
#     X_train=X_train_scaled_selected,
#     X_test=X_test_scaled_selected,
#     y_train=y_train_orig, 
#     y_test=y_test_orig)
# predictions_orig = ER_O_orig.train_test(model, type='overlapping')

# ER_O_pam50 = MultiLabel_EnsembleRakel(
#     X_train=X_train_scaled_selected,
#     X_test=X_test_scaled_selected,
#     y_train=y_train_pam50, 
#     y_test=y_test_pam50)
# predictions_pam50 = ER_O_pam50.train_test(model, type='overlapping')

# ER_O_mcut = MultiLabel_EnsembleRakel(
#     X_train=X_train_scaled_selected,
#     X_test=X_test_scaled_selected,
#     y_train=y_train_mcut, 
#     y_test=y_test_mcut)
# predictions_mcut = ER_O_mcut.train_test(model, type='overlapping')

# print('Original case labels:')
# print_all_scores(y_test_orig, predictions_orig)
# print('PAM50 case labels:')
# print_all_scores(y_test_pam50, predictions_pam50)
# print('PAM50 case labels after M-cut strategy:')
# print_all_scores(y_test_mcut, predictions_mcut)

# print('\nTest relaxed accuracy (PAM50): {}'.\
#       format(relaxed_accuracy(y_test_orig, predictions_pam50)))
# print('\nTest relaxed accuracy (original): {}'.\
#       format(relaxed_accuracy(y_test_pam50, predictions_pam50)))