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
from multilabel_metrics import *
import mlflow
import json

# Set parameters
plt.style.use('ggplot')
matplotlib.rcParams['font.size'] = 8
plt.rcParams["figure.figsize"] = (8, 6)

exp_name = datetime.datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
if not os.path.exists(exp_name):
    results_path = 'results/' + exp_name
    os.mkdir(results_path) 

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

    # Assign rank based on the value of the membership/correlation
    y_ranked_labels = y_corr.apply(rank_indices, axis=1) - 1
    y_ranked_labels = pd.DataFrame(y_ranked_labels.values.tolist(), columns=y_corr.columns, index=y_corr.index)

    # Compare class distributions for original and two cases with m-cut
    plot_class_distribution_comparison(data, y_mcut_labels, y_mcut_labels_neg)

    # Compute labels from two strategies (M-cut and 5th percentile)
    y_mcut_5perc_labels = create_mcut_nth_percentile_labels(
        m_cut_labels=y_mcut_labels,
        correlations=y_corr_non_neg,
        y=y_pam50,
        keep_primary=True,
        N=5
    )

    y_mcut_10perc_labels = create_mcut_nth_percentile_labels(
        m_cut_labels=y_mcut_labels,
        correlations=y_corr_non_neg,
        y=y_pam50,
        keep_primary=True,
        N=10
    )

    y_mcut_25perc_labels = create_mcut_nth_percentile_labels(
        m_cut_labels=y_mcut_labels,
        correlations=y_corr_non_neg,
        y=y_pam50,
        keep_primary=True,
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

    # TODO: Plot primary and secondary labels assigned
    ax_prim_vs_sec = plot_stacked_bars_primary_secondary_label_assigned(y_pam50, y_mcut_labels, y_mcut_5perc_labels, 
                                                                         y_mcut_10perc_labels, y_mcut_25perc_labels)

    # Plot stacked bars for all cases
    ax_stacked = plot_stacked_bars(y_pam50, y_mcut_labels, y_mcut_5perc_labels, 
                                       y_mcut_10perc_labels, y_mcut_25perc_labels)
    ax_stacked.set_title(
        'Labels count depending on PAM50 (primary) label')
    
    X_train, X_test, \
    y_train_pam50, y_test_pam50, \
    y_train_mcut, y_test_mcut, \
    y_train_orig, y_test_orig, \
    y_train_5perc, y_test_5perc, \
    y_train_10perc, y_test_10perc, \
    y_train_25perc, y_test_25perc, \
    y_train_ranked, y_test_ranked = \
        train_test_split(X, y_pam50, y_mcut_labels, y_orig, 
                            y_mcut_5perc_labels, y_mcut_10perc_labels, y_mcut_25perc_labels,
                            y_ranked_labels,
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

best_model_name = 'bestmodel_' + 'run_08-05-2023_10:32:03.pkl'
# best_model_name = 'bestmodel_' + 'run_21-06-2023_19:09:12.pkl'
# best_model_name = 'bestmodel_' + 'run_21-06-2023_20:22:21.pkl'
# best_model_name = 'bestmodel_' + 'run_21-06-2023_21:42:11.pkl'

with open(os.path.join('models/multi-label_models', best_model_name), 'rb') as file:
    model = pickle.load(file)

if isinstance(model, LogisticRegression):
    model_name = 'LRegression'
elif isinstance(model, RandomForestClassifier):
    model_name = 'RForest'
elif isinstance(model, xgb.XGBClassifier):
    model_name = 'XGBoost'
elif isinstance(model, SVC):
    model_name = 'SVC'

################################ TRAIN & TEST ####################################

PROBLEM_TRANSF = {
    'Binary Relevance': True,
    'Chain Classifier': False,
    'Label Powerset': False
}

ALGO_ADAPT = {
    'MLkNN': False,
    'MLARAM': False
}

ENSEMBLE = {
    'Chain Classifiers': False,
    'RAKEL': False
}

##################### PROBLEM TRANSFORMATION ####################
if PROBLEM_TRANSF['Binary Relevance']:

    print('------------- Binary Relenace -------------')

    # BR_mcut = MultiLabel_BinaryRelevance(
    #     X_train=X_train_scaled_selected,
    #     X_test=X_test_scaled_selected,
    #     y_train=y_train_mcut, 
    #     y_test=y_test_mcut)
    # predictions_mcut, prob_predictions_mcut = \
    #     BR_mcut.train_test(model, optimize_model=True)

    # print('-- PAM50 case labels after M-cut strategy:')
    # print_all_scores(y_test_mcut, predictions_mcut, prob_predictions_mcut, 
    #                 y_test_orig, y_test_pam50, 
    #                 txt_file_name=os.path.join(results_path, 'BR_' + model_name + '_mcut.txt'))

    BR_rank = MultiLabel_BinaryRelevance(
        X_train=X_train_scaled_selected,
        X_test=X_test_scaled_selected,
        y_train=y_train_ranked, 
        y_test=y_test_ranked)
    predictions_rank, prob_predictions_rank, best_model, best_params, cv_scores = \
        BR_rank.train_test(model, optimize_model=True)

    # BR_5perc = MultiLabel_BinaryRelevance(
    #     X_train=X_train_scaled_selected,
    #     X_test=X_test_scaled_selected,
    #     y_train=y_train_5perc, 
    #     y_test=y_test_5perc)
    # predictions_5perc, prob_predictions_5perc, best_model, best_params, cv_scores = \
    #     BR_5perc.train_test(model, optimize_model=True)

    # print('-- PAM50 case labels after M-cut and 5th percentile strategy:')
    # print_all_scores(y_test_5perc, predictions_5perc, prob_predictions_5perc, 
    #                 y_test_orig, y_test_pam50, 
    #                 txt_file_name=os.path.join(results_path, 'BR_' + model_name + '_mcut_5perc.txt'))
    
    # with open(os.path.join(results_path, 'BR_' + model_name + '_mcut_5perc_predictions.pkl'), 'wb') as f:
    #     pickle.dump(predictions_5perc, f)
    # with open(os.path.join(results_path, 'BR_' + model_name + '_mcut_5perc_prob_predictions.pkl'), 'wb') as f:
    #     pickle.dump(predictions_5perc, f)
    # with open(os.path.join(results_path, 'BR_' + model_name + '_mcut_5perc_bestmodel.pkl'), 'wb') as f:
    #     pickle.dump(best_model, f)
    # with open(os.path.join(results_path, 'BR_' + model_name + '_mcut_5perc_bestparams.pkl'), 'wb') as f:
    #     pickle.dump(best_params, f)
    # with open(os.path.join(results_path, 'BR_' + model_name + '_mcut_5perc_cv_scores.json'), "w") as f:
    #     json.dump(cv_scores, f)

    # BR_10perc = MultiLabel_BinaryRelevance(
    #     X_train=X_train_scaled_selected,
    #     X_test=X_test_scaled_selected,
    #     y_train=y_train_10perc, 
    #     y_test=y_test_10perc)
    # predictions_10perc, prob_predictions_10perc = \
    #     BR_10perc.train_test(model, optimize_model=True)

    # print('-- PAM50 case labels after M-cut and 10th percentile strategy:')
    # print_all_scores(y_test_10perc, predictions_10perc, prob_predictions_10perc, 
    #                 y_test_orig, y_test_pam50, 
    #                 txt_file_name=os.path.join(results_path,'BR_' + model_name + '_mcut_10perc.txt'))

    # BR_25perc = MultiLabel_BinaryRelevance(
    #     X_train=X_train_scaled_selected,
    #     X_test=X_test_scaled_selected,
    #     y_train=y_train_25perc, 
    #     y_test=y_test_25perc)
    # predictions_25perc = BR_25perc.train_test(model, optimize_model=True)

    # ax_BR = plot_bar_counts_of_label_predictions(predictions_orig, predictions_pam50,
    #                                             predictions_mcut, predictions_5perc,
    #                                             predictions_10perc, predictions_25perc)
    # ax_BR.set_title('Binary Relevance (XGBoost) number of label predictions')

# - Classifier Chain -
if PROBLEM_TRANSF['Chain Classifier']:

    print('------------- Classifier Chain -------------')

    # CC_mcut = MultiLabel_Chains(
    #     X_train=X_train_scaled_selected,
    #     X_test=X_test_scaled_selected,
    #     y_train=y_train_mcut, 
    #     y_test=y_test_mcut)
    # predictions_mcut, prob_predictions_mcut = CC_mcut.train_test(model, optimize=False, optimize_model=True)

    # print('-- PAM50 case labels after M-cut strategy:')
    # print_all_scores(y_test_mcut, predictions_mcut, prob_predictions_mcut, 
    #                 y_test_orig, y_test_pam50, 
    #                 txt_file_name=os.path.join(results_path, 'CC_' + model_name + '_mcut.txt'))

    CC_5perc = MultiLabel_Chains(
        X_train=X_train_scaled_selected,
        X_test=X_test_scaled_selected,
        y_train=y_train_5perc, 
        y_test=y_test_5perc)
    predictions_5perc, prob_predictions_5perc, best_model, best_params, cv_scores = \
        CC_5perc.train_test(model,  optimize=True, optimize_model=True)

    print('-- PAM50 case labels after M-cut and 5th percentile strategy:')
    print_all_scores(y_test_5perc, predictions_5perc, prob_predictions_5perc, 
                    y_test_orig, y_test_pam50, 
                    txt_file_name=os.path.join(results_path,'CC_' + model_name + '_mcut_5perc.txt'))
    
    with open(os.path.join(results_path, 'CC_' + model_name + '_mcut_5perc_predictions.pkl')) as f:
        pickle.dump(predictions_5perc, f)
    with open(os.path.join(results_path, 'CC_' + model_name + '_mcut_5perc_prob_predictions.pkl')) as f:
        pickle.dump(predictions_5perc, f)
    with open(os.path.join(results_path, 'CC_' + model_name + '_mcut_5perc_bestmodel.pkl'), 'wb') as f:
        pickle.dump(best_model, f)
    with open(os.path.join(results_path, 'CC_' + model_name + '_mcut_5perc_bestparams.pkl'), 'wb') as f:
        pickle.dump(best_params, f)
    with open(os.path.join(results_path, 'CC_' + model_name + '_mcut_5perc_cv_scores.json'), "w") as f:
        json.dump(cv_scores, f)

    # CC_10perc = MultiLabel_Chains(
    #     X_train=X_train_scaled_selected,
    #     X_test=X_test_scaled_selected,
    #     y_train=y_train_10perc, 
    #     y_test=y_test_10perc)
    # predictions_10perc, prob_predictions_10perc = CC_10perc.train_test(model,  optimize=False, optimize_model=True)

    # print('-- PAM50 case labels after M-cut and 10th percentile strategy:')
    # print_all_scores(y_test_10perc, predictions_10perc, prob_predictions_10perc, 
    #                 y_test_orig, y_test_pam50, 
    #                 txt_file_name=os.path.join(results_path,'CC_' + model_name + '_mcut_10perc.txt'))

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


# - Label Powerset -
if PROBLEM_TRANSF['Label Powerset']:
    print('------------- Label PowerSet -------------')

    # LP_mcut = MultiLabel_PowerSet(
    #     X_train=X_train_scaled_selected,
    #     X_test=X_test_scaled_selected,
    #     y_train=y_train_mcut, 
    #     y_test=y_test_mcut)
    # predictions_mcut, prob_predictions_mcut  = LP_mcut.train_test(model, optimize_model=True)

    # print('-- PAM50 case labels after M-cut strategy:')
    # print_all_scores(y_test_mcut, predictions_mcut, prob_predictions_mcut, 
    #                 y_test_orig, y_test_pam50, 
    #                 txt_file_name=os.path.join(results_path, 'LP_' + model_name + '_mcut.txt'))

    LP_5perc = MultiLabel_PowerSet(
        X_train=X_train_scaled_selected,
        X_test=X_test_scaled_selected,
        y_train=y_train_5perc, 
        y_test=y_test_5perc)
    predictions_5perc, prob_predictions_5perc, best_model, best_params, cv_scores = \
        LP_5perc.train_test(model, optimize_model=True)

    print('-- PAM50 case labels after M-cut & 5th percentile strategy:')
    print_all_scores(y_test_5perc, predictions_5perc, prob_predictions_5perc, 
                    y_test_orig, y_test_pam50, 
                    txt_file_name=os.path.join(results_path,'LP_' + model_name + '_mcut_5perc.txt'))
    
    with open(os.path.join(results_path, 'LP_' + model_name + '_mcut_5perc_predictions.pkl')) as f:
        pickle.dump(predictions_5perc, f)
    with open(os.path.join(results_path, 'LP_' + model_name + '_mcut_5perc_prob_predictions.pkl')) as f:
        pickle.dump(predictions_5perc, f)
    with open(os.path.join(results_path, 'LP_' + model_name + '_mcut_5perc_bestmodel.pkl'), 'wb') as f:
        pickle.dump(best_model, f)
    with open(os.path.join(results_path, 'LP_' + model_name + '_mcut_5perc_bestparams.pkl'), 'wb') as f:
        pickle.dump(best_params, f)
    with open(os.path.join(results_path, 'LP_' + model_name + '_mcut_5perc_cv_scores.json'), "w") as f:
        json.dump(cv_scores, f)

    # LP_10perc = MultiLabel_PowerSet(
    #     X_train=X_train_scaled_selected,
    #     X_test=X_test_scaled_selected,
    #     y_train=y_train_10perc, 
    #     y_test=y_test_10perc)
    # predictions_10perc, prob_predictions_10perc = LP_10perc.train_test(model, optimize_model=True)

    # print('-- PAM50 case labels after M-cut & 10th percentile strategy:')
    # print_all_scores(y_test_10perc, predictions_10perc, prob_predictions_10perc, 
    #                 y_test_orig, y_test_pam50, 
    #                 txt_file_name=os.path.join(results_path,'LP_' + model_name + '_mcut_10perc.txt'))

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


##################### ALGORITHM ADAPTATION ####################

print('------------- ALGORITHM ADAPTATION -------------')

if ALGO_ADAPT['MLARAM']:
    print('------------- MLARAM ------------')

    # AA_kNN_mcut = MultiLabel_Adapted(
    #     X_train=X_train_scaled_selected,
    #     X_test=X_test_scaled_selected,
    #     y_train=y_train_mcut, 
    #     y_test=y_test_mcut)
    # predictions_mcut, prob_predictions_mcut = AA_kNN_mcut.train_test(
    #     type='MLARAM', optimize=True)

    # print('-- PAM50 case labels after M-cut strategy:')
    # print_all_scores(y_test_mcut, predictions_mcut, prob_predictions_mcut, 
    #                 y_test_orig, y_test_pam50, 
    #                 txt_file_name=os.path.join(results_path, 'MLARAM_mcut.txt'))

    AA_kNN_mcut = MultiLabel_Adapted(
        X_train=X_train_scaled_selected,
        X_test=X_test_scaled_selected,
        y_train=y_train_5perc, 
        y_test=y_test_5perc)
    predictions_5perc, prob_predictions_5perc = AA_kNN_mcut.train_test(
        type='MLARAM', optimize=True)


    print('-- PAM50 case labels after M-cut & 5th percentile strategy:')
    print_all_scores(y_test_5perc, predictions_5perc, prob_predictions_5perc, 
                    y_test_orig, y_test_pam50, 
                    txt_file_name=os.path.join(results_path, 'MLARAM_5perc.txt'))
    
    with open(os.path.join(results_path, 'MLARAM_mcut_5perc_predictions.pkl')) as f:
        pickle.dump(predictions_5perc, f)
    with open(os.path.join(results_path, 'MLARAM_mcut_5perc_prob_predictions.pkl')) as f:
        pickle.dump(predictions_5perc, f)

    # AA_kNN_mcut = MultiLabel_Adapted(
    #     X_train=X_train_scaled_selected,
    #     X_test=X_test_scaled_selected,
    #     y_train=y_train_10perc, 
    #     y_test=y_test_10perc)
    # predictions_10perc, prob_predictions_10perc = AA_kNN_mcut.train_test(
    #     type='MLARAM', optimize=True)

    # print('-- PAM50 case labels after M-cut & 10th percentile strategy:')
    # print_all_scores(y_test_10perc, predictions_10perc, prob_predictions_10perc, 
    #                 y_test_orig, y_test_pam50, 
    #                 txt_file_name=os.path.join(results_path, 'MLARAM_10perc.txt'))

if ALGO_ADAPT['MLkNN']:  
    print('------------- ML-kNN-------------')

    # AA_kNN_mcut = MultiLabel_Adapted(
    #     X_train=X_train_scaled_selected,
    #     X_test=X_test_scaled_selected,
    #     y_train=y_train_mcut, 
    #     y_test=y_test_mcut)
    # predictions_mcut, prob_predictions_mcut = AA_kNN_mcut.train_test(
    #     type='MLkNN', optimize=True)

    # print('-- PAM50 case labels after M-cut strategy:')
    # print_all_scores(y_test_mcut, predictions_mcut, prob_predictions_mcut, 
    #                 y_test_orig, y_test_pam50, 
    #                 txt_file_name=os.path.join(results_path, 'MLkNN_mcut.txt'))

    AA_kNN_mcut = MultiLabel_Adapted(
        X_train=X_train_scaled_selected,
        X_test=X_test_scaled_selected,
        y_train=y_train_5perc, 
        y_test=y_test_5perc)
    predictions_5perc, prob_predictions_5perc = AA_kNN_mcut.train_test(
        type='MLkNN', optimize=True)


    print('-- PAM50 case labels after M-cut & 5th percentile strategy:')
    print_all_scores(y_test_5perc, predictions_5perc, prob_predictions_5perc, 
                    y_test_orig, y_test_pam50, 
                    txt_file_name=os.path.join(results_path, 'MLkNN_5perc.txt'))
    
    with open(os.path.join(results_path, 'MLkNN_mcut_5perc_predictions.pkl')) as f:
        pickle.dump(predictions_5perc, f)
    with open(os.path.join(results_path, 'MLkNN_mcut_5perc_prob_predictions.pkl')) as f:
        pickle.dump(predictions_5perc, f)

    # AA_kNN_mcut = MultiLabel_Adapted(
    #     X_train=X_train_scaled_selected,
    #     X_test=X_test_scaled_selected,
    #     y_train=y_train_10perc, 
    #     y_test=y_test_10perc)
    # predictions_10perc, prob_predictions_10perc = AA_kNN_mcut.train_test(
    #     type='MLkNN', optimize=True)

    # print('-- PAM50 case labels after M-cut & 10th percentile strategy:')
    # print_all_scores(y_test_10perc, predictions_10perc, prob_predictions_10perc, 
    #                 y_test_orig, y_test_pam50, 
    #                 txt_file_name=os.path.join(results_path, 'MLkNN_10perc.txt'))



##################### ENSEMBLE METHODS ####################

print('------------- ENSEMBLE METHODS  -------------')

if ENSEMBLE['Chain Classifiers']:
    print('------------- ENSEMBLE CHAINS -------------')

    # ECC_mcut = MultiLabel_EnsembleChains(
    #     X_train=X_train_scaled_selected,
    #     X_test=X_test_scaled_selected,
    #     y_train=y_train_mcut, 
    #     y_test=y_test_mcut)
    # predictions_mcut, prob_predictions_mcut = ECC_mcut.train_test(model, N=50, optimize=True)

    # print('-- PAM50 case labels after M-cut strategy:')
    # print_all_scores(y_test_mcut, predictions_mcut, prob_predictions_mcut, 
    #                 y_test_orig, y_test_pam50, 
    #                 txt_file_name=os.path.join(results_path, 'EnsembleCC_' + model_name + '_mcut.txt'))

    ECC_5perc = MultiLabel_EnsembleChains(
        X_train=X_train_scaled_selected,
        X_test=X_test_scaled_selected,
        y_train=y_train_5perc, 
        y_test=y_test_5perc)
    predictions_5perc, prob_predictions_5perc = ECC_5perc.train_test(model, N=50, optimize=True)

    print('-- PAM50 case labels after M-cut & 5th percentile strategy:')
    print_all_scores(y_test_10perc, predictions_5perc, prob_predictions_5perc, 
                    y_test_orig, y_test_pam50, 
                    txt_file_name=os.path.join(results_path, 'EnsembleCC_' + model_name + '_5perc.txt'))
    
    with open(os.path.join(results_path, 'EnsembleCC_' + model_name + '_mcut_5perc_predictions.pkl')) as f:
        pickle.dump(predictions_5perc, f)
    with open(os.path.join(results_path, 'EnsembleCC_' + model_name + '_mcut_5perc_prob_predictions.pkl')) as f:
        pickle.dump(predictions_5perc, f)

    # ECC_10perc = MultiLabel_EnsembleChains(
    #     X_train=X_train_scaled_selected,
    #     X_test=X_test_scaled_selected,
    #     y_train=y_train_10perc, 
    #     y_test=y_test_10perc)
    # predictions_10perc, prob_predictions_10perc = ECC_10perc.train_test(model, N=50, optimize=True)

    # print('-- PAM50 case labels after M-cut & 10th percentile strategy:')
    # print_all_scores(y_test_10perc, predictions_10perc, prob_predictions_10perc, 
    #                 y_test_orig, y_test_pam50, 
    #                 txt_file_name=os.path.join(results_path, 'EnsembleCC_' + model_name + '_10perc.txt'))
    
if ENSEMBLE['RAKEL']:

    print('------------- ENSEMBLE RAKEL - DISTINCT -------------')

    # ER_mcut = MultiLabel_EnsembleRakel(
    #     X_train=X_train_scaled_selected,
    #     X_test=X_test_scaled_selected,
    #     y_train=y_train_mcut, 
    #     y_test=y_test_mcut)
    # predictions_mcut, prob_predictions_mcut = ER_mcut.train_test(model, optimize=True, type='distinct')

    # print('-- PAM50 case labels after M-cut strategy:')
    # print_all_scores(y_test_mcut, predictions_mcut, prob_predictions_mcut, 
    #                 y_test_orig, y_test_pam50, 
    #                 txt_file_name=os.path.join(results_path, 'EnsembleRakel_' + model_name + '_mcut.txt'))

    ER_5perc = MultiLabel_EnsembleRakel(
        X_train=X_train_scaled_selected,
        X_test=X_test_scaled_selected,
        y_train=y_train_5perc, 
        y_test=y_test_5perc)
    predictions_5perc, prob_predictions_5perc = ER_5perc.train_test(model, optimize=True, type='distinct')

    print('-- PAM50 case labels after M-cut & 5th percentile strategy:')
    print_all_scores(y_test_10perc, predictions_5perc, prob_predictions_5perc, 
                    y_test_orig, y_test_pam50, 
                    txt_file_name=os.path.join(results_path, 'EnsembleRakel_' + model_name + '_5perc.txt'))
    
    with open(os.path.join(results_path, 'EnsembleRakel_' + model_name + '_mcut_5perc_predictions.pkl')) as f:
        pickle.dump(predictions_5perc, f)
    with open(os.path.join(results_path, 'EnsembleRakel_' + model_name + '_mcut_5perc_prob_predictions.pkl')) as f:
        pickle.dump(predictions_5perc, f)

    # ER_10perc = MultiLabel_EnsembleRakel(
    #     X_train=X_train_scaled_selected,
    #     X_test=X_test_scaled_selected,
    #     y_train=y_train_10perc, 
    #     y_test=y_test_10perc)
    # predictions_10perc, prob_predictions_10perc = ER_10perc.train_test(model, optimize=True, type='distinct')

    # print('-- PAM50 case labels after M-cut & 10th percentile strategy:')
    # print_all_scores(y_test_10perc, predictions_10perc, prob_predictions_10perc, 
    #                 y_test_orig, y_test_pam50, 
    #                 txt_file_name=os.path.join(results_path, 'EnsembleRakel_' + model_name + '_10perc.txt'))

