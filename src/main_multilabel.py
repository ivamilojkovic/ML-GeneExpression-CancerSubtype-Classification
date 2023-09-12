import pickle, pandas as pd, os, json
import matplotlib, matplotlib.pyplot as plt

from multilabel_classification import *
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import FunctionTransformer
from utils import *
from multilabel_metrics import *

# Set parameters
plt.style.use('ggplot')
matplotlib.rcParams['font.size'] = 8
plt.rcParams["figure.figsize"] = (8, 6)

exp_name = datetime.datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
if not os.path.exists(exp_name):
    results_path = 'results/' + exp_name
    os.mkdir(results_path) 

DATA_TYPE = 'CRIS'

########################### Load the data ###########################
if DATA_TYPE == 'CRIS':
    label_values = ['CRIS.A', 'CRIS.B', 'CRIS.C', 'CRIS.D', 'CRIS.E']
    with open('data/tcga_cris_raw_24356_620samples.pkl', 'rb') as file:
        data = pickle.load(file) 
    X = data.drop(columns=['Patient ID', 'Subtype-from Parker centroids'] + label_values, inplace=False)
    y_pam50 = data['Subtype-from Parker centroids']
    y_orig = data['Subtype-from Parker centroids'] # this is not important

elif DATA_TYPE == 'BRCA':
    with open('data/dataset_multilabel.pkl', 'rb') as file:
        data = pickle.load(file)
        label_values = ['Basal', 'Her2', 'LumA', 'LumB', 'Normal']
        X = data.drop(columns=[
            'expert_PAM50_subtype', 'tcga_id',
            'Subtype-from Parker centroids', 
            'MaxCorr'] + label_values, inplace=False)
        y_orig = data.expert_PAM50_subtype
        y_pam50 = data['Subtype-from Parker centroids']

# Get the memberships/correlations and set negative ones to zero
y_corr = data[label_values]
y_corr_non_neg = discard_negative_correlations(y_corr)

# M-cut strategy to assign labels (1 or 0) to each class
y_mcut_labels, mcut_threshs = m_cut_strategy_class_assignment(y_corr, non_neg_values=True)
y_mcut_labels_neg, _ = m_cut_strategy_class_assignment(y_corr, non_neg_values=False)
print('M-cut average and std threshold value: ', np.mean(mcut_threshs), np.std(mcut_threshs))

# Check if there are samples with no labels after mcut
samples_no_labels = y_mcut_labels.sum(axis=1) == 0
samples_no_labels_idx = y_mcut_labels.sum(axis=1)[samples_no_labels].index

if samples_no_labels.sum() > 0:
    for i in list(samples_no_labels_idx):
        y_mcut_labels.loc[i, y_pam50.iloc[samples_no_labels_idx][i]] = 1

# Double check
samples_no_labels = y_mcut_labels.sum(axis=1) == 0
print('Number of samples unlabeld: ', samples_no_labels.sum())

# Assign rank based on the value of the membership/correlation
y_ranked_labels = y_corr.apply(rank_indices, axis=1) - 1
y_ranked_labels = pd.DataFrame(y_ranked_labels.values.tolist(), columns=y_corr.columns, index=y_corr.index)

# Compare class distributions for original and two cases with m-cut
plot_class_distribution_comparison(data, y_mcut_labels, y_mcut_labels_neg)

# Compute labels from two strategies (M-cut and 5th percentile)
y_mcut_5perc_labels, _ = create_mcut_nth_percentile_labels(
    m_cut_labels=y_mcut_labels,
    correlations=y_corr_non_neg,
    y=y_pam50,
    keep_primary=False,
    N=5
)

y_mcut_10perc_labels, _ = create_mcut_nth_percentile_labels(
    m_cut_labels=y_mcut_labels,
    correlations=y_corr_non_neg,
    y=y_pam50,
    keep_primary=False,
    N=10
)

y_mcut_25perc_labels, _ = create_mcut_nth_percentile_labels(
    m_cut_labels=y_mcut_labels,
    correlations=y_corr_non_neg,
    y=y_pam50,
    keep_primary=False,
    N=25
)

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
y_train_ranked, y_test_ranked, \
y_train_corr, y_test_corr = \
    train_test_split(X, y_pam50, 
                        y_mcut_labels, y_orig, 
                        y_mcut_5perc_labels, 
                        y_mcut_10perc_labels, 
                        y_mcut_25perc_labels,
                        y_ranked_labels, y_corr,
                        test_size=0.3, random_state=1, stratify=y_pam50)

# Data standardization: log(CMP + 0.1)
X_train = X_train.divide(X_train.sum(axis=1), axis=0) * 1e6
X_test = X_test.divide(X_test.sum(axis=1), axis=0) * 1e6
scaler = FunctionTransformer(log_transform)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)

# Feature selection 

# Load selected features (hybrid)
if DATA_TYPE == 'CRIS':
    hybrid_feat_selection_path = os.path.join('data/cris/new2_without_corr_removed_feat_select_gt_40_perc_occur.pkl') # could be ['hybrid_features_800.pickle'...]
else:
    hybrid_feat_selection_path = os.path.join('data/brca/without_corr_removed_feat_select_gt_50_perc_occur.pkl') # could be ['hybrid_features_800.pickle'...]
        
with open(hybrid_feat_selection_path, 'rb') as file:
    selected_feat = pickle.load(file)

# best_feat_model = SelectKBest(score_func=f_classif, k=500) 
# best_feat_model.fit(X_train_scaled, y_train_orig)
# df_scores = pd.DataFrame(best_feat_model.scores_)
# df_feats = pd.DataFrame(X.columns)

# featureScores = pd.concat([df_feats, df_scores],axis=1)
# featureScores.columns = ['Feature', 'Score'] 

# selected_feat = featureScores.sort_values(by='Score')[-500:]['Feature']

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

ML_APPROACHES = {
    'Problem transformation': ['Classifier Chain', 'Binary Relevance', 'Label Powerset'], # 'Binary Relevance', 'Classifier Chain'
    # 'Ensemble': ['Classifier Chain', 'RAKEL'],
    #'Algorithm adaptation': ['MLkNN', 'MLARAM']
}

MODEL_TYPES = ['XGBoost', 'Logistic Regression', 'Random Forest', 'Support Vector Machine']

for ML_APPROACH in ML_APPROACHES:
    if ML_APPROACH == 'Algorithm adaptation':
        for ML_STRATEGY in ML_APPROACHES[ML_APPROACH]:
            ##################### ALGORITHM ADAPTATION ####################

            print('------------- ALGORITHM ADAPTATION -------------')

            if ML_STRATEGY=='MLARAM':
                print('------------- MLARAM ------------')

                AA_kNN_mcut = MultiLabel_Adapted(
                    X_train=X_train_scaled_selected,
                    X_test=X_test_scaled_selected,
                    y_train=y_train_5perc, 
                    y_test=y_test_5perc)
                predictions_5perc, prob_predictions_5perc, best_model, best_params, cv_scores  = AA_kNN_mcut.train_test(
                    type='MLARAM', optimize=True)

                print('-- PAM50 case labels after M-cut & 5th percentile strategy:')
                print_all_scores(y_test_5perc, predictions_5perc, prob_predictions_5perc, 
                                y_test_orig, y_test_pam50, y_corr=y_test_corr,
                                txt_file_name=os.path.join(results_path, 'MLARAM_5perc.txt'))
                
                with open(os.path.join(results_path, 'MLARAM_mcut_5perc_predictions.pkl'), 'wb') as f:
                    pickle.dump(predictions_5perc, f)
                with open(os.path.join(results_path, 'MLARAM_mcut_5perc_prob_predictions.pkl'), 'wb') as f:
                    pickle.dump(predictions_5perc, f)
                with open(os.path.join(results_path, 'MLARAM_mcut_5perc_bestmodel.pkl'), 'wb') as f:
                    pickle.dump(best_model, f)
                with open(os.path.join(results_path, 'MLARAM_mcut_5perc_bestparams.pkl'), 'wb') as f:
                    pickle.dump(best_params, f)
                with open(os.path.join(results_path, 'MLARAM_mcut_5perc_cv_scores.json'), "w") as f:
                    json.dump(cv_scores, f)

            if ML_STRATEGY=='MLkNN':  
                print('------------- ML-kNN-------------')

                AA_kNN_mcut = MultiLabel_Adapted(
                    X_train=X_train_scaled_selected,
                    X_test=X_test_scaled_selected,
                    y_train=y_train_5perc, 
                    y_test=y_test_5perc)
                predictions_5perc, prob_predictions_5perc, best_model, best_params, cv_scores = AA_kNN_mcut.train_test(
                    type='MLkNN', optimize=True)

                print('-- PAM50 case labels after M-cut & 5th percentile strategy:')
                print_all_scores(y_test_5perc, predictions_5perc, prob_predictions_5perc, 
                                y_test_orig, y_test_pam50, y_corr=y_test_corr,
                                txt_file_name=os.path.join(results_path, 'MLkNN_5perc.txt'))
                
                with open(os.path.join(results_path, 'MLkNN_mcut_5perc_predictions.pkl'), 'wb') as f:
                    pickle.dump(predictions_5perc, f)
                with open(os.path.join(results_path, 'MLkNN_mcut_5perc_prob_predictions.pkl'), 'wb') as f:
                    pickle.dump(predictions_5perc, f)
                with open(os.path.join(results_path, 'MLkNN_mcut_5perc_bestmodel.pkl'), 'wb') as f:
                    pickle.dump(best_model, f)
                with open(os.path.join(results_path, 'MLkNN_mcut_5perc_bestparams.pkl'), 'wb') as f:
                    pickle.dump(best_params, f)
                with open(os.path.join(results_path, 'MLkNN_mcut_5perc_cv_scores.json'), "w") as f:
                    json.dump(cv_scores, f)

            continue

    if ML_APPROACH == 'Problem transformation':
        for MODEL_TYPE in MODEL_TYPES:

            if MODEL_TYPE == 'XGBoost':
                best_model_name = 'bestmodel_' + 'run_08-05-2023_10:32:03.pkl'
            elif MODEL_TYPE == 'Random Forest':
                best_model_name = 'bestmodel_' + 'run_21-06-2023_19:09:12.pkl'
            elif MODEL_TYPE == 'Logistic Regression':
                best_model_name = 'bestmodel_' + 'run_21-06-2023_20:22:21.pkl'
            elif MODEL_TYPE == 'Support Vector Machine':
                best_model_name = 'bestmodel_' + 'run_21-06-2023_21:42:11.pkl'
            else:
                print('This model does not exist')
                break

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

            ############################ PROBLEM TRANSFORMATION ##############################
            for ML_STRATEGY in ML_APPROACHES[ML_APPROACH]:
                if ML_STRATEGY == 'Binary Relevance':

                    print('------------- Binary Relenace -------------')

                    BR_5perc = MultiLabel_BinaryRelevance(
                        X_train=X_train_scaled_selected,
                        X_test=X_test_scaled_selected,
                        y_train=y_train_5perc, 
                        y_test=y_test_5perc)
                    predictions_5perc, prob_predictions_5perc, best_model, best_params, cv_scores = \
                        BR_5perc.train_test(model, optimize_model=True)

                    print('-- PAM50 case labels after M-cut and 5th percentile strategy:')
                    print_all_scores(y_test_5perc, predictions_5perc, prob_predictions_5perc, 
                                    y_test_orig, y_test_pam50, y_corr=y_test_corr,
                                    txt_file_name=os.path.join(results_path, 'BR_' + model_name + '_mcut_5perc.txt'))

                    print('Ordered subset accuracy: ', ordered_subset_accuracy(
                        y_test_mcut=y_test_5perc, predictions=predictions_5perc, 
                        y_test_corr=y_test_corr, prob_predictions=prob_predictions_5perc))
                    print('Subset accuracy for orders below k=1: ', k_orders_subset_accuracy(
                        y_test_mcut=y_test_5perc, predictions=predictions_5perc, 
                        y_test_corr=y_test_corr, prob_predictions=prob_predictions_5perc, k=1))
                    print('Subset accuracy for orders below k=2: ', k_orders_subset_accuracy(
                        y_test_mcut=y_test_5perc, predictions=predictions_5perc, 
                        y_test_corr=y_test_corr, prob_predictions=prob_predictions_5perc, k=2))
                    print('Subset accuracy for orders below k=3: ', k_orders_subset_accuracy(
                        y_test_mcut=y_test_5perc, predictions=predictions_5perc, 
                        y_test_corr=y_test_corr, prob_predictions=prob_predictions_5perc, k=3))
                    
                    with open(os.path.join(results_path, 'BR_' + model_name + '_mcut_5perc.txt'), 'a') as file:
                        additional_lines = \
                            "\nOrdered subset accuracy: " + str(ordered_subset_accuracy(
                        y_test_mcut=y_test_5perc, predictions=predictions_5perc, 
                        y_test_corr=y_test_corr, prob_predictions=prob_predictions_5perc)) + \
                            "\nSubset accuracy for orders below k=1: " + str( k_orders_subset_accuracy(
                        y_test_mcut=y_test_5perc, predictions=predictions_5perc, 
                        y_test_corr=y_test_corr, prob_predictions=prob_predictions_5perc, k=1)) + \
                            "\nSubset accuracy for orders below k=2: " + str( k_orders_subset_accuracy(
                        y_test_mcut=y_test_5perc, predictions=predictions_5perc, 
                        y_test_corr=y_test_corr, prob_predictions=prob_predictions_5perc, k=2)) + \
                            "\nSubset accuracy for orders below k=3: " + str( k_orders_subset_accuracy(
                        y_test_mcut=y_test_5perc, predictions=predictions_5perc, 
                        y_test_corr=y_test_corr, prob_predictions=prob_predictions_5perc, k=3)) 

                        # Write the additional lines to the file
                        file.write(additional_lines)
                    
                    with open(os.path.join(results_path, 'BR_' + model_name + '_mcut_5perc_predictions.pkl'), 'wb') as f:
                        pickle.dump(predictions_5perc, f)
                    with open(os.path.join(results_path, 'BR_' + model_name + '_mcut_5perc_prob_predictions.pkl'), 'wb') as f:
                        pickle.dump(predictions_5perc, f)
                    with open(os.path.join(results_path, 'BR_' + model_name + '_mcut_5perc_bestmodel.pkl'), 'wb') as f:
                        pickle.dump(best_model, f)
                    with open(os.path.join(results_path, 'BR_' + model_name + '_mcut_5perc_bestparams.pkl'), 'wb') as f:
                        pickle.dump(best_params, f)
                    with open(os.path.join(results_path, 'BR_' + model_name + '_mcut_5perc_cv_scores.json'), "w") as f:
                        json.dump(cv_scores, f)

                # - Classifier Chain -
                if ML_STRATEGY == 'Classifier Chain':

                    print('------------- Classifier Chain -------------')

                    CC_5perc = MultiLabel_Chains(
                        X_train=X_train_scaled_selected,
                        X_test=X_test_scaled_selected,
                        y_train=y_train_5perc, 
                        y_test=y_test_5perc)
                    predictions_5perc, prob_predictions_5perc, best_model, best_params, cv_scores = \
                        CC_5perc.train_test(model,  optimize=False, optimize_model=True)

                    print('-- PAM50 case labels after M-cut and 5th percentile strategy:')
                    print_all_scores(y_test_5perc, predictions_5perc, prob_predictions_5perc, 
                                    y_test_orig, y_test_pam50, y_corr=y_test_corr,
                                    txt_file_name=os.path.join(results_path,'CC_' + model_name + '_mcut_5perc.txt'))
                    
                    with open(os.path.join(results_path,'CC_' + model_name + '_mcut_5perc.txt'), 'a') as file:
                        additional_lines = \
                            "\nOrdered subset accuracy: " + str(ordered_subset_accuracy(
                        y_test_mcut=y_test_5perc, predictions=predictions_5perc, 
                        y_test_corr=y_test_corr, prob_predictions=prob_predictions_5perc)) + \
                            "\nSubset accuracy for orders below k=1: " + str( k_orders_subset_accuracy(
                        y_test_mcut=y_test_5perc, predictions=predictions_5perc, 
                        y_test_corr=y_test_corr, prob_predictions=prob_predictions_5perc, k=1)) + \
                            "\nSubset accuracy for orders below k=2: " + str( k_orders_subset_accuracy(
                        y_test_mcut=y_test_5perc, predictions=predictions_5perc, 
                        y_test_corr=y_test_corr, prob_predictions=prob_predictions_5perc, k=2)) + \
                            "\nSubset accuracy for orders below k=3: " + str( k_orders_subset_accuracy(
                        y_test_mcut=y_test_5perc, predictions=predictions_5perc, 
                        y_test_corr=y_test_corr, prob_predictions=prob_predictions_5perc, k=3)) 

                        # Write the additional lines to the file
                        file.write(additional_lines)
                    
                    with open(os.path.join(results_path, 'CC_' + model_name + '_mcut_5perc_predictions.pkl'), 'wb') as f:
                        pickle.dump(predictions_5perc, f)
                    with open(os.path.join(results_path, 'CC_' + model_name + '_mcut_5perc_prob_predictions.pkl'), 'wb') as f:
                        pickle.dump(predictions_5perc, f)
                    with open(os.path.join(results_path, 'CC_' + model_name + '_mcut_5perc_bestmodel.pkl'), 'wb') as f:
                        pickle.dump(best_model, f)
                    with open(os.path.join(results_path, 'CC_' + model_name + '_mcut_5perc_bestparams.pkl'), 'wb') as f:
                        pickle.dump(best_params, f)
                    with open(os.path.join(results_path, 'CC_' + model_name + '_mcut_5perc_cv_scores.json'), "w") as f:
                        json.dump(cv_scores, f)

                # - Label Powerset -
                if ML_STRATEGY == 'Label Powerset':
                    print('------------- Label PowerSet -------------')

                    LP_5perc = MultiLabel_PowerSet(
                        X_train=X_train_scaled_selected,
                        X_test=X_test_scaled_selected,
                        y_train=y_train_5perc, 
                        y_test=y_test_5perc)
                    predictions_5perc, prob_predictions_5perc, best_model, best_params, cv_scores = \
                        LP_5perc.train_test(model, optimize_model=True)

                    print('-- PAM50 case labels after M-cut & 5th percentile strategy:')
                    print_all_scores(y_test_5perc, predictions_5perc, prob_predictions_5perc, 
                                    y_test_orig, y_test_pam50, y_corr=y_test_corr,
                                    txt_file_name=os.path.join(results_path, 'LP_' + model_name + '_mcut_5perc.txt'))
                    
                    with open(os.path.join(results_path, 'LP_' + model_name + '_mcut_5perc.txt'), 'a') as file:
                        additional_lines = \
                            "\nOrdered subset accuracy: " + str(ordered_subset_accuracy(
                        y_test_mcut=y_test_5perc, predictions=predictions_5perc, 
                        y_test_corr=y_test_corr, prob_predictions=prob_predictions_5perc)) + \
                            "\nSubset accuracy for orders below k=1: " + str( k_orders_subset_accuracy(
                        y_test_mcut=y_test_5perc, predictions=predictions_5perc, 
                        y_test_corr=y_test_corr, prob_predictions=prob_predictions_5perc, k=1)) + \
                            "\nSubset accuracy for orders below k=2: " + str( k_orders_subset_accuracy(
                        y_test_mcut=y_test_5perc, predictions=predictions_5perc, 
                        y_test_corr=y_test_corr, prob_predictions=prob_predictions_5perc, k=2)) + \
                            "\nSubset accuracy for orders below k=3: " + str( k_orders_subset_accuracy(
                        y_test_mcut=y_test_5perc, predictions=predictions_5perc, 
                        y_test_corr=y_test_corr, prob_predictions=prob_predictions_5perc, k=3)) 

                        # Write the additional lines to the file
                        file.write(additional_lines)
                    
                    with open(os.path.join(results_path, 'LP_' + model_name + '_mcut_5perc_predictions.pkl'), 'wb') as f:
                        pickle.dump(predictions_5perc, f)
                    with open(os.path.join(results_path, 'LP_' + model_name + '_mcut_5perc_prob_predictions.pkl'), 'wb') as f:
                        pickle.dump(predictions_5perc, f)
                    with open(os.path.join(results_path, 'LP_' + model_name + '_mcut_5perc_bestmodel.pkl'), 'wb') as f:
                        pickle.dump(best_model, f)
                    with open(os.path.join(results_path, 'LP_' + model_name + '_mcut_5perc_bestparams.pkl'), 'wb') as f:
                        pickle.dump(best_params, f)
                    with open(os.path.join(results_path, 'LP_' + model_name + '_mcut_5perc_cv_scores.json'), "w") as f:
                        json.dump(cv_scores, f)

    ##################### ENSEMBLE METHODS ####################
    if ML_APPROACH == 'Ensemble':
    
        for ML_STRATEGY in ML_APPROACHES[ML_APPROACH]:

            if ML_STRATEGY=='Classifier Chain':
                print('------------- ENSEMBLE CHAINS -------------')

                for MODEL_TYPE in MODEL_TYPES:

                    if MODEL_TYPE == 'XGBoost':
                        best_model_name = 'bestmodel_' + 'run_08-05-2023_10:32:03.pkl'
                    elif MODEL_TYPE == 'Random Forest':
                        best_model_name = 'bestmodel_' + 'run_21-06-2023_19:09:12.pkl'
                    elif MODEL_TYPE == 'Logistic Regression':
                        best_model_name = 'bestmodel_' + 'run_21-06-2023_20:22:21.pkl'
                    elif MODEL_TYPE == 'Support Vector Machine':
                        best_model_name = 'bestmodel_' + 'run_21-06-2023_21:42:11.pkl'
                    else:
                        print('This model does not exist')
                        break

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


                    ECC_5perc = MultiLabel_EnsembleChains(
                        X_train=X_train_scaled_selected,
                        X_test=X_test_scaled_selected,
                        y_train=y_train_5perc, 
                        y_test=y_test_5perc)
                    predictions_5perc, prob_predictions_5perc, best_model, best_params, cv_scores = ECC_5perc.train_test(model, N=50, optimize=True)

                    print('-- PAM50 case labels after M-cut & 5th percentile strategy:')
                    print_all_scores(y_test_10perc, predictions_5perc, prob_predictions_5perc, 
                                    y_test_orig, y_test_pam50, y_corr=y_test_corr,
                                    txt_file_name=os.path.join(results_path, 'EnsembleCC_' + model_name + '_5perc.txt'))
                    
                    with open(os.path.join(results_path,'EnsembleCC_' + model_name + '_5perc.txt'), 'a') as file:
                        additional_lines = \
                            "\nOrdered subset accuracy: " + str(ordered_subset_accuracy(
                        y_test_mcut=y_test_5perc, predictions=predictions_5perc, 
                        y_test_corr=y_test_corr, prob_predictions=prob_predictions_5perc)) + \
                            "\nSubset accuracy for orders below k=1: " + str( k_orders_subset_accuracy(
                        y_test_mcut=y_test_5perc, predictions=predictions_5perc, 
                        y_test_corr=y_test_corr, prob_predictions=prob_predictions_5perc, k=1)) + \
                            "\nSubset accuracy for orders below k=2: " + str( k_orders_subset_accuracy(
                        y_test_mcut=y_test_5perc, predictions=predictions_5perc, 
                        y_test_corr=y_test_corr, prob_predictions=prob_predictions_5perc, k=2)) + \
                            "\nSubset accuracy for orders below k=3: " + str( k_orders_subset_accuracy(
                        y_test_mcut=y_test_5perc, predictions=predictions_5perc, 
                        y_test_corr=y_test_corr, prob_predictions=prob_predictions_5perc, k=3)) 

                        # Write the additional lines to the file
                        file.write(additional_lines)

                    with open(os.path.join(results_path, 'EnsembleCC_' + model_name + '_mcut_5perc_predictions.pkl'), 'wb') as f:
                        pickle.dump(predictions_5perc, f)
                    with open(os.path.join(results_path, 'EnsembleCC_' + model_name + '_mcut_5perc_prob_predictions.pkl'), 'wb') as f:
                        pickle.dump(predictions_5perc, f)
                    with open(os.path.join(results_path, 'EnsembleCC_' + model_name + '_mcut_5perc_bestmodel.pkl'), 'wb') as f:
                        pickle.dump(best_model, f)
                    with open(os.path.join(results_path, 'EnsembleCC_' + model_name + '_mcut_5perc_bestparams.pkl'), 'wb') as f:
                        pickle.dump(best_params, f)
                    with open(os.path.join(results_path, 'EnsembleCC_' + model_name + '_mcut_5perc_cv_scores.json'), "w") as f:
                        json.dump(cv_scores, f)

                
            if ML_STRATEGY=='RAKEL':

                print('------------- ENSEMBLE RAKEL - DISTINCT -------------')

                for MODEL_TYPE in MODEL_TYPES:

                    if MODEL_TYPE == 'XGBoost':
                        best_model_name = 'bestmodel_' + 'run_08-05-2023_10:32:03.pkl'
                    elif MODEL_TYPE == 'Random Forest':
                        best_model_name = 'bestmodel_' + 'run_21-06-2023_19:09:12.pkl'
                    elif MODEL_TYPE == 'Logistic Regression':
                        best_model_name = 'bestmodel_' + 'run_21-06-2023_20:22:21.pkl'
                    elif MODEL_TYPE == 'Support Vector Machine':
                        best_model_name = 'bestmodel_' + 'run_21-06-2023_21:42:11.pkl'
                    else:
                        print('This model does not exist')
                        break

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

                    ER_5perc = MultiLabel_EnsembleRakel(
                        X_train=X_train_scaled_selected,
                        X_test=X_test_scaled_selected,
                        y_train=y_train_5perc, 
                        y_test=y_test_5perc)
                    predictions_5perc, prob_predictions_5perc, best_model, best_params, cv_scores = ER_5perc.train_test(model, optimize=True, type='distinct')

                    print('-- PAM50 case labels after M-cut & 5th percentile strategy:')
                    print_all_scores(y_test_10perc, predictions_5perc, prob_predictions_5perc, 
                                    y_test_orig, y_test_pam50, y_corr=y_test_corr,
                                    txt_file_name=os.path.join(results_path, 'EnsembleRakel_' + model_name + '_5perc.txt'))
                    
                    with open(os.path.join(results_path, 'EnsembleRakel_' + model_name + '_5perc.txt'), 'a') as file:
                        additional_lines = \
                            "\nOrdered subset accuracy: " + str(ordered_subset_accuracy(
                        y_test_mcut=y_test_5perc, predictions=predictions_5perc, 
                        y_test_corr=y_test_corr, prob_predictions=prob_predictions_5perc)) + \
                            "\nSubset accuracy for orders below k=1: " + str( k_orders_subset_accuracy(
                        y_test_mcut=y_test_5perc, predictions=predictions_5perc, 
                        y_test_corr=y_test_corr, prob_predictions=prob_predictions_5perc, k=1)) + \
                            "\nSubset accuracy for orders below k=2: " + str( k_orders_subset_accuracy(
                        y_test_mcut=y_test_5perc, predictions=predictions_5perc, 
                        y_test_corr=y_test_corr, prob_predictions=prob_predictions_5perc, k=2)) + \
                            "\nSubset accuracy for orders below k=3: " + str( k_orders_subset_accuracy(
                        y_test_mcut=y_test_5perc, predictions=predictions_5perc, 
                        y_test_corr=y_test_corr, prob_predictions=prob_predictions_5perc, k=3)) 

                        # Write the additional lines to the file
                        file.write(additional_lines)

                    with open(os.path.join(results_path, 'EnsembleRakel_' + model_name + '_mcut_5perc_predictions.pkl'), 'wb') as f:
                        pickle.dump(predictions_5perc, f)
                    with open(os.path.join(results_path, 'EnsembleRakel_' + model_name + '_mcut_5perc_prob_predictions.pkl'), 'wb') as f:
                        pickle.dump(predictions_5perc, f)
                    with open(os.path.join(results_path, 'EnsembleRakel_' + model_name + '_mcut_5perc_bestmodel.pkl'), 'wb') as f:
                        pickle.dump(best_model, f)
                    with open(os.path.join(results_path, 'EnsembleRakel_' + model_name + '_mcut_5perc_bestparams.pkl'), 'wb') as f:
                        pickle.dump(best_params, f)
                    with open(os.path.join(results_path, 'EnsembleRakel_' + model_name + '_mcut_5perc_cv_scores.json'), "w") as f:
                        json.dump(cv_scores, f)



                