from sklearn.preprocessing import LabelEncoder, FunctionTransformer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import SelectFromModel, SelectKBest, RFE, f_classif, chi2
from mlxtend.feature_selection import SequentialFeatureSelector
import os, pickle, datetime, time

import pickle as pkl
from data_preprocessing import ClassBalance, remove_extreme
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from utils import cmp_metrics
import xgboost as xgb
import lightgbm as lgb
from utils import log_transform, plot_before_after_counts

plt.style.use('ggplot')
sns.set_theme()


def test_on(exp_path: str, dict_sampled: dict, n_times: int = 1):

    TEST_DOWNSAMPLE=False

    # Extract experiment params
    curr_dir = os.getcwd()
    with open(os.path.join(curr_dir, 'experiments', exp_path), 'rb') as file:
        exp = pickle.load(file)

    exp_name = exp_path.split('/')[-1]
    with open('models/bestmodel_' + exp_name, 'rb') as f:
        model = pickle.load(f)

    # Load the dataset
    DATASET_PATH = "tcga_brca_raw_19036_1053samples.pkl"

    with open(DATASET_PATH, 'rb') as file:
        dataset = pkl.load(file) 

    X = dataset.drop(columns=['expert_PAM50_subtype', 'tcga_id', \
                              'sample_id', 'cancer_type'], inplace=False)
    y = dataset.expert_PAM50_subtype

    # Remove extreme values (genes, samples)
    X, potential_samples_to_remove = remove_extreme(X)

    # Split the dataset
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=exp['test_size'], 
                         random_state=1, shuffle=True, stratify=y)     

    # Downsample testset - same number of samples in each class
    if TEST_DOWNSAMPLE:
        counts_test_before = y_test.value_counts()
        min_num_samples = (y_test == 'Normal').sum()
        sampling_strategy = {
                'LumA': min_num_samples,
                'LumB': min_num_samples,
                'Basal': min_num_samples,
                'Her2': min_num_samples,
                'Normal': min_num_samples
            }
        cb = ClassBalance(X=X_test, y=y_test)
        uniform_test = cb.resampling(sampling_strategy)

        X_test = uniform_test.drop(columns='expert_PAM50_subtype', inplace=False)
        y_test = uniform_test.expert_PAM50_subtype
        counts_test_after = y_test.value_counts()

        for sample in y_test.index:
            if sample not in dict_sampled:
                dict_sampled[sample] = 0
            dict_sampled[sample] += 1

        # Plot class balance difference 
        plot_before_after_counts(counts_test_before, counts_test_after)

    # Encode the class labels
    LB = LabelEncoder()
    y_train = LB.fit_transform(y_train)
    y_test = LB.transform(y_test)

    # Data standardization | normalization
    X_train = X_train.divide(X_train.sum(axis=1), axis=0) * 1e6
    X_test = X_test.divide(X_test.sum(axis=1), axis=0) * 1e6
    scaler = FunctionTransformer(log_transform)
    X_train_scaled = scaler.fit_transform(X_train)
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_test_scaled = scaler.transform(X_test)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)

    # Feature selection
    if exp['feature_selection_method'] == 'univariate':
        best_feat_model = SelectKBest(score_func=f_classif, k=exp['n_features_to_select']) # k needs to be defined
        best_feat_model.fit(X_train_scaled, y_train)
        df_scores = pd.DataFrame(best_feat_model.scores_)
        df_feats = pd.DataFrame(X.columns)

        featureScores = pd.concat([df_feats, df_scores],axis=1)
        featureScores.columns = ['Feature', 'Score'] 
        
        plt.figure()
        featureScores.nlargest(50, 'Score').plot(kind='barh')
        plt.title(exp['feature_selection_method'] + ' feature selection')
        print(featureScores.nlargest(10, 'Score'))

        selected_feat = featureScores.sort_values(by='Score')[-exp['n_features_to_select']:]['Feature']

    # Extract the data frames with only selected features
    X_train_scaled_selected = X_train_scaled[list(selected_feat)]
    X_test_scaled_selected = X_test_scaled[list(selected_feat)]

    # Test on this test set
    # --------------- Compute predictions ------------------
    pred = model.predict(X_test_scaled_selected.values)
    pred_train = model.predict(X_train_scaled_selected.values)

    # ---------------- Compute metrics ---------------------
    print('------ Test scores ------')
    metrics = cmp_metrics(pred, y_test)
    print('------- Train scores --------')
    metrics_train = cmp_metrics(pred_train, y_train)
    
    return pred, metrics, dict_sampled
    
if __name__=='__main__':
    N_iter = 1
    prec, rec, f1 = [], [], []
    dict_sampled = {}
    for n in range(N_iter):
        # best case-1: 'run_21-04-2023_14:47:19.pkl', 'run_28-04-2023_03:08:41.pkl'
        # best case-2: 'run_14-04-2023_13:35:12.pkl', 'run_28-04-2023_04:20:20.pkl'
        # best case-3: 'run_15-04-2023_01:02:58.pkl', 'run_28-04-2023_02:43:48.pkl'
        pred, dict_metrics, dict_sampled = test_on('run_28-04-2023_02:43:48.pkl', dict_sampled)
        plt.close('all')

        prec.append(dict_metrics['Precision per class'])
        rec.append(dict_metrics['Recall per class'])
        f1.append(dict_metrics['F1 score per class'])

    prec_stacked = np.stack(prec)
    rec_stacked = np.stack(rec)
    f1_stacked = np.stack(f1)

    mean_prec = prec_stacked.mean(axis=0)
    mean_rec =prec_stacked.mean(axis=0)
    mean_f1 = f1_stacked.mean(axis=0)

    std_prec = prec_stacked.std(axis=0)
    std_rec = rec_stacked.std(axis=0)
    std_f1 = f1_stacked.std(axis=0)

    ax = plt.figure()
    df_lr = pd.DataFrame({'Precision': mean_prec,
                    'Recall': mean_rec,
                    'F1 score': mean_f1}, 
                    index=['LumA', 'LumB', 'Basal', 'Her2', 'Normal'])
    ax = df_lr.plot(kind='bar', rot=30, title='Scores for case-3 best model',
                    yerr=[std_prec, std_rec, std_f1])
    plt.legend(loc='lower right')
    print()

    



    




        
    




