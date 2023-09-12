from sklearn.preprocessing import LabelEncoder, FunctionTransformer
from sklearn.metrics import confusion_matrix, make_scorer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import SelectFromModel, SelectKBest, RFE, RFECV, f_classif
from sklearn.decomposition import PCA
from mlxtend.feature_selection import SequentialFeatureSelector
import xgboost as xgb, lightgbm as lgb

import os, datetime, time
import pickle, pandas as pd, numpy as np, json
import matplotlib.pyplot as plt, seaborn as sns
import matplotlib as mpl

from data_preprocessing import ClassBalance, remove_extreme
from utils import cmp_metrics, m_cut_strategy_class_assignment
from utils import log_transform, plot_before_after_counts, plot_pca, NumpyEncoder
from config_model import *
from singlelabel_metrics import *

plt.style.use('ggplot')
plt.rcParams.update({'font.size': 12})
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.serif"] = ["Times New Roman"]
sns.set_theme()

from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)

import mlflow, hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf
from config import *

import warnings
warnings.filterwarnings("ignore")

cs = ConfigStore.instance()
cs.store(name='project_config', node=ProjectConfig)

######################### MAIN ###############################

@hydra.main(config_path="../conf", config_name="config")
def main(cfg: ProjectConfig):

    plt.style.use('ggplot')
    plt.rcParams.update({'font.size': 12})
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"]
    sns.set_theme()

    # Loading the dataset
    if cfg.train.brca_cris == 'BRCA':
        if cfg.train.use_multilabel_dataset:
            with open(cfg.paths.ml_dataset, 'rb') as file:
                dataset = pickle.load(file) 
            X = dataset.drop(columns=['expert_PAM50_subtype', 'tcga_id',
                            'Subtype-from Parker centroids', 'MaxCorr',
                                'Basal', 'Her2', 'LumA', 'LumB', 'Normal'], inplace=False)
            y = dataset['Subtype-from Parker centroids']
        else:
            with open(cfg.paths.brca_dataset, 'rb') as file:
                dataset = pickle.load(file) 
            X = dataset.drop(columns=['expert_PAM50_subtype', 'tcga_id', \
                                    'sample_id', 'cancer_type'], inplace=False)
            y = dataset.expert_PAM50_subtype

    elif cfg.train.brca_cris == 'CRIS':
        label_values = ['CRIS.A', 'CRIS.B', 'CRIS.C', 'CRIS.D', 'CRIS.E']
        with open(cfg.paths.cris_dataset, 'rb') as file:
            dataset = pickle.load(file) 
        X = dataset.drop(columns=['Patient ID', 'Subtype-from Parker centroids'] + label_values, inplace=False)
        y = dataset['Subtype-from Parker centroids']
    
    # Remove extreme values (genes, samples) from initial preprocessing
    X, potential_samples_to_remove, \
        feat_to_remove, feat_to_keep = remove_extreme(X, change_X = True)

    # Split the dataset
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=cfg.train.test_size, 
                        random_state=cfg.train.random_state_split, shuffle=True, stratify=y)    

    ax = y_train.value_counts().plot(kind='bar', title='Class label before')
    counts_before = y_train.value_counts().values
    counts_test_before = y_test.value_counts().values
    ax.tick_params(axis='x', rotation=30)

    # Solve data imbalance issue
    if cfg.train.solve_class_imbalance:
        cb = ClassBalance(X=X_train, y=y_train)
        # cb_test = ClassBalance(X=X_test, y=y_test)
        if cfg.train.type_class_imbalance == 'case1':
            balanced_dataset = cb.cut_LumA(thresh=cfg.train.thresh_lumA)
            # counts_after = balanced_dataset.expert_PAM50_subtype.value_counts()
            # ratios_after = counts_after/sum(counts_after)
            # thresh_new = np.floor(X_test.shape[0]*ratios_after)
            # new_test_set = cb_test.resampling(
            #     balance_treshs=thresh_new, seed=42)

        elif cfg.train.type_class_imbalance == 'case2':
            balanced_dataset = cb.cut_LumA_LumB_Basal(thresh=[100, 100, 80])
            counts_after = balanced_dataset.expert_PAM50_subtype.value_counts()
            
        elif cfg.train.type_class_imbalance == 'case3':
            sampling_strategy = {
                'LumA': 100,
                'LumB': 100,
                'Basal': 100,
                'Her2': 80,
                'Normal': 80
            }
            balanced_dataset, new_samples = cb.resampling_with_generation(sampling_strategy)
            counts_after = balanced_dataset.expert_PAM50_subtype.value_counts()

        elif cfg.train.type_class_imbalance == 'case4':
            sampling_strategy = {
                'LumA': 100,
                'LumB': 100,
                'Basal': 100,
                'Her2': 100,
                'Normal': 100
            }
            balanced_dataset, new_samples = cb.resampling_with_generation(sampling_strategy)
            counts_after = balanced_dataset.expert_PAM50_subtype.value_counts()

        elif cfg.train.type_class_imbalance == 'case5':
            class_counts = y_train.value_counts()
            mul = 1.5 # Increase with 50% 
            new_class_counts = round(class_counts*mul)
            counts_after = new_class_counts.values
            counts_diff = list(np.subtract(counts_after, counts_before))

            sampling_strategy = {
                'LumA': int(new_class_counts['LumA']),
                'LumB': int(new_class_counts['LumB']),
                'Basal': int(new_class_counts['Basal']),
                'Her2': int(new_class_counts['Her2']),
                'Normal': int(new_class_counts['Normal'])
            }
            print('New class count:\n', sampling_strategy)
            
            balanced_dataset = cb.resampling_with_generation(sampling_strategy)
            experiment_params['class_balance_thresholds'] = sampling_strategy

        X_train = balanced_dataset.drop(columns='expert_PAM50_subtype', inplace=False)
        y_train = balanced_dataset.expert_PAM50_subtype

        # Plot class balance difference 
        plot_before_after_counts(counts_before, counts_after)

    # Testing part (will be used later)
    if cfg.train.downsample_test:
        list_X_test, list_y_test = [], []
        N_iter = 100
        min_num_samples = (y_test == 'Normal').sum()
        sampling_strategy = {
                'LumA': min_num_samples,
                'LumB': min_num_samples,
                'Basal': min_num_samples,
                'Her2': min_num_samples,
                'Normal': min_num_samples
            }
        cb = ClassBalance(X=X_test, y=y_test)
        normal_set, first_normal_set = set(), set()
        for i in range(N_iter):  
            uniform_test = cb.resampling(sampling_strategy)
            X_test = uniform_test.drop(columns='expert_PAM50_subtype', inplace=False)
            y_test = uniform_test.expert_PAM50_subtype

            # Since it is just stacked - sort by index
            # idxs = np.argsort(y_test.index)
            # y_test = y_test.iloc[idxs]
            # X_test = X_test.iloc[idxs, :]

            # Check if normal class is always selected in the same way
            idx_list = uniform_test.expert_PAM50_subtype.index
            idx_normal = np.where(np.array(uniform_test.expert_PAM50_subtype)== 'Normal')[0]
            normal_samples = idx_list[idx_normal]
            normal_set.update(set(normal_samples))

            if i==0:
                first_normal_set.update(set(normal_samples))

            list_y_test.append(y_test)
            list_X_test.append(X_test)

        # If the first and the last sets are the same than the samplig is done right!
        print('Sets are equal: ', first_normal_set==normal_set)

        counts_test_after = y_test.value_counts()

        # Plot class balance difference 
        plot_before_after_counts(counts_test_before, counts_test_after)

    # Encode the class labels
    LB = LabelEncoder() 
    y_train = pd.Series(LB.fit_transform(y_train), index=y_train.index)
    if cfg.train.downsample_test:
        for i, yt in enumerate(list_y_test):
            list_y_test[i] = pd.Series(LB.transform(yt), index=yt.index)
    else:
        y_test = LB.transform(y_test)

    # Data standardization | normalization
    X_train = X_train.divide(X_train.sum(axis=1), axis=0) * 1e6
    if cfg.train.downsample_test:
        for i, X_test in enumerate(list_X_test):
            list_X_test[i] = X_test.divide(X_test.sum(axis=1), axis=0) * 1e6
    else:
        X_test = X_test.divide(X_test.sum(axis=1), axis=0) * 1e6
    scaler = FunctionTransformer(log_transform)
    X_train_scaled = scaler.fit_transform(X_train)
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
    if cfg.train.downsample_test:
        for i, X_test in enumerate(list_X_test):
            list_X_test[i] = pd.DataFrame(scaler.transform(X_test), columns=X.columns)
    else:
        X_test_scaled = scaler.transform(X_test)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)
        
    # --------- Feature selection ------------

    if cfg.train.type_feat_selection == 'univariate':

        # Use F-test to select defined number of features (500 or 1000)
        best_feat_model = SelectKBest(score_func=f_classif, k=cfg.train.num_feat) # k needs to be defined
        best_feat_model.fit(X_train_scaled, y_train)

        selected_feat = list(X_train_scaled.columns[best_feat_model.get_support()])
    
    elif cfg.train.type_feat_selection == 'hybrid':
        if cfg.train.brca_cris == 'CRIS':
            hybrid_feat_selection_path = os.path.join('/Users/ivamilojkovic/Breast-Cancer-Analysis/data/cris/new2_without_corr_removed_feat_select_gt_40_perc_occur.pkl') 
        else:
            hybrid_feat_selection_path = os.path.join('/Users/ivamilojkovic/Breast-Cancer-Analysis/data/brca/without_corr_removed_feat_select_gt_50_perc_occur.pkl') # could be ['hybrid_features_800.pickle'...]
            
        if not os.path.exists(hybrid_feat_selection_path):
            pass
            # # Apply filtering method 
            # filter_model = SelectKBest(score_func=f_classif, k=cfg.train.num_feat) # k needs to be defined
            # filter_model.fit(X_train_scaled, y_train)
            # mask = filter_model.get_support()
            # selected_feat = X.columns[mask]
            # X_selected = X_train_scaled[selected_feat]

            # # Now apply backward feature elimination
            # model = LogisticRegression(penalty='l2')
            # selector = RFECV(estimator=model, scoring='accuracy', step=1, cv=3, verbose=3, n_jobs=1)
            # selector.fit(X_selected, y_train)
            # print('Number of fetures selected: ', selector.n_features_)
            # selected_feat = X_selected.columns[selector.get_support()]
            # experiment_params['n_features_selected'] = selector.n_features_

            # with open(hybrid_feat_selection_path, 'wb') as file:
            #     pickle.dump(selected_feat, file)
        
        # Load selected features
        with open(hybrid_feat_selection_path, 'rb') as file:
            selected_feat = pickle.load(file)

    # Check how many features from initial preprocessing overlap after main selection
    overlapped_feat = set(selected_feat).intersection(set(feat_to_keep))
    print('Number of features overlapping: {}'.format(len(overlapped_feat)))
    print('Number of features selected: ', len(selected_feat))

    # Extract the data frames with only selected features
    X_train_scaled_selected = X_train_scaled[selected_feat]
    if cfg.train.downsample_test:
        for i, X_test in enumerate(list_X_test):
            list_X_test[i] = X_test[list(selected_feat)]
    else:
        X_test_scaled_selected = X_test_scaled[list(selected_feat)]

    # MODEL TYPE = {
    #   Logistic Regression
    #   KNN
    #   Decision Tree
    #   SVC
    #   Random Forest
    #   XGBoost
    #   LightGBM
    #   AdaBoost
    #   }

    # MODEL_TYPES = ['Logistic Regression', 'KNN', 'Decision Tree', 
    #                'Random Forest', 'SVC', 'XGBoost',
    #                'LightGBM', 'AdaBoost']
    MODEL_TYPES = ['Logistic Regression', 'SVC', 'Random Forest', 'XGBoost']
    # MODEL_TYPES = ['KNN']

    for MODEL_TYPE in MODEL_TYPES:

        print('----------------------------------------------------------')
        print('------------------- ' + MODEL_TYPE + ' -------------------')
        print('----------------------------------------------------------')

        mlflow.start_run()

        # ---------- Track parameters --------------
        mlflow.log_param("test_size", cfg.train.test_size)
        mlflow.log_param("random_state", cfg.train.random_state)
        mlflow.log_param("grid_search", cfg.train.optim)
        mlflow.log_param("num_folds", cfg.train.num_folds)
        mlflow.log_param("num_feat_select", cfg.train.num_feat)
        mlflow.log_param("type_feat_select", cfg.train.type_feat_selection)
        mlflow.log_param("type_solution_CI", cfg.train.type_class_imbalance)
        mlflow.log_param("grid_scoring", cfg.train.grid_scoring)

        # Get the current date and time (set experiment name)
        now = datetime.datetime.now()
        experiment_name =  'run_' + now.strftime("%d-%m-%Y_%H:%M:%S")
        experiment_params = {
            'solve_ibm': cfg.train.solve_class_imbalance,
            'ci_type': cfg.train.type_class_imbalance,
            'cross_validation': cfg.train.cross_val,
            'multi_label_classification': cfg.train.use_multilabel_dataset,
            'n_folds': cfg.train.num_folds,
            'n_features_to_select': cfg.train.num_feat,
            'test_size': cfg.train.test_size,
            'feature_selection_method': cfg.train.type_feat_selection,
            'optimized': cfg.train.optim,
            'threshold_cut_LumA': cfg.train.thresh_lumA
        }

        experiment_params['model_type'] = MODEL_TYPE
        mlflow.log_param("model_type", MODEL_TYPE)

        if cfg.train.type_class_imbalance == 'case1':
            if cfg.train.thresh_lumA == 200:
                case_name = cfg.train.type_class_imbalance + 'a'
            elif cfg.train.thresh_lumA == 225:
                case_name = cfg.train.type_class_imbalance + 'b'
            elif cfg.train.thresh_lumA == 250:
                case_name = cfg.train.type_class_imbalance + 'c'
            elif cfg.train.thresh_lumA == 175:
                case_name = cfg.train.type_class_imbalance + 'd'
        else:
            case_name = cfg.train.type_class_imbalance
        

        #####################################################################
        ############################  MAIN PART #############################
        #####################################################################

        # Define a model
        if MODEL_TYPE == 'MLP Classifier': 
            classifier = MLPClassifier(random_state=cfg.train.random_state, max_iter=200, early_stopping=True)
        
        elif MODEL_TYPE == 'Logistic Regression':
            if cfg.train.optim:
                classifier = LogisticRegression(random_state=cfg.train.random_state)
            else:
                if cfg.train.brca_cris == 'BRCA':
                    path = '/Users/ivamilojkovic/Breast-Cancer-Analysis/final_results/BRCA/single-label/SUBTYPE_PAM50/feat_select_hybrid'
                    exp_name = 'bestmodel_LogisticRegression_run_28-08-2023_13:08:54.pkl'
                elif cfg.train.brca_cris == 'CRIS':
                    path = '/Users/ivamilojkovic/Breast-Cancer-Analysis/final_results/CRIS/single-label/01-14-52'
                    exp_name = 'bestmodel_LogisticRegression_run_01-09-2023_01:15:05.pkl'
            with open(os.path.join(path, exp_name), 'rb') as file:
                classifier = pickle.load(file)
            
        elif MODEL_TYPE == 'KNN':
            classifier = KNeighborsClassifier()

        elif MODEL_TYPE == 'Decision Tree':
            classifier = DecisionTreeClassifier(random_state=cfg.train.random_state)

        elif MODEL_TYPE == 'SVC':
            if cfg.train.optim:
                classifier = SVC(random_state=cfg.train.random_state)
            else:
                if cfg.train.brca_cris == 'BRCA':
                    path = '/Users/ivamilojkovic/Breast-Cancer-Analysis/final_results/BRCA/single-label/SUBTYPE_PAM50/feat_select_hybrid'
                    exp_name = 'bestmodel_SVC_run_28-08-2023_14:00:35.pkl'
                elif cfg.train.brca_cris == 'CRIS':
                    path = '/Users/ivamilojkovic/Breast-Cancer-Analysis/final_results/CRIS/single-label/01-14-52'
                    exp_name = 'bestmodel_SVC_run_01-09-2023_01:59:53.pkl'
            with open(os.path.join(path, exp_name), 'rb') as file:
                classifier = pickle.load(file)

        elif MODEL_TYPE == 'Random Forest':
            if cfg.train.optim:
                classifier = RandomForestClassifier(random_state=cfg.train.random_state)
            else:
                if cfg.train.brca_cris == 'BRCA':
                    path = '/Users/ivamilojkovic/Breast-Cancer-Analysis/final_results/BRCA/single-label/SUBTYPE_PAM50/feat_select_hybrid'
                    exp_name = 'bestmodel_RandomForest_run_28-08-2023_13:44:47.pkl'
                elif cfg.train.brca_cris == 'CRIS':
                    path = '/Users/ivamilojkovic/Breast-Cancer-Analysis/final_results/CRIS/single-label/01-14-52'
                    exp_name = 'bestmodel_RandomForest_run_01-09-2023_01:42:48.pkl'
            with open(os.path.join(path, exp_name), 'rb') as file:
                classifier = pickle.load(file)
                
        elif MODEL_TYPE == 'XGBoost':
            if cfg.train.optim:
                classifier = xgb.XGBClassifier(random_state=cfg.train.random_state)
            else:
                if cfg.train.brca_cris == 'BRCA':
                        path = '/Users/ivamilojkovic/Breast-Cancer-Analysis/final_results/BRCA/single-label/SUBTYPE_PAM50/feat_select_hybrid'
                        exp_name = 'bestmodel_XGBoost_run_28-08-2023_05:09:55.pkl'
                elif cfg.train.brca_cris == 'CRIS':
                    path = '/Users/ivamilojkovic/Breast-Cancer-Analysis/final_results/CRIS/single-label/01-14-52'
                    exp_name = 'bestmodel_XGBoost_run_01-09-2023_03:49:46.pkl'
                with open(os.path.join(path, exp_name), 'rb') as file:
                    classifier = pickle.load(file)

        elif MODEL_TYPE == 'LightGBM':
            classifier = lgb.LGBMClassifier(random_state=cfg.train.random_state)
        elif MODEL_TYPE == 'AdaBoost':
            classifier = AdaBoostClassifier(random_state=cfg.train.random_state)
        else:
            print('There no such model to be choosen! Try again!')
            exit()

        # Save data for training and testing
        with open(os.path.join(cfg.paths.artefacts, experiment_name + '.pickle'), 'wb') as file:
            pickle.dump([X_train_scaled_selected, LB.inverse_transform(y_train),
                         X_test_scaled_selected, LB.inverse_transform(y_test)], file)
            
        plt.close('all')

        #################################### FIND OPTIMAL MODEL ####################################
        if cfg.train.optim:
            gs = GridSearchCV(
                estimator=classifier, 
                param_grid=MODEL_PARAMS[MODEL_TYPE], 
                scoring=cfg.train.grid_scoring,
                cv=StratifiedKFold(n_splits=cfg.train.num_folds, shuffle=True, random_state=123), 
                n_jobs=1, verbose=2, 
                return_train_score=True,
                refit=True)
            gs.fit(X_train_scaled_selected.values, y_train)

            # Get the optimal model and its parameters
            model = gs.best_estimator_
            best_params = gs.best_params_
            experiment_params['model_params'] = best_params

            # Get cross-validation scores (averagre and std value of train/validation score)
            best_idx = gs.best_index_
            avg_val_scores, std_val_scores = gs.cv_results_['mean_test_score'][best_idx], gs.cv_results_['std_test_score'][best_idx]
            avg_train_scores, std_train_scores = gs.cv_results_['mean_train_score'][best_idx], gs.cv_results_['std_train_score'][best_idx]

            # Access the results for each fold
            train_scores = {
                'MCC': [],
                'weighted_precision': [],
                'weighted_recall': [],
                'weighted_f1': [],
                'macro_precision': [],
                'macro_recall': [],
                'macro_f1': []
            }
            val_scores = {
                'MCC': [],
                'weighted_precision': [],
                'weighted_recall': [],
                'weighted_f1': [],
                'macro_precision': [],
                'macro_recall': [],
                'macro_f1': []
            }
            
            cv_results = gs.cv_results_

            # Loop through each fold and compute the desired metrics
            for train_indices, val_indices in gs.cv.split(X_train_scaled_selected, y_train):
                
                # Predict on the train and validation sets using the best model
                y_train_pred = model.predict(X_train_scaled_selected.iloc[train_indices, :])
                y_train_true = y_train.iloc[train_indices]
                y_val_pred = model.predict(X_train_scaled_selected.iloc[val_indices, :])
                y_val_true = y_train.iloc[val_indices]

                # Calculate the desired metrics for the current fold - train set
                train_scores['weighted_precision'].append(precision_score(y_train_true, y_train_pred, average='weighted'))
                train_scores['weighted_recall'].append(recall_score(y_train_true, y_train_pred, average='weighted'))
                train_scores['weighted_f1'].append(f1_score(y_train_true, y_train_pred, average='weighted'))
                train_scores['macro_precision'].append(precision_score(y_train_true, y_train_pred, average='macro'))
                train_scores['macro_recall'].append(recall_score(y_train_true, y_train_pred, average='macro'))
                train_scores['macro_f1'].append(f1_score(y_train_true, y_train_pred, average='macro'))
                train_scores['MCC'].append(matthews_corrcoef(y_train_true, y_train_pred))

                # Calculate the desired metrics for the current fold - validation set
                val_scores['weighted_precision'].append(precision_score(y_val_true, y_val_pred, average='weighted'))
                val_scores['weighted_recall'].append(recall_score(y_val_true, y_val_pred, average='weighted'))
                val_scores['weighted_f1'].append(f1_score(y_val_true, y_val_pred, average='weighted'))
                val_scores['macro_precision'].append(precision_score(y_val_true, y_val_pred, average='macro'))
                val_scores['macro_recall'].append(recall_score(y_val_true, y_val_pred, average='macro'))
                val_scores['macro_f1'].append(f1_score(y_val_true, y_val_pred, average='macro'))
                val_scores['MCC'].append(matthews_corrcoef(y_val_true, y_val_pred))

            # Calculate the mean for each list in the dictionary
            mean_train_metrics, std_train_metrics = {}, {}
            for key, value in train_scores.items():
                mean_train_metrics[key] = np.mean(value)
                std_train_metrics[key] = np.std(value)
            mean_val_metrics, std_val_metrics = {}, {}
            for key, value in val_scores.items():
                mean_val_metrics[key] = np.mean(value)
                std_val_metrics[key] = np.std(value)
            scores = {
                'Train': {
                    'average': mean_train_metrics,
                    'std': std_train_metrics,
                },
                'Validation': {
                    'average': mean_val_metrics,
                    'std': std_val_metrics,
                }
            }

            # Only the accuracy here..
            experiment_params['cv_avg_val_scores'] = avg_val_scores
            experiment_params['cv_avg_train_scores'] = avg_train_scores
            experiment_params['cv_std_val_scores'] = std_val_scores
            experiment_params['cv_std_train_scores'] = std_train_scores

            print("Average training score:", avg_train_scores)
            print("Training score standard deviation:", std_train_scores)
            print("Average validation score:", avg_val_scores)
            print("Validation score standard deviation:", std_val_scores)

            # Save experiment params as .json file
            exp_params_filename = 'exp_params_' + MODEL_TYPE.replace(' ', '') + '_' + experiment_name + '.json'
            json_file = json.dumps(experiment_params, cls=NumpyEncoder)
            with open(exp_params_filename, "w") as file:
                json.dump(json_file, file)

            # Save the best model
            best_model_filename = 'bestmodel_' + MODEL_TYPE.replace(' ', '') + '_' + experiment_name + '.pickle'
            with open(os.path.join(cfg.paths.single_label_model, best_model_filename), 'wb') as file:
                pickle.dump(model, file)

            # Save CV scores as dictionary (validation/train and mean and std values)
            result_sl_path = os.path.join(
                cfg.paths.result, 'single-label', 
                'best_model_cv_scores_' + MODEL_TYPE.replace(' ', '') + '_' + experiment_name + '.json'
            ) 
            with open(result_sl_path, "w") as outfile:
                json.dump(scores, outfile)

            # ---------- Track model ---------------
            mlflow.sklearn.log_model(model, "best_model")
            
        else:
            start = time.time()
            model = classifier.fit(X_train_scaled_selected, y_train)
            stop = time.time()

            experiment_params['model_params'] = MODEL_PARAMS[MODEL_TYPE]
        
        # --------------- Compute predictions ------------------
        pred_train = model.predict(X_train_scaled_selected.values)
        if cfg.train.downsample_test:
            prec_s, rec_s, f1_s = [], [], []
            for i in range(N_iter):
                pred = model.predict(list_X_test[i])
                test_metrics = cmp_metrics(pred, list_y_test[i])
                prec_s.append(test_metrics['Precision per class'])
                rec_s.append(test_metrics['Recall per class'])
                f1_s.append(test_metrics['F1 score per class'])

            prec_stacked = np.stack(prec_s)
            rec_stacked = np.stack(rec_s)
            f1_stacked = np.stack(f1_s)

            mean_prec = prec_stacked.mean(axis=0)
            mean_rec =rec_stacked.mean(axis=0)
            mean_f1 = f1_stacked.mean(axis=0)

            std_prec = prec_stacked.std(axis=0)
            std_rec = rec_stacked.std(axis=0)
            std_f1 = f1_stacked.std(axis=0)

            ax = plt.figure()
            df_lr = pd.DataFrame({'Precision': mean_prec,
                            'Recall': mean_rec,
                            'F1 score': mean_f1}, 
                            index= LB.classes_)
            if not cfg.train.solve_class_imbalance:
                ax = df_lr.plot(kind='bar', rot=30, title='Scores for ' + 'case-0' +' best model',
                                yerr=[std_prec, std_rec, std_f1])
            else:
                ax = df_lr.plot(kind='bar', rot=30, title='Scores for ' + case_name +' best model',
                                yerr=[std_prec, std_rec, std_f1])
            plt.legend(loc='lower right')
            plt.savefig('Confuson matrix ' + MODEL_TYPE + '.png')
            print()

        else:
            # Compute the predictions 
            pred = model.predict(X_test_scaled_selected.values)

            # ---------------- Compute metrics ---------------------
            test_metrics = cmp_metrics(pred, y_test)
            train_metrics = cmp_metrics(pred_train, y_train)
            experiment_params['Test results'] = test_metrics
            experiment_params['Train results'] = train_metrics

            # Save experiment params as .json file
            exp_params_filename = 'exp_params_' + MODEL_TYPE.replace(' ', '') + '_' + experiment_name + '.json'
            json_file = json.dumps(experiment_params, cls=NumpyEncoder)
            with open(exp_params_filename, "w") as file:
                json.dump(json_file, file)

            ax = plt.figure(figsize=(16, 8))
            df_lr = pd.DataFrame({'Precision': test_metrics['Precision per class'],
                            'Recall': test_metrics['Recall per class'],
                            'F1 score': test_metrics['F1 score per class']}, 
                            index=LB.classes_)
            ax = df_lr.plot(kind='bar', rot=30, title=MODEL_TYPE, width=0.8)
            # Add values above each bar
            for p in ax.patches:
                ax.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha='center', va='center', xytext=(0, 5), textcoords='offset points', fontsize=10)
            plt.legend(loc='lower right')
            plt.gcf().set_size_inches(12, 6)
            plt.savefig('Barplot ' + MODEL_TYPE + '.png')

            # ----------------- Track metrics -------------------
            mlflow.log_metric("accuracy_weighted_average", test_metrics['Accuracy weighted'])
            mlflow.log_metric("precision_weighted_average", test_metrics['Precision weighted'])
            mlflow.log_metric("recall_weighted_average", test_metrics['Recall weighted'])
            mlflow.log_metric("f1_score_weighted_average", test_metrics['F1 score weighted'])

            mlflow.log_metric("precision_macro_average", test_metrics['Precision macro'])
            mlflow.log_metric("recall_macro_average", test_metrics['Recall macro'])
            mlflow.log_metric("f1_score_macro_average", test_metrics['F1 score macro'])

            mlflow.log_metric("mcc", test_metrics['MCC'])
            #mlflow.log_metric("ROC AUC", test_metrics['ROC_AUC'])

            # Plot confusion matrix
            conf_mat = confusion_matrix(y_test, pred) 
            # conf_mat_norm = confusion_matrix(y_test, pred, normalize='true')
            conf_mat_percentage = conf_mat / conf_mat.sum(axis=1).reshape(-1,1)

            class_cnts = ['{0:0.0f}'.format(value) for value in conf_mat.flatten()]
            class_percentages = ['{0:.2%}'.format(value) for value in conf_mat_percentage.flatten()]   
            # class_cnt_norm = ['{0:.2f}'.format(value) for value in conf_mat_norm.flatten()]

            labels = [f'{v1}\n{v2}' for \
                        v1, v2 in zip(class_cnts, class_percentages)] # class_per
            labels = np.asarray(labels).reshape(5,5)

            fig = plt.figure()
            df = pd.DataFrame(conf_mat_percentage, index = [i for i in LB.classes_], columns = [i for i in LB.classes_])
            #sns.heatmap(df.div(df.values.sum()), annot=True)
            sns.heatmap(df, annot=labels, fmt='', cmap='Greens')
            if not cfg.train.solve_class_imbalance:
                plt.title(MODEL_TYPE)
            else:
                plt.title('Confusion matrix for ' + case_name + ' best model')

            plt.ylabel('True class')
            plt.xlabel('Predicted class')
            plt.savefig(MODEL_TYPE + '.png')
            plt.show(block=False)

            # Save the experiment properties as .pickle file
            if not cfg.train.downsample_test and cfg.train.optim:
                if not cfg.train.solve_class_imbalance:
                    with open(os.path.join(cfg.paths.experiment, 'case_0', experiment_name + '.pickle'), 'wb') as file:
                        pickle.dump(experiment_params, file)
                else:
                    if cfg.train.type_class_imbalance=='case1':
                        with open(os.path.join(cfg.paths.experiment, 'case_1', experiment_name + '.pickle'), 'wb') as file:
                            pickle.dump(experiment_params, file)
                    if cfg.train.type_class_imbalance=='case2':
                        with open(os.path.join(cfg.paths.experiment, 'case_2', experiment_name + '.pickle'), 'wb') as file:
                            pickle.dump(experiment_params, file)
                    if cfg.train.type_class_imbalance=='case3':
                        with open(os.path.join(cfg.paths.experiment, 'case_3', experiment_name + '.pickle'), 'wb') as file:
                            pickle.dump(experiment_params, file)
                    if cfg.train.type_class_imbalance=='case4':
                        with open(os.path.join(cfg.paths.experiment, 'case_4', experiment_name + '.pickle'), 'wb') as file:
                            pickle.dump(experiment_params, file)

        time.sleep(2)
        mlflow.end_run()

if __name__=='__main__':
    main()
    