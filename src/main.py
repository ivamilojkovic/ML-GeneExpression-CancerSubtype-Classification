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
#from mlxtend.feature_selection import SequentialFeatureSelector
import os, pickle, datetime, time

import pickle as pkl
from data_preprocessing import ClassBalance, remove_extreme
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from utils import cmp_metrics, m_cut_strategy_class_assignment
import xgboost as xgb
import lightgbm as lgb
from utils import log_transform, plot_before_after_counts, plot_pca
from sklearn.decomposition import PCA
from config_model import *

plt.style.use('ggplot')
plt.rcParams.update({'font.size': 12})
sns.set_theme()

from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)

import mlflow
import hydra
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

    # Loading the dataset
    with open(cfg.paths.dataset, 'rb') as file:
        dataset = pkl.load(file) 

    X = dataset.drop(columns=['expert_PAM50_subtype', 'tcga_id', \
                            'sample_id', 'cancer_type'], inplace=False)
    y = dataset.expert_PAM50_subtype

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

    # PCA
    if cfg.train.type_class_imbalance=='case3' or cfg.train.type_class_imbalance=='case4':
        pca_2 = PCA(n_components=2)
        df_pca_2 = pd.DataFrame(pca_2.fit_transform(X_train_scaled), columns=['PCA_1', 'PCA_2'])

        pca_3 = PCA(n_components=3)
        df_pca_3 = pd.DataFrame(pca_3.fit_transform(X_train_scaled), columns=['PCA_1', 'PCA_2', 'PCA_3'])

        unique = pd.concat([X_train_scaled, scaler.transform(new_samples)]).drop_duplicates(keep=False)
        new_train_samples = pd.concat([X_train_scaled, unique]).drop_duplicates(keep=False)

        plot_pca(df_pca_2, LB.inverse_transform(y_train), new_train_samples.index)
        plot_pca(df_pca_3, LB.inverse_transform(y_train), new_train_samples.index, dim=3)
        
    # Feature selection
    if cfg.train.type_feat_selection == 'univariate':
        best_feat_model = SelectKBest(score_func=f_classif, k=cfg.train.num_feat) # k needs to be defined
        best_feat_model.fit(X_train_scaled, y_train)
        df_scores = pd.DataFrame(best_feat_model.scores_)
        df_feats = pd.DataFrame(X.columns)

        featureScores = pd.concat([df_feats, df_scores],axis=1)
        featureScores.columns = ['Feature', 'Score'] 
        
        plt.figure()
        featureScores.nlargest(50, 'Score').plot(kind='barh')
        plt.title(cfg.train.type_feat_selection + ' feature selection')
        print(featureScores.nlargest(10, 'Score'))

        selected_feat = featureScores.sort_values(by='Score')[-cfg.train.num_feat:]['Feature']
    
    elif cfg.train.type_feat_selection == 'hybrid':
        best_feat_model = SelectKBest(score_func=f_classif, k=cfg.train.num_feat) # k needs to be defined
        best_feat_model.fit(X_train_scaled, y_train)
        mask = best_feat_model.get_support()
        selected_feat = X.columns[mask]
        X_selected = X_train_scaled[selected_feat]

        # Now apply backward feature elimination
        model = DecisionTreeClassifier(criterion='log_loss', random_state=42)
        selector = SequentialFeatureSelector(model, k_features=10, 
                                            forward=False, verbose=2,
                                            floating=False, cv=3, 
                                            scoring='f1_weighted', n_jobs=-1)
        selector.fit(X_selected, y_train)
        selected_feat = list(selector.k_feature_names_)

    elif cfg.train.type_feat_selection == 'filter':
        with open('high_corr_feat.pkl', 'rb') as file:
            selected_feat = pickle.load(file)

    elif cfg.train.type_feat_selection == 'recursive':
        rfe_selector = RFE(estimator=LogisticRegression(), 
                        n_features_to_select=cfg.train.num_feat, step=10, verbose=5)
        rfe_selector.fit(X_train_scaled, y_train)
        rfe_support = rfe_selector.get_support()

        selected_feat = X.loc[:,rfe_support].columns.tolist()
        print(str(len(selected_feat)), 'selected features')

    elif cfg.train.type_feat_selection == 'Wrapper':
        best_feat_model  = ExtraTreesClassifier(n_estimators=10)
        best_feat_model.fit(X_train_scaled, y_train)
        
        plt.figure()
        feat_importances = pd.Series(best_feat_model.feature_importances_, index=X.columns)
        feat_importances.nlargest(50).plot(kind='barh')

        plt.figure()
        plt.plot(sorted(best_feat_model.feature_importances_))
        plt.title(cfg.train.type_feat_selection + ' feature selection')

        selected_feat = dict(feat_importances.sort_values()[-cfg.train.num_feat:]).keys()

    elif cfg.train.type_feat_selection == 'embedded':
        model = LogisticRegression(C=0.1)
        selector = SelectFromModel(model, threshold=0.025) 
        selector.fit_transform(X_train_scaled, y_train)
        selected_feat = X_train_scaled.columns[selector.get_support()]

    elif cfg.train.type_feat_selection == 'wrapper':
        if os.path.exists('feat50.pkl'):
            with open('feat50.pkl', 'rb') as f:
                selected_feat = pickle.load(f)
        else:
            # Forward Feature Selection
            model = DecisionTreeClassifier(criterion='log_loss', random_state=42)
            selector = SequentialFeatureSelector(model, k_features=50, 
                                                forward=True, verbose=2,
                                                floating=False, cv=3, 
                                                scoring='f1_weighted', n_jobs=-1)
            selector.fit(X_train_scaled, y_train)
            selected_feat = list(selector.k_feature_names_)
    
    elif cfg.train.type_feat_selection == 'wrapper_reg':
        model = Lasso(alpha=0.0001, random_state=42)
        selector = SelectFromModel(model, threshold=0.001) 
        selector.fit_transform(X_train_scaled, y_train)
        selector.fit(X_train_scaled, y_train)
        selected_feat = X_train_scaled.columns[selector.get_support()]

    # Check how many features from initial preprocessing overlap after main selection
    overlapped_feat = set(selected_feat).intersection(set(feat_to_keep))
    print('Number of features overlapping: {}'.format(len(overlapped_feat)))

    # Extract the data frames with only selected features
    X_train_scaled_selected = X_train_scaled[list(selected_feat)]
    if cfg.train.downsample_test:
        for i, X_test in enumerate(list_X_test):
            list_X_test[i] = X_test[list(selected_feat)]
    else:
        X_test_scaled_selected = X_test_scaled[list(selected_feat)]

    # MODEL TYPE = {
    #   MLP Classifier
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
    #                'SVC', 'Random Forest', 'XGBoost', 
    #                'LightGBM', 'AdaBoost']

    MODEL_TYPES = ['XGBoost']

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
            'multi_label_classification': False,
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
                if cfg.train.type_class_imbalance=='case1': 
                    if cfg.train.thresh_lumA == 200:
                        exp_name = 'run_09-05-2023_02:29:40'
                    elif cfg.train.thresh_lumA == 225:
                        exp_name = 'run_11-05-2023_01:10:33'
                    elif cfg.train.thresh_lumA == 250:
                        exp_name = 'run_11-05-2023_19:37:56'
                    elif cfg.train.thresh_lumA == 175:
                        exp_name = 'run_09-05-2023_20:51:26'

                if cfg.train.type_class_imbalance=='case2':
                    exp_name = 'run_09-05-2023_20:51:26'
                elif cfg.train.type_class_imbalance=='case3':
                    exp_name = 'run_10-05-2023_00:18:29'
                elif cfg.train.type_class_imbalance=='case4':
                    exp_name = 'run_10-05-2023_12:10:55'

                with open(os.path.join(cfg.paths.model, 'bestmodel_' + exp_name + '.pkl'), 'rb') as file:
                    classifier = pickle.load(file)
            
        elif MODEL_TYPE == 'KNN':
            classifier = KNeighborsClassifier()

        elif MODEL_TYPE == 'Decision Tree':
            classifier = DecisionTreeClassifier(random_state=cfg.train.random_state)

        elif MODEL_TYPE == 'SVC':
            if cfg.train.optim:
                classifier = SVC(random_state=cfg.train.random_state)
            else:
                exp_name = 'run_09-05-2023_02:29:40'
                with open(os.path.join(cfg.paths.model, 'bestmodel_' + exp_name + '.pkl'), 'rb') as file:
                    classifier = pickle.load(file)

        elif MODEL_TYPE == 'Random Forest':
            if cfg.train.solve_class_imbalance:
                if cfg.train.type_class_imbalance == 'case3':
                    classifier = RandomForestClassifier(
                        random_state=cfg.train.random_state,
                        criterion='gini', min_samples_leaf=3,
                        min_samples_split=2, n_estimators=50)
                elif cfg.train.type_class_imbalance == 'case1':
                    exp_name = 'run_08-05-2023_03:06:27'
                    with open(os.path.join(cfg.paths.model, 'bestmodel_' + exp_name + '.pkl'), 'rb') as file:
                        classifier = pickle.load(file)
                else:
                    classifier = RandomForestClassifier(random_state=cfg.train.random_state)
            else:
                classifier = RandomForestClassifier(random_state=cfg.train.random_state)

        elif MODEL_TYPE == 'XGBoost':
            if cfg.train.optim:
                classifier = xgb.XGBClassifier(random_state=cfg.train.random_state)
            else:
                exp_name = 'run_08-05-2023_10:32:03'
                with open(os.path.join(cfg.paths.model, 'bestmodel_' + exp_name + '.pkl'), 'rb') as file:
                    classifier = pickle.load(file)
        elif MODEL_TYPE == 'LightGBM':
            classifier = lgb.LGBMClassifier(random_state=cfg.train.random_state)
        elif MODEL_TYPE == 'AdaBoost':
            classifier = AdaBoostClassifier(random_state=cfg.train.random_state)
        else:
            print('There no such model to be choosen! Try again!')
            exit()

        # ---------------- Cross-validation ---------------
        if cfg.train.cross_val:
            scores = cross_val_score(classifier, X_train_scaled, y_train,
                                    scoring='neg_log_loss', 
                                    cv=cfg.train.num_folds, verbose=5)
            

        # Save data for training and testing
        with open(os.path.join(cfg.paths.artefacts, experiment_name + '.pkl'), 'wb') as file:
            pickle.dump([X_train_scaled_selected, LB.inverse_transform(y_train),
                         X_test_scaled_selected, LB.inverse_transform(y_test)], file)
            
        # ----------------- TRAIN & TEST ------------------
        if cfg.train.optim:
            # Define Grid Search
            gs = GridSearchCV(classifier, param_grid=MODEL_PARAMS[MODEL_TYPE], 
                            scoring=cfg.train.grid_scoring, cv=cfg.train.num_folds, 
                            verbose=2, refit=True, n_jobs=1, return_train_score=True)
            start = time.time()
            gs.fit(X_train_scaled_selected.values, y_train)
            stop = time.time()

            model = gs.best_estimator_
            best_params = gs.best_params_
            experiment_params['model_params'] = best_params
            with open(os.path.join(cfg.paths.model, 'bestmodel_' + experiment_name + '.pkl'), 'wb') as file:
                pickle.dump(model, file)

            # ---------- Track model ---------------
            mlflow.sklearn.log_model(model, "best_model")
            
        else:
            start = time.time()
            model = classifier.fit(X_train_scaled_selected, y_train)
            stop = time.time()

            experiment_params['model_params'] = MODEL_PARAMS[MODEL_TYPE]

        experiment_params['training_time'] =  stop-start
        
        # --------------- Compute predictions ------------------
        pred_train = model.predict(X_train_scaled_selected.values)
        pred_prob = model.predict_proba(X_train_scaled_selected.values)
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
                            index=['LumA', 'LumB', 'Basal', 'Her2', 'Normal'])
            if not cfg.train.solve_class_imbalance:
                ax = df_lr.plot(kind='bar', rot=30, title='Scores for ' + 'case-0' +' best model',
                                yerr=[std_prec, std_rec, std_f1])
            else:
                ax = df_lr.plot(kind='bar', rot=30, title='Scores for ' + case_name +' best model',
                                yerr=[std_prec, std_rec, std_f1])
            plt.legend(loc='lower right')
            print()

        else:
            pred = model.predict(X_test_scaled_selected.values)
            prob_pred = model.predict_proba(X_test_scaled_selected.values)

            # ---------------- M-cut strategy ----------------------
            m_cut_labels = m_cut_strategy_class_assignment(prob_pred, non_neg_values=True)
        
            # ---------------- Compute metrics ---------------------
            test_metrics = cmp_metrics(pred, y_test)
            train_metrics = cmp_metrics(pred_train, y_train)
            experiment_params['Test results'] = test_metrics
            experiment_params['Train results'] = train_metrics

            ax = plt.figure()
            df_lr = pd.DataFrame({'Precision': test_metrics['Precision per class'],
                            'Recall': test_metrics['Recall per class'],
                            'F1 score': test_metrics['F1 score per class']}, 
                            index=['LumA', 'LumB', 'Basal', 'Her2', 'Normal'])
            ax = df_lr.plot(kind='bar', rot=30, title='Scores for ' + case_name +' best model')
            plt.legend(loc='lower right')

            # ----------------- Track metrics -------------------
            mlflow.log_metric("accuracy_weighted_average", test_metrics['Accuracy weighted'])
            mlflow.log_metric("precision_weighted_average", test_metrics['Precision weighted'])
            mlflow.log_metric("recall_weighted_average", test_metrics['Recall weighted'])
            mlflow.log_metric("f1_score_weighted_average", test_metrics['F1 score weighted'])

            mlflow.log_metric("precision_macro_average", test_metrics['Precision unweighted'])
            mlflow.log_metric("recall_macro_average", test_metrics['Recall unweighted'])
            mlflow.log_metric("f1_score_macro_average", test_metrics['F1 score unweighted'])

            mlflow.log_metric("mcc", test_metrics['MCC'])
            #mlflow.log_metric("ROC AUC", test_metrics['ROC_AUC'])

            # Plot confusion matrix
            conf_mat = confusion_matrix(y_test, pred) 
            conf_mat_percentage = conf_mat / conf_mat.sum(axis=1)  

            class_cnts = ['{0:0.0f}'.format(value) for value in conf_mat.flatten()]

            class_percentages = ['{0:.2%}'.format(value) for value in conf_mat_percentage.flatten()]    
            labels = [f'{v1}\n{v2}' for \
                      v1, v2 in zip(class_cnts, class_percentages)]
            labels = np.asarray(labels).reshape(5,5)

            fig = plt.figure()
            df = pd.DataFrame(conf_mat_percentage, index = [i for i in LB.classes_], columns = [i for i in LB.classes_])
            #sns.heatmap(df.div(df.values.sum()), annot=True)
            sns.heatmap(df, annot=labels, fmt='', cmap='Greens')
            if not cfg.train.solve_class_imbalance:
                plt.title('Confusion matrix for ' + 'case-0' +' best model')
            else:
                plt.title('Confusion matrix for ' + case_name + ' best model')
            
            plt.ylabel('True class')
            plt.xlabel('Predicted class')
            plt.show()

        # Save the experiment properties as .pkl file
        if not cfg.train.downsample_test and cfg.train.optim:
            if not cfg.train.solve_class_imbalance:
                with open(os.path.join(cfg.paths.experiment, 'case_0', experiment_name + '.pkl'), 'wb') as file:
                    pickle.dump(experiment_params, file)
            else:
                if cfg.train.type_class_imbalance=='case1':
                    with open(os.path.join(cfg.paths.experiment, 'case_1', experiment_name + '.pkl'), 'wb') as file:
                        pickle.dump(experiment_params, file)
                if cfg.train.type_class_imbalance=='case2':
                    with open(os.path.join(cfg.paths.experiment, 'case_2', experiment_name + '.pkl'), 'wb') as file:
                        pickle.dump(experiment_params, file)
                if cfg.train.type_class_imbalance=='case3':
                    with open(os.path.join(cfg.paths.experiment, 'case_3', experiment_name + '.pkl'), 'wb') as file:
                        pickle.dump(experiment_params, file)
                if cfg.train.type_class_imbalance=='case4':
                    with open(os.path.join(cfg.paths.experiment, 'case_4', experiment_name + '.pkl'), 'wb') as file:
                        pickle.dump(experiment_params, file)

        time.sleep(2)
        mlflow.end_run()

if __name__=='__main__':
    main()
    