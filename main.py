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

from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)

def main():

    EXP_PATH = 'experiments'

    # Set the parameters
    SOLVE_IMB = True # Solve class imbalance problem
    TYPE_CI = 'case1'
    CROSS_VAL = False
    TEST_SIZE = 0.3
    RANDOM_STATE = 4
    OPTIM = True
    DOWNSAMP_TEST = False

    N_folds = 10
    N_feats = 500

    # Feature selection can be: 
    # univariate, wrapper, embedded, hybrid...
    FEAT_SELECT = 'univariate'

    # Get the current date and time
    now = datetime.datetime.now()
    experiment_name =  'run_' + now.strftime("%d-%m-%Y_%H:%M:%S")
    experiment_params = {
        'solve_ibm': SOLVE_IMB,
        'ci_type': TYPE_CI,
        'cross_validation': CROSS_VAL,
        'multi_label_classification': False,
        'n_folds': N_folds,
        'n_features_to_select': N_feats,
        'test_size': TEST_SIZE,
        'feature_selection_method': FEAT_SELECT,
        'optimized': OPTIM
    }

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

    MODEL_TYPE = 'Random Forest'

    MODEL_PARAMS = {
        'MLP Classifier': {
            'hidden_layer_sizes':[(20, 10, 5), (30, 10, 10, 5), 
                                  (20, 5, 10), (20, 10, 5, 10)],
            'activation': ['logistic', 'tanh'],
            'solver': ['lbfgs', 'sgd', 'adam'], 
            'alpha': [1e-1, 1e-2, 1e-3, 1e-4],
            'batch_size': [5, 10, 15, 20],
            'learning_rate': ['constant', 'invscaling', 'adaptive'],
            'learning_rate_init': [0.01, 0.001, 0.0001]
        },
        'Logistic Regression': {
            'penalty': ['l2', None], 
            'tol': [1e-1, 1e-2, 1e-3, 1e-4],
            'solver': ['lbfgs', 'newton-cg', 'newton-cholesky', 'sag'],
            'max_iter': [100, 500, 1000],
            'C': [0.01, 0.05, 0.1, 0.5, 1, 5, 10],
        },
        'KNN': {
            'n_neighbors': [3, 4, 5, 6, 7, 8],  
            'weights': ['uniform'], 
            'leaf_size': [10, 20, 30, 40], 
            'metric': ['minkowski']
        },
        'Decision Tree': {
            'criterion': ['gini', 'entropy', 'log_loss'], 
            'max_depth': [5, 10, 20], 
            'min_samples_split': [2, 3, 4, 5],
            'min_samples_leaf': [2, 3, 4, 5], 
            'splitter': ['best', 'random'],
            'class_weight': [None, 'balanced'],
        },
        'SVC': {
            'C': [0.01, 0.05, 0.1, 0.5, 1, 5, 10],
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'degree': [2, 3, 4, 5, 6],
            'gamma': ['scale', 'auto'],
            'tol': [1e-1, 1e-2, 1e-3, 1e-4],
            'class_weight': [None, 'balanced']
        },
        'Random Forest': {
            'n_estimators':[50, 100, 150], 
            'criterion': ['gini', 'entropy', 'log_loss'], 
            'min_samples_split': [2, 3, 4, 5], 
            'min_samples_leaf': [1, 2, 3]
        },
        'XGBoost': {
            'max_depth': [3, 4, 5, 7],
            'base_score': [0.5], 
            'booster': ['gbtree'],
            'learning_rate': [0.1, 0.01, 0.05],
            'n_estimators': [50, 100],
            'reg_alpha': [0], 
            'reg_lambda': [0, 1, 10],
            'gamma': [0, 0.25, 1],
            'subsample': [0.8],
            'colsample_bytree': [0.5]
        },
        'LightGBM': {
            'num_leaves': [2, 3, 5],
            'max_depth': [3, 5, 7, 9],
            'n_estimators': [50, 100],
            'learning_rate':[0.01, 0.03, 0.1, 0.3],
            'reg_lambda': [10, 30, 60],
            'reg_alpha': [10, 30, 60],
            'min_gain_to_split': [2, 5, 10]
        },
        'AdaBoost': {
            'n_estimators': [50, 100, 150, 200], 
            'learning_rate': [0.01, 0.03, 0.1, 0.3, 1.0, 10]
        }
    }

    experiment_params['model_type'] = MODEL_TYPE

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
        train_test_split(X, y, test_size=TEST_SIZE, 
                         random_state=1, shuffle=True, stratify=y)                  
    ax = y_train.value_counts().plot(kind='bar', title='Class label before')
    counts_before = y_train.value_counts().values

    # Downsample testset - same number of samples in each class
    if DOWNSAMP_TEST:
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

        # Plot class balance difference 
        plot_before_after_counts(counts_test_before, counts_test_after)

    # Solve data imbalance issue
    if SOLVE_IMB:
        cb = ClassBalance(X=X_train, y=y_train)
        if TYPE_CI == 'case1':
            balanced_dataset = cb.cut_LumA(thresh=200)
            counts_after = balanced_dataset.expert_PAM50_subtype.value_counts()

        elif TYPE_CI == 'case2':
            balanced_dataset = cb.cut_LumA_LumB_Basal(thresh=[100, 100, 80])
            counts_after = balanced_dataset.expert_PAM50_subtype.value_counts()

        elif TYPE_CI == 'case3':
            sampling_strategy = {
                'LumA': 100,
                'LumB': 100,
                'Basal': 100,
                'Her2': 80,
                'Normal': 80
            }
            balanced_dataset = cb.resampling_with_generation(sampling_strategy)
            counts_after = balanced_dataset.expert_PAM50_subtype.value_counts()

        elif TYPE_CI == 'case4':
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

            # TODO: Correct plotting here
            #ax = df.plot.bar(stacked=False)

        X_train = balanced_dataset.drop(columns='expert_PAM50_subtype', inplace=False)
        y_train = balanced_dataset.expert_PAM50_subtype

        # Plot class balance difference 
        plot_before_after_counts(counts_before, counts_after)

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
    if FEAT_SELECT == 'univariate':
        best_feat_model = SelectKBest(score_func=f_classif, k=N_feats) # k needs to be defined
        best_feat_model.fit(X_train_scaled, y_train)
        df_scores = pd.DataFrame(best_feat_model.scores_)
        df_feats = pd.DataFrame(X.columns)

        featureScores = pd.concat([df_feats, df_scores],axis=1)
        featureScores.columns = ['Feature', 'Score'] 
        
        plt.figure()
        featureScores.nlargest(50, 'Score').plot(kind='barh')
        plt.title(FEAT_SELECT + ' feature selection')
        print(featureScores.nlargest(10, 'Score'))

        selected_feat = featureScores.sort_values(by='Score')[-N_feats:]['Feature']
    
    elif FEAT_SELECT == 'hybrid':
        best_feat_model = SelectKBest(score_func=f_classif, k=N_feats) # k needs to be defined
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

    elif FEAT_SELECT == 'filter':
        with open('high_corr_feat.pkl', 'rb') as file:
            selected_feat = pickle.load(file)

    elif FEAT_SELECT == 'recursive':
        rfe_selector = RFE(estimator=LogisticRegression(), 
                           n_features_to_select=N_feats, step=10, verbose=5)
        rfe_selector.fit(X_train_scaled, y_train)
        rfe_support = rfe_selector.get_support()

        selected_feat = X.loc[:,rfe_support].columns.tolist()
        print(str(len(selected_feat)), 'selected features')

    elif FEAT_SELECT == 'Wrapper':
        best_feat_model  = ExtraTreesClassifier(n_estimators=10)
        best_feat_model.fit(X_train_scaled, y_train)
        
        plt.figure()
        feat_importances = pd.Series(best_feat_model.feature_importances_, index=X.columns)
        feat_importances.nlargest(50).plot(kind='barh')

        plt.figure()
        plt.plot(sorted(best_feat_model.feature_importances_))
        plt.title(FEAT_SELECT + ' feature selection')

        selected_feat = dict(feat_importances.sort_values()[-N_feats:]).keys()

    elif FEAT_SELECT == 'embedded':
        model = LogisticRegression(C=0.1)
        selector = SelectFromModel(model, threshold=0.025) 
        selector.fit_transform(X_train_scaled, y_train)
        selected_feat = X_train_scaled.columns[selector.get_support()]

    elif FEAT_SELECT == 'wrapper':
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
    
    elif FEAT_SELECT == 'wrapper_reg':
        model = Lasso(alpha=0.0001, random_state=42)
        selector = SelectFromModel(model, threshold=0.001) 
        selector.fit_transform(X_train_scaled, y_train)
        selector.fit(X_train_scaled, y_train)
        selected_feat = X_train_scaled.columns[selector.get_support()]

    # Extract the data frames with only selected features
    X_train_scaled_selected = X_train_scaled[list(selected_feat)]
    X_test_scaled_selected = X_test_scaled[list(selected_feat)]

    # Define a model
    if MODEL_TYPE == 'MLP Classifier': 
        classifier = MLPClassifier(random_state=RANDOM_STATE, max_iter=200, early_stopping=True)
    elif MODEL_TYPE == 'Logistic Regression':
        classifier = LogisticRegression(random_state=RANDOM_STATE)
    elif MODEL_TYPE == 'KNN':
        classifier = KNeighborsClassifier()
    elif MODEL_TYPE == 'Decision Tree':
        classifier = DecisionTreeClassifier(random_state=RANDOM_STATE)
    elif MODEL_TYPE == 'SVC':
        classifier = SVC(random_state=RANDOM_STATE)
    elif MODEL_TYPE == 'Random Forest':
        classifier = RandomForestClassifier(random_state=RANDOM_STATE)
    elif MODEL_TYPE == 'XGBoost':
        classifier = xgb.XGBClassifier(random_state=RANDOM_STATE)
    elif MODEL_TYPE == 'LightGBM':
        classifier = lgb.LGBMClassifier(random_state=RANDOM_STATE)
    elif MODEL_TYPE == 'AdaBoost':
        classifier = AdaBoostClassifier(random_state=RANDOM_STATE)
    else:
        print('There no such model to be choosen! Try again!')
        exit()

    # ---------------- Cross-validation ---------------
    if CROSS_VAL:
        scores = cross_val_score(classifier, X_train_scaled, y_train,
                                 scoring='neg_log_loss', 
                                 cv=N_folds, verbose=5)
        
    # ----------------- TRAIN & TEST ------------------
    if OPTIM:
        # Define Grid Search
        gs = GridSearchCV(classifier, param_grid=MODEL_PARAMS[MODEL_TYPE], 
                        scoring='accuracy', cv=N_folds, verbose=3)
        start = time.time()
        gs.fit(X_train_scaled_selected.values, y_train)
        stop = time.time()

        model = gs.best_estimator_
        best_params = gs.best_params_
        experiment_params['model_params'] = best_params
        experiment_name['best_model'] = model

    else:
        start = time.time()
        model = classifier.fit(X_train_scaled_selected, y_train)
        stop = time.time()

        experiment_params['model_params'] = MODEL_PARAMS[MODEL_TYPE]

    experiment_params['training_time'] =  stop-start
    
    # --------------- Compute predictions ------------------
    pred = model.predict(X_test_scaled_selected.values)

    # ---------------- Compute metrics ---------------------
    metrics = cmp_metrics(pred, y_test)
    experiment_params['results'] = metrics

    conf_mat = confusion_matrix(y_test, pred)        
    #experiment_params['results']['confusion_matrix'] = conf_mat

    df = pd.DataFrame(conf_mat, index = [i for i in LB.classes_], columns = [i for i in LB.classes_])
    sns.heatmap(df.div(df.values.sum()), annot=True)
    plt.show()

    #print('After selection there is: {} genes!\nBefore selection we had {} genes!'.format(selected_features.shape[0], X.shape[1]))

    # Save the experiment properties as .json file
    with open(os.path.join(EXP_PATH, experiment_name + '.pkl'), 'wb') as file:
        pickle.dump(experiment_params, file)

    time.sleep(2)

if __name__=='__main__':
    main()