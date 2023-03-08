from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, LabelBinarizer
from sklearn.metrics import confusion_matrix, jaccard_score
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import SelectFromModel, SelectKBest, RFE, f_classif, chi2
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.multioutput import ClassifierChain
import json, os, pickle
import datetime

import pickle as pkl
from data_preprocessing import ClassBalance
import matplotlib.pyplot as plt
import time
import seaborn as sns
import pandas as pd
import numpy as np
from utils import cmp_metrics

def main():

    EXP_PATH = 'experiments'

    # Set the parameters
    SOLVE_IMB = False # Solve class imbalance problem
    SMOTE = True
    CROSS_VAL = False
    MULTI_LABEL = False
    TEST_SIZE = 0.3
    RANDOM_STATE = 4
    OPTIM = True

    N_folds = 10
    N_feats = 500

    # Feature selection can be: Univariate, Recursive...
    FEAT_SELECT = 'Univariate' 

    # Get the current date and time
    now = datetime.datetime.now()
    experiment_name =  'run_' + now.strftime("%d-%m-%Y_%H:%M:%S")
    experiment_params = {
        'solve_ibm': SOLVE_IMB,
        'use_smote': SMOTE,
        'cross_validation': CROSS_VAL,
        'multi_label_classification': MULTI_LABEL,
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
    #   }

    MODEL_TYPE = 'Logistic Regression'

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
            'weights': 'uniform', 
            'leaf_size': [10, 20, 30, 40], 
            'metric': 'minkowski'
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
            'C': [0.1, 0.5, 1, 5, 10],
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'],
            'degree': [3, 4, 5, 6],
            'gamma': ['scale', 'auto'],
            'tol': [1e-1, 1e-2, 1e-3, 1e-4],
            'class_weight': [None, 'balanced']
        }
    }

    experiment_params['model_type'] = MODEL_TYPE

    # Load the dataset
    DATASET_PATH = "tcga_brca_raw_19036_1053samples.pkl"

    with open(DATASET_PATH, 'rb') as file:
        dataset = pkl.load(file) 

    X = dataset.drop(columns=['expert_PAM50_subtype', 'tcga_id', 'sample_id', 'cancer_type'], inplace=False)
    y = dataset.expert_PAM50_subtype

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=1, 
                                                        shuffle=True, stratify=y)                  
    ax = y_train.value_counts().plot(kind='bar', title='Class label before')

    counts_before = y_train.value_counts().values

    # Solve data imbalance issue
    if SOLVE_IMB:
        cb = ClassBalance(X=X_train, y=y_train)

        if not SMOTE:
            balance_treshs = {
                'LumA': 100,
                'LumB': 100,
                'Basal': 100,
                'Her2': 80,
                'Normal': 50
            }
            balanced_dataset = cb.resampling(balance_treshs)
            experiment_params['class_balance_thresholds'] = balance_treshs

        else:
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

        ax = balanced_dataset.expert_PAM50_subtype.value_counts().plot(kind='bar', title='Class label after')
        X_train = balanced_dataset.drop(columns='expert_PAM50_subtype', inplace=False)
        y_train = balanced_dataset.expert_PAM50_subtype

        # Plot class balance difference 
        df = pd.DataFrame({'Original': counts_before,
                           'Generated': counts_diff}, index=new_class_counts.keys())
        ax = df.plot.bar(stacked=True)

    
    # Encode the class labels
    if MULTI_LABEL:
        LB = LabelBinarizer()
        y_train = LB.fit_transform(y_train)
        y_test = LB.transform(y_test)
    else:
        LB = LabelEncoder()
        y_train = LB.fit_transform(y_train)
        y_test = LB.transform(y_test)

    # Data standardization | normalization
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_test_scaled = scaler.transform(X_test)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)

    # Feature selection
    if FEAT_SELECT == 'Univariate':
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

    elif FEAT_SELECT == 'Recursive':
        rfe_selector = RFE(estimator=LogisticRegression(), n_features_to_select=N_feats, step=10, verbose=5)
        rfe_selector.fit(X_train_scaled, y_train)
        rfe_support = rfe_selector.get_support()

        selected_feat = X.loc[:,rfe_support].columns.tolist()
        print(str(len(selected_feat)), 'selected features')

    else:
        best_feat_model  = ExtraTreesClassifier(n_estimators=10)
        best_feat_model.fit(X_train_scaled, y_train)
        
        plt.figure()
        feat_importances = pd.Series(best_feat_model.feature_importances_, index=X.columns)
        feat_importances.nlargest(50).plot(kind='barh')

        plt.figure()
        plt.plot(sorted(best_feat_model.feature_importances_))
        plt.title(FEAT_SELECT + ' feature selection')

        selected_feat = dict(feat_importances.sort_values()[-N_feats:]).keys()

    # Extract the data frames with only selected features
    X_train_scaled_selected = X_train_scaled[selected_feat]
    X_test_scaled_selected = X_test_scaled[selected_feat]

    # Define a model
    if MODEL_TYPE == 'MLP Classifier': 
        classifier = MLPClassifier(andom_state=RANDOM_STATE, max_iter=200, early_stopping=True)
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
    else:
        print('There no such model to be choosen! Try again!')
        exit()

    # Define Grid Search
    gs = GridSearchCV(classifier, param_grid=MODEL_PARAMS[MODEL_TYPE], 
                      scoring='accuracy', cv=N_folds, verbose=5)

    # Cross-validation to see if there is overfitting
    if CROSS_VAL and not MULTI_LABEL:
        scores = cross_val_score(classifier, X_train_scaled, y_train,
                                 scoring='neg_log_loss', 
                                 cv=N_folds, verbose=5)

    if MULTI_LABEL:
        # Use multi-label classifier to wrap the base classifier 
        classifier = OneVsRestClassifier(estimator=classifier)
        if CROSS_VAL:
            scores = cross_val_score(classifier, X_train_scaled, y_train,
                                 scoring='f1_samples', 
                                 cv=N_folds, verbose=5)
            experiment_params['cross_val_scores'] = scores
            print(scores)

    # Train and test
    if OPTIM:
        start = time.time()
        gs.fit(X_train_scaled_selected, y_train)
        stop = time.time()

        model = gs.best_estimator_
        best_params = gs.best_params_
        experiment_params['model_params'] = best_params

    else:
        start = time.time()
        model = classifier.fit(X_train_scaled_selected, y_train)
        stop = time.time()

        experiment_params['model_params'] = MODEL_PARAMS[MODEL_TYPE]


    experiment_params['training_time'] =  stop-start
    
    # Compute predictions
    pred = model.predict(X_test_scaled_selected)

    # Compute metrics
    metrics = cmp_metrics(pred, y_test)
    experiment_params['results'] = metrics

    if not MULTI_LABEL:
        conf_mat = confusion_matrix(y_test, pred)        
        #experiment_params['results']['confusion_matrix'] = conf_mat

        df = pd.DataFrame(conf_mat, index = [i for i in LB.classes_], columns = [i for i in LB.classes_])
        sns.heatmap(df.div(df.values.sum()), annot=True)
        plt.show()

    else:
        jacc_score = jaccard_score(pred, y_test, average='samples')
        experiment_params['results']['jaccard_score'] = jacc_score
        

    if MULTI_LABEL:

        # Fit an ensemble of selected classifier chains and 
        # compute the average prediction of all the chains.

        chains = [ClassifierChain(classifier, order='random', random_state=i) for i in range(10)]
        chain_preds = []
        for chain in chains:
            chain.fit(X_train_scaled_selected, y_train)
            chain_preds.append(chain.predict(X_test_scaled_selected))

        chain_preds = np.array(chain_preds)
        chain_jaccard_scores = [
            jaccard_score(y_pred >=0.5, y_test, average='samples') for y_pred in chain_preds
        ]

        y_pred_avg = chain_preds.mean(axis=0)
        avg_jaccard_score = jaccard_score(y_test, y_pred_avg >=0.5, average='samples')

        model_scores = [jacc_score] + chain_jaccard_scores + [avg_jaccard_score]
        experiment_params['results']['chain_scores'] = model_scores

        ### PLOT
        model_names = (
            "Independent",
            "Chain 1",
            "Chain 2",
            "Chain 3",
            "Chain 4",
            "Chain 5",
            "Chain 6",
            "Chain 7",
            "Chain 8",
            "Chain 9",
            "Chain 10",
            "Ensemble",
        )

        x_pos = np.arange(len(model_names))

        # Plot the Jaccard similarity scores for the independent model, each of the
        # chains, and the ensemble (note that the vertical axis on this plot does
        # not begin at 0).

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.grid(True)
        ax.set_title("Classifier Chain Ensemble Performance Comparison")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(model_names, rotation="vertical")
        ax.set_ylabel("Jaccard Similarity Score")
        ax.set_ylim([min(model_scores) * 0.9, max(model_scores) * 1.1])
        colors = ["r"] + ["b"] * len(chain_jaccard_scores) + ["g"]
        ax.bar(x_pos, model_scores, alpha=0.5, color=colors)
        plt.tight_layout()
        plt.show()

    # Feature selection
    #selection_model = SelectFromModel(classifier, prefit=True)
    #X_new = selection_model.transform(X_train_scaled)
    #selected_feat_idx = selection_model.get_support()
    #selected_features = X_train.columns[selected_feat_idx]

    #print('After selection there is: {} genes!\nBefore selection we had {} genes!'.format(selected_features.shape[0], X.shape[1]))

    # Save the experiment properties as .json file
    with open(os.path.join(EXP_PATH, experiment_name + '.pkl'), 'wb') as file:
        pickle.dump(experiment_params, file)

    time.sleep(2)

if __name__=='__main__':
    main()