from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, LabelBinarizer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, log_loss
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import SelectFromModel, SelectKBest, SequentialFeatureSelector
import pickle as pkl
from data_preprocessing import ClassBalance
import matplotlib.pyplot as plt
import time
import seaborn as sns
import pandas as pd

def main():

    # Set the parameters
    SOLVE_IMB = True # Solve class imbalance problem
    SMOTE = True
    CROSS_VAL = True

    # MODEL TYPE = {
    #   MLP Classifier
    #   Logistic Regression
    #   KNN
    #   Decision Tree
    #   SVC
    #   Random Forest
    #   }

    MODEL_TYPE = 'MLP Classifier'

    # Load the dataset
    DATASET_PATH = "tcga_brca_raw_19036_1053samples.pkl"

    with open(DATASET_PATH, 'rb') as file:
        dataset = pkl.load(file) 

    X = dataset.drop(columns=['expert_PAM50_subtype', 'tcga_id', 'sample_id', 'cancer_type'], inplace=False)
    y = dataset.expert_PAM50_subtype

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, 
                                                        shuffle=True, stratify=y)                  
    ax = y_train.value_counts().plot(kind='bar', title='Class label before')

    # Solve data imbalance issue
    cb = ClassBalance(X=X_train, y=y_train)

    if SOLVE_IMB and not SMOTE:
        
        balance_treshs = {
            'LumA': 100,
            'LumB': 100,
            'Basal': 100,
            'Her2': 80,
            'Normal': 50
        }
        balanced_dataset = cb.resampling(balance_treshs)

    elif SOLVE_IMB and SMOTE:

        sampling_strategy = {
            'LumA': 150,
            'LumB': 150,
            'Basal': 120,
            'Her2': 100,
            'Normal': 80
        }

        balanced_dataset = cb.resampling_with_generation(sampling_strategy)

    ax = balanced_dataset.expert_PAM50_subtype.value_counts().plot(kind='bar', title='Class label after')
    X_train = balanced_dataset.drop(columns='expert_PAM50_subtype', inplace=False)
    y_train = balanced_dataset.expert_PAM50_subtype
    
    # Encode the class labels
    LB = LabelBinarizer()
    y_train = LB.fit_transform(y_train)
    y_test = LB.transform(y_test)

    # Data standardization | normalization
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_test_scaled = scaler.transform(X_test)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)

    # Feature selection
    feat_select_model  = ExtraTreesClassifier(n_estimators=10)
    feat_select_model.fit(X_train_scaled, y_train)
    
    plt.figure()
    feat_importances = pd.Series(feat_select_model.feature_importances_, index=X.columns)
    feat_importances.nlargest(50).plot(kind='barh')
    plt.show()

    plt.figure()
    plt.plot(sorted(feat_select_model.feature_importances_))
    plt.show()

    selected_feat = dict(feat_importances.sort_values()[-500:]).keys()

    X_train_scaled_selected = X_train_scaled[selected_feat]
    X_test_scaled_selected = X_test_scaled[selected_feat]


    # Define a model
    if MODEL_TYPE == 'MLP Classifier': 
        classifier = MLPClassifier(hidden_layer_sizes=(20, 10, 5),
                                solver='lbfgs', random_state=4, 
                                alpha=1e-4, batch_size=5)
    elif MODEL_TYPE == 'Logistic Regression':
        classifier = LogisticRegression(penalty='l2', tol=1e-3, random_state=4,
                                        solver='lbfgs', max_iter = 1000)
    elif MODEL_TYPE == 'KNN':
        classifier = KNeighborsClassifier(n_neighbors=5, weights='uniform', 
                                          leaf_size=30, metric='minkowski')

    elif MODEL_TYPE == 'Decision Tree':
        classifier = DecisionTreeClassifier(criterion='gini', max_depth=10, 
                                            min_samples_split=2, min_samples_leaf=3, 
                                            random_state=4)
    elif MODEL_TYPE == 'SVC':
        classifier = SVC(C = 10, kernel='rbf', degree=4, 
                         gamma='scale', random_state=4)

    elif MODEL_TYPE == 'Random Forest':
        classifier = RandomForestClassifier(n_estimators=100, criterion='gini',
                                            min_samples_split=2, min_samples_leaf=2, 
                                            random_state=4)
    else:
        print('There no such model to be choosen! Try again!')
        exit()

    # Cross-validation to see if there is overfitting
    if CROSS_VAL:
        scores = cross_val_score(classifier, X_train_scaled, y_train,
                                scoring='neg_log_loss', 
                                cv=5, verbose=5)

    # Train and test
    model = classifier.fit(X_train_scaled_selected, y_train)
    pred = model.predict(X_test_scaled_selected)
    prob_pred = model.predict_proba(X_test_scaled_selected)

    precision = precision_score(pred, y_test, average='weighted')
    recall = recall_score(pred, y_test, average='weighted')
    print('Score test: ', precision, recall)

    conf_mat = confusion_matrix(y_test, pred)
    df = pd.DataFrame(conf_mat, index = [i for i in le.classes_], columns = [i for i in le.classes_])
    sns.heatmap(df.div(df.values.sum()), annot=True)
    plt.show()

    # Feature selection
    #selection_model = SelectFromModel(classifier, prefit=True)
    #X_new = selection_model.transform(X_train_scaled)
    #selected_feat_idx = selection_model.get_support()
    #selected_features = X_train.columns[selected_feat_idx]

    #print('After selection there is: {} genes!\nBefore selection we had {} genes!'.format(selected_features.shape[0], X.shape[1]))

    time.sleep(2)

if __name__=='__main__':
    main()