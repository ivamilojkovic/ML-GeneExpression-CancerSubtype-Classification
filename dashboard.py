import streamlit as st
import pandas as pd 
import numpy as np
import pickle
import seaborn as sns
import os, glob

import matplotlib.pyplot as plt
plt.style.use('ggplot')
sns.set_theme()
plt.rcParams["figure.figsize"] = (12, 4)

def load_data(path):
    with open(path, 'rb') as file:
        data = pickle.load(file) 
    data.drop(columns=['expert_PAM50_subtype', 'tcga_id', \
                        'sample_id', 'cancer_type'], inplace=True)
    return data

def load_model(path):
    fig, ax = plt.subplots()
    with open(os.path.join(curr_dir, 'experiments', path), 'rb') as file:
        exp = pickle.load(file)

    df_rf = pd.DataFrame({'Precision': exp['results']['Precision per class'],
                    'Recall': exp['results']['Recall per class'],
                    'F1 score': exp['results']['F1 score per class']}, 
                    index=['LumA', 'LumB', 'Basal', 'Ref2', 'Normal'])
    df_rf.plot(kind='bar', rot=30, title=exp['model_type'], ax=ax)
    for g in ax.patches:
        ax.annotate(format(g.get_height(), '.3f'),
                    (g.get_x() + g.get_width() / 2., g.get_height()),
                    ha = 'center', va = 'center',
                    xytext = (0, 5),
                    textcoords = 'offset points',
                    fontsize=6)
    ax.legend(loc='lower right')
    plt.tight_layout()
    st.pyplot(fig)


curr_dir = os.getcwd()
experiments = sorted(glob.glob('experiments/*.pkl'),  key=os.path.getmtime)

###########################################################################

st.title('RESULTS')

st.write(load_data('tcga_brca_raw_19036_1053samples.pkl'))

case = st.radio(
    "Select a case for class imbalance:",
    ('case 1', 'case 2', 'case 3'))

if case == 'case 1':
    model = st.radio(
        "Select a model:",
        ('Logistic Regression', 'Random Forest', 
         'Support Vector Machine', 'XGBoost' ))
    
    path_case1 = ['run_14-04-2023_13:35:12.pkl',
                'run_04-04-2023_19:00:19.pkl',
                'run_14-04-2023_18:51:21.pkl',
                'run_14-04-2023_14:34:31.pkl']

    if model == 'Logistic Regression':
        load_model(path=path_case1[0])
    elif model == 'Random Forest':
        load_model(path=path_case1[1])
    elif model == 'Support Vector Machine':
        load_model(path=path_case1[3])
    elif model == 'XGBoost':
        load_model(path=path_case1[2])
        
    
elif case == 'case 2':
    st.write("You didn\'t select comedy.")
elif case == 'case 3':
    st.write('')