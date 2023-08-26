import pandas as pd
from sklearn.utils import resample
#from imblearn.over_sampling import SMOTE 

def remove_extreme(X: pd.DataFrame, change_X: bool=True):
    """
        Removes extreme genes and finds potential samples to be removed!
    """
    cmp_X = X.divide(X.sum(axis=1), axis=0) * 1e6

    # 1. Check presence of NULL values and remove genes that contain 80% of them
    X_nans = cmp_X.isnull().sum(axis = 0)
    X_nans_80 = (X_nans / cmp_X.shape[0]).gt(0.8)
    X = X.loc[:, ~X_nans_80]

    print('There are {} columns with more than 80% of Null values!'.\
          format(X_nans_80.sum()))
    
    # 2. Check where the raw count values are lower than 4 for more than 20% of samples
    X_4_keep = (X > 4).sum(axis=0)
    X_4_20_keep = (X_4_keep / cmp_X.shape[0]).gt(0.2)
    feat_to_remove = set(X.columns).difference(set(X.loc[:, X_4_20_keep].columns))
    feat_to_keep = X.loc[:, X_4_20_keep].columns
    if change_X:
        X = X.loc[:, X_4_20_keep]

    print('There are {} columns with more than 20% of count values greater than 4!'.\
          format(X_4_20_keep.sum()))
    
    # 3. Check if there are samples where the sum of the expression 
    # values of the 5 most expressed genes is greater than the 20% of 
    # the total expression of the sample

    sample_sums = cmp_X.sum(axis=1)
    top_5_sum = [sum(cmp_X.iloc[row, :].nlargest(5).tolist())*0.2 \
                 for row in range(cmp_X.shape[0])]
    
    samples_to_remove = []
    if (sample_sums<top_5_sum).sum():
        print('There are {} samples that have total expression \
              lower than 20% of sum of 5 most expressed genes!'.\
                format((sample_sums<top_5_sum).sum()))
        s = (sample_sums<top_5_sum)
        samples_to_remove = s[s].index.values

    return X, samples_to_remove, feat_to_remove, feat_to_keep


class ClassBalance:
    def __init__(self, X: pd.DataFrame, y: pd.Series):
        self.X = X
        self.y = y
        self.data = pd.concat([X, y], axis=1)

    def cut_LumA(self, thresh: float):
        resampled_data = resample(self.data.loc[self.y=='LumA',:], 
                                    random_state=42, 
                                    replace=False,
                                    n_samples=thresh)
        resampled_data = pd.concat([self.data.loc[self.y!='LumA',:], resampled_data])
        return resampled_data
    
    def cut_LumA_LumB_Basal(self, thresh):
        resampled_lumA = resample(self.data.loc[self.y=='LumA',:], 
                                    random_state=42, 
                                    replace=False,
                                    n_samples=thresh[0])
        resampled_lumB = resample(self.data.loc[self.y=='LumB',:], 
                                    random_state=42, 
                                    replace=False,
                                    n_samples=thresh[1])
        resampled_basal = resample(self.data.loc[self.y=='Basal',:], 
                                    random_state=42, 
                                    replace=False,
                                    n_samples=thresh[2])
        
        rest = (self.y!='LumA').values & (self.y!='LumB').values & (self.y!='Basal').values
        resampled_data = pd.concat([self.data.loc[rest,:], resampled_lumA, resampled_lumB, resampled_basal])
        return resampled_data
    
    def augment_Normal(self, sampling_strategy):
        smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42, k_neighbors=5)
        X_smote, y_smote = smote.fit_resample(self.X, self.y.ravel())
        y_smote = pd.DataFrame(y_smote, columns=['expert_PAM50_subtype'])
        return pd.concat([X_smote, y_smote], axis=1)


    def resampling(self, balance_treshs: dict, seed=None):

        idx_sampled = []
        for (class_label, balance_thresh) in balance_treshs.items():

            idx_sampled += list(resample(self.y[self.y==class_label].index, 
                                    replace=False, random_state=seed,
                                    n_samples=int(balance_thresh)))

        return self.data.loc[idx_sampled, :]

    def resampling_with_generation(self, sampling_strategy: dict):

        new_list_data = []
        for (label, value) in sampling_strategy.items():
            counts = (self.y==label).sum()
            if  counts > value:
                downsampled_data = resample(self.data[self.y==label], 
                                            random_state=42, 
                                            replace=False, 
                                            n_samples=value)
                new_list_data.append(downsampled_data)
            else:
                new_list_data.append(self.data[self.y==label])

        data_cut = pd.concat(new_list_data)
        X = data_cut.drop(columns='expert_PAM50_subtype', inplace=False)
        y = data_cut.expert_PAM50_subtype

        smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42, k_neighbors=5)
        X_smote, y_smote = smote.fit_resample(X, y.ravel())
        y_smote = pd.DataFrame(y_smote, columns=['expert_PAM50_subtype'])
    
        new_samples = pd.concat([X, X_smote]).drop_duplicates(keep=False)
        print('Balance status after SMOTE:\n')
        print(y_smote.value_counts())

        return pd.concat([X_smote, y_smote], axis=1), new_samples
    




