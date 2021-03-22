import numpy as np
import pandas as pd
from sklearn.decomposition import FactorAnalysis
from scipy.stats import pareto
import random

def make_mix_ood(dataset, n_rows=None, 
                 external_path= "../datasets/YearPredictionMSD.txt", normalize=True):
    
    data = pd.read_csv("../datasets/" + dataset + "/test", delim_whitespace=True, header=None)
    
    if n_rows is None:
        n_rows = data.shape[0] # size of test set
    
    # get indices of categorical featrues

    cat_ind = [] # feature indices
    with open("../datasets/" + dataset + "/pool.cd") as cd_file:
        for line in cd_file:
            if line.split()[1] == "Categ":
                cat_ind.append(int(line.split()[0]))

    # create new categorical features
    
    new_cat_features = []

    for ind in cat_ind:
        cat_values = set(data.values[:, ind])
        new_cat_features.append([random.sample(cat_values, 1)[0] for i in range(n_rows)])
       
    # get external dataset
    
    if external_path == '../datasets/slice_localization.csv':
        external_data = pd.read_csv(external_path)
        external_data.drop(columns=[external_data.columns[60], external_data.columns[70], 
                           external_data.columns[180], external_data.columns[280], external_data.columns[190],
                           external_data.columns[352]], inplace=True) # remove columns with zero variance
    else:
        external_data = pd.read_csv(external_path, header=None)
            
    ext_values = external_data.values
            
    if data.shape[1] > external_data.shape[1]: # special treatment for kdd datasets with large number of neatures
        r_1 = np.random.rand(53500, 1) # 0 - target
        r_2 = ext_values[:, :190] # 1-190 - numerical features
        r_3 = np.random.rand(53500, 18) # 191-208 - categorical features
        r_4 = ext_values[:, 190:191] # 209 - numerical feature
        r_5  = np.random.rand(53500, 20) # 210-229 - categorical features
        r_6 = ext_values[:, 191:] # remaining features are numerical
        r_7 = np.random.rand(53500, 1) # last feature is constant in data
    
        ext_values = np.concatenate((r_1, r_2, r_3, r_4, r_5, r_6, r_7), axis=1)
        

    print(data.shape)
    print(ext_values.shape)

    assert data.shape[1] <= ext_values.shape[1]
    
    # create numerical ood dataset
    
    num_features = data.shape[1]
    
    ext_values = ext_values[:n_rows, :num_features] 

    if normalize:
        # replace categorical features in data by 0s for correct operation below
        data_values = data.values
        for ind in cat_ind:
            data_values[:, ind] = np.zeros(n_rows)
        data_values = data_values.astype("float64")
        print(ext_values.std(axis=0))
        ext_values = (ext_values - ext_values.mean(axis=0)) / ext_values.std(axis=0)
        ext_values = ext_values * data_values.std(axis=0) + data_values.mean(axis=0)
        
    # replace values for categorical features
    
    ext_values = ext_values.astype("object")
    for i, ind in enumerate(cat_ind):
        ext_values[:, ind] = new_cat_features[i]
        
    pd.DataFrame(ext_values).to_csv('../datasets/ood/' + dataset, index=False, header=None, sep="\t")

if __name__ == '__main__':

    datasets = ["adult", "amazon", "click", "internet", 
                "appetency", "churn", "upselling", "kick"]
                
    datasets = ["appetency", "churn", "upselling"]

    for dataset in datasets:
        if dataset in ["appetency", "churn", "upselling"]:
            make_mix_ood(dataset, external_path='../datasets/slice_localization.csv')
        else:
            make_mix_ood(dataset)
        print("Finished", dataset)

