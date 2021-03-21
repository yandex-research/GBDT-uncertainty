import numpy as np
import pandas as pd

def make_mix_ood(dataset, n_rows=None, external_path= "../datasets/YearPredictionMSD.txt", 
                 normalize=True):

    if dataset == "YearPredictionMSD":
        # https://archive.ics.uci.edu/ml/datasets/YearPredictionMSD
        data = pd.read_csv("../datasets/" + dataset + ".txt", header=None)
        index_target = 0
    else:
        # repository with all UCI datasets
        url = "https://raw.githubusercontent.com/yaringal/DropoutUncertaintyExps/master/UCI_Datasets/" + dataset + "/data/"
        data = pd.read_csv(url + "data.txt", delim_whitespace=True, header=None)
        # print(data)
        n_splits = int(np.loadtxt(url + "n_splits.txt"))
        index_features = [int(i) for i in np.loadtxt(url + "index_features.txt")]
        index_target = int(np.loadtxt(url + "index_target.txt"))

    if n_rows is None:
        n_rows = int(round(data.shape[0] * 0.1)) # size of test set

    if external_path == '../datasets/slice_localization.csv':
        external_data = pd.read_csv(external_path)
        external_data.drop(columns=[external_data.columns[60], external_data.columns[70], external_data.columns[180], external_data.columns[190], external_data.columns[352]], inplace=True)
    else:
        external_data = pd.read_csv(external_path, header=None)
    assert data.shape[1] <= external_data.shape[1]
    num_features = data.shape[1]
    values = external_data.values[:n_rows, :num_features]
    if normalize:
        print(values.std(axis=0))
        values = (values - values.mean(axis=0)) / values.std(axis=0)
        values = values * data.values.std(axis=0) + data.values.mean(axis=0)
    values = np.delete(values, [index_target], 1) # remove column corresponding to target
    pd.DataFrame(values).to_csv('../datasets/ood/' + dataset, index=False, 
                                header=None, sep="\t")

if __name__ == '__main__':

    datasets = ["bostonHousing", "concrete", "energy", "kin8nm",
                "naval-propulsion-plant", "power-plant", "protein-tertiary-structure", "wine-quality-red", "yacht", "YearPredictionMSD"]

    for dataset in datasets:
        if dataset == "YearPredictionMSD":
            make_mix_ood(dataset, n_rows=51630, external_path='../datasets/slice_localization.csv')
        else:
            make_mix_ood(dataset)
        print("Finished", dataset)

