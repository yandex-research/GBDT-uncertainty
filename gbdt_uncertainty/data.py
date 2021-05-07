import os
from category_encoders.leave_one_out import LeaveOneOutEncoder
from catboost.utils import read_cd
import numpy as np
from catboost import Pool

def load_regression_dataset(name):
    if name == "YearPredictionMSD":
        # https://archive.ics.uci.edu/ml/datasets/YearPredictionMSD
        data = np.loadtxt("datasets/" + name + ".txt", delimiter=",")
        n_splits = 1
        index_features = [i for i in range(1, 91)]
        index_target = 0
    else:
        # repository with all UCI datasets
        url = "https://raw.githubusercontent.com/yaringal/DropoutUncertaintyExps/master/UCI_Datasets/" + name + "/data/"
        data = np.loadtxt(url + "data.txt")
        n_splits = int(np.loadtxt(url + "n_splits.txt"))
        index_features = [int(i) for i in np.loadtxt(url + "index_features.txt")]
        index_target = int(np.loadtxt(url + "index_target.txt"))

    X = data[:, index_features]  # features
    y = data[:, index_target]  # target

    # prepare data for all train/test splits
    index_train = []
    index_test = []
    for i in range(n_splits):
        if name == "YearPredictionMSD":
            # default split for this dataset, see https://archive.ics.uci.edu/ml/datasets/YearPredictionMSD
            index_train.append([i for i in range(463715)])
            index_test.append([i for i in range(463715, 515345)])
        else:
            index_train.append([int(i) for i in np.loadtxt(url + "index_train_" + str(i) + ".txt")])
            index_test.append([int(i) for i in np.loadtxt(url + "index_test_" + str(i) + ".txt")])

    return X, y, index_train, index_test, n_splits


def make_train_val_test(X, y, index_train, index_test, fold):
    # train_all consists of all train instances
    X_train_all = X[index_train[fold], :]
    y_train_all = y[index_train[fold]]

    X_test = X[index_test[fold], :]
    y_test = y[index_test[fold]]

    # for parameter tuning we use 20% of train dataset for validation
    num_training_examples = int(0.8 * X_train_all.shape[0])
    X_train = X_train_all[0:num_training_examples, :]
    y_train = y_train_all[0:num_training_examples]
    X_validation = X_train_all[num_training_examples:, :]
    y_validation = y_train_all[num_training_examples:]

    return X_train_all, y_train_all, X_train, y_train, X_validation, y_validation, X_test, y_test


def process_classification_dataset(name):
    # converting categorical features to numerical

    data_dir = os.path.join('datasets', name)
    train_file = os.path.join(data_dir, 'full_train')
    test_file = os.path.join(data_dir, 'test')
    cd_file = os.path.join(data_dir, 'pool.cd')

    train = np.loadtxt(train_file, delimiter="\t", dtype="object")
    test = np.loadtxt(test_file, delimiter="\t", dtype="object")
    cd = read_cd(cd_file, data_file=train_file)

    # Target can be called 'Label' or 'Target' in pool.cd
    try:
        label_ind = cd['column_type_to_indices']['Label']
    except:
        label_ind = cd['column_type_to_indices']['Target']

    np.random.seed(42)  # fix random seed
    train = np.random.permutation(train)

    y_train = train[:, label_ind]
    y_train = y_train.reshape(-1)

    y_test = test[:, label_ind]
    y_test = y_test.reshape(-1)

    cat_features = cd['column_type_to_indices']['Categ']  # features to be replaced

    enc = LeaveOneOutEncoder(cols=cat_features, return_df=False, random_state=10, sigma=0.3)

    transformed_train = enc.fit_transform(train, y_train).astype("float64")
    X_train = np.delete(transformed_train, label_ind, 1)  # remove target column

    transformed_test = enc.transform(test).astype("float64")
    X_test = np.delete(transformed_test, label_ind, 1)  # remove target column

    return np.nan_to_num(X_train), y_train, np.nan_to_num(X_test), y_test, enc


def load_KDD_dataset(path, eval=False):
    cat_features = [1, 2, 3, 6, 11, 20, 21]

    train_data, train_labels = [], []
    data, labels = [], []
    test_data, test_labels = [], []
    ood_data, ood_labels = [], []

    if not os.path.exists(f'{path}/train_labels.txt') or eval is False:
        train_counts = np.loadtxt(f'{path}/train_counts.txt', dtype=np.float32)
        with open(f'{path}/kdd_train_compressed.csv', 'r') as f:
            for line in f.readlines():
                line = line[:-2].split(',')
                train_data.append(line[:-1])
                train_labels.append(line[-1])

        with open(f'{path}/train_labels.txt', 'w') as f:
            for label in set(train_labels):
                f.write(label + '\n')
        known_attacks = set(train_labels)
    else:
        with open(f'{path}/train_labels.txt', 'r') as f:
            known_attacks = []
            for line in f.readlines():
                known_attacks.append(line[:-1])
            known_attacks = set(known_attacks)

    with open(f'{path}/corrected_compressed.csv', 'r') as f:
        counts = np.loadtxt(f'{path}/corrected_counts.txt')
        for line in f.readlines():
            line = line[:-2].split(',')
            data.append(line[:-1])
            labels.append(line[-1])

    test_attacks = set(labels)
    ood_attacks = test_attacks - known_attacks
    test_counts = []

    for d, l, c in zip(data, labels, counts):
        if l in known_attacks:
            test_data.append(d)
            test_labels.append(l)
            test_counts.append(c)
        elif l in ood_attacks:
            ood_data.append(d)
            ood_labels.append(l)

    if eval:
        train_dataset = None
    else:
        train_dataset = Pool(data=train_data,
                             label=train_labels,
                             weight=train_counts,
                             cat_features=cat_features)

    test_dataset = Pool(data=test_data,
                        label=test_labels,
                        weight=np.asarray(test_counts),
                        cat_features=cat_features)

    ood_dataset = Pool(data=ood_data,
                       label=ood_labels,
                       cat_features=cat_features)

    return train_dataset, test_dataset, ood_dataset
