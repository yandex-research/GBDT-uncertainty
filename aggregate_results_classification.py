import numpy as np
from catboost import Pool, CatBoostClassifier
from catboost.utils import read_cd
from gbdt_uncertainty.data import process_classification_dataset
from gbdt_uncertainty.assessment import prr_class, ood_detect, nll_class
from gbdt_uncertainty.uncertainty import entropy_of_expected_class, expected_entropy_class, entropy
from sklearn.metrics import zero_one_loss, log_loss
from scipy.stats import ttest_rel
import math
import os
import joblib
import sys
from collections import defaultdict

datasets = ["adult", "amazon", "click", "internet", "appetency", "churn", "upselling", "kick"]

algorithms = ['sgb-fixed', 'sglb-fixed'] 

# for proper tables
convert_name = {"adult": "Adult", "amazon": "Amazon", "click": "Click", 
                "internet": "Internet", "appetency": "KDD-Appetency", "churn": "KDD-Churn",
                "upselling": "KDD-Upselling", "kick": "Kick"}
     
def sigmoid(z):
    return 1/(1 + np.exp(-z))
     
def load_model(name, alg, i):
    if alg == "rf":
        model = joblib.load("results/models/" + name + "_" + alg + "_" + str(i))
    else:
        model = CatBoostClassifier()
        model.load_model("results/models/" + name + "_" + alg + "_" + str(i)) 
    return model
    
def rf_virtual_ensembles_predict(model, X, count=10):
    trees = model.estimators_
    num_trees = len(trees)
    ens_preds = []
    for i in range(count):
        indices = range(int(i*num_trees/count), int((i+1)*num_trees/count))
        all_preds = []
        for ind in indices:
            all_preds.append(trees[ind].predict_proba(X))
        all_preds = np.array(all_preds)
        preds = np.mean(all_preds, axis=0)
        ens_preds.append(preds)
    ens_preds = np.array(ens_preds)

    return np.swapaxes(ens_preds, 0, 1)
    
def virtual_ensembles_predict(X, model, alg, num_models=10):
    if alg == "rf":
        all_preds = rf_virtual_ensembles_predict(model, X, count=num_models)
    else:
        all_preds = model.virtual_ensembles_predict(X, prediction_type='VirtEnsembles', virtual_ensembles_count=num_models)
        all_preds = sigmoid(all_preds)
        all_preds = np.concatenate((1 - all_preds, all_preds), axis=2)
    return np.swapaxes(all_preds, 0, 1)
    
def compute_significance(values_all, metric, minimize=True):

    values_mean = np.mean(values_all, axis=1) 
    
    # choose best algorithm
    if minimize:
        best_idx = np.nanargmin(values_mean)
    else:
        best_idx = np.nanargmax(values_mean)
        
    textbf = {best_idx} # for all algorithms insignificantly different from the best one
    # compute statistical significance on test

    for idx in range(len(values_mean)):
        test = ttest_rel(values_all[best_idx], values_all[idx]) # paired t-test
        if test[1] > 0.05:
            textbf.add(idx)
            
    return values_mean, textbf

def compute_best(values, minimize=True):

    # choose best algorithm
    if minimize:
        best_idx = np.nanargmin(values)
    else:
        best_idx = np.nanargmax(values)
        
    textbf = {best_idx} 
    for idx in range(len(values)):
        if values[best_idx] == values[idx]: 
            textbf.add(idx)
            
    return textbf
    
def make_table_entry(values_all, metric, minimize=True, round=2):
    
    num_values = len(values_all)
    
    values_mean, textbf = compute_significance(values_all, metric, minimize=minimize)

    # prepare all results in latex format

    table = ""

    for idx in range(num_values):
        if idx in textbf:
            table += "\\textbf{" + str(np.round(values_mean[idx], round)) + "} "
        else:    
            table += str(np.round(values_mean[idx], round)) + " "
        table += "& " 
            
    return table

def normalize_test_labels(y_test):
    y_test_norm = []
    c0 = min(y_test)
    for y in y_test:
        if y == c0:
            y_test_norm.append(0)
        else:
            y_test_norm.append(1)
    return np.array(y_test_norm)
            
def aggregate_results(name, modes = ["single", "ens", "virt"], 
                      algorithms = ['sgb-fixed', 'sglb-fixed'], num_models = 10):
    

    results = [] # metric values for all algorithms and all folds
        
    for mode in modes:
        for alg in algorithms:
        
            if alg == "rf":
                train_pool, y_train, test_pool, y_test, enc = process_classification_dataset(name)
                
                # process ood data
                cd = read_cd("datasets/"+name+"/pool.cd", data_file = "datasets/"+name+"/test")
                try: 
                    label_ind = cd['column_type_to_indices']['Label']
                except:
                    label_ind = cd['column_type_to_indices']['Target']

                ood_test_pool = np.loadtxt("datasets/ood/" + name, delimiter="\t", dtype="object")
                ood_test_pool = enc.transform(ood_test_pool).astype("float64")
                ood_test_pool = np.delete(ood_test_pool, label_ind, 1)
                ood_size = len(ood_test_pool)
                
            else:
                test_pool = Pool(data="datasets/"+name+"/test", column_description="datasets/"+name+"/pool.cd")
                ood_test_pool = Pool(data="datasets/ood/" + name, column_description="datasets/"+name+"/pool.cd")
                ood_size = ood_test_pool.num_row()

                y_test = test_pool.get_label()
            
            test_size = len(y_test)
            domain_labels = np.concatenate([np.zeros(test_size), np.ones(ood_size)])
                    
            y_test_norm = normalize_test_labels(y_test)
        
            values = defaultdict() # metric values for all folds for given algorithm

            if mode == "single":
                # use 0th model from ensemble as a single model
                model = load_model(name, alg, 0)
                preds = model.predict(test_pool)
                preds_proba = model.predict_proba(test_pool)
    
                values["error"] = (preds != y_test).astype(int)
                values["nll"] = nll_class(y_test_norm, preds_proba)
                values["TU_prr"] = prr_class(y_test_norm, preds_proba, entropy(preds_proba), False)
                values["KU_prr"] = float("nan")
                values["KU_auc"] = float("nan")
                    
                ood_preds_proba = model.predict_proba(ood_test_pool)
                in_measure = entropy(preds_proba)
                out_measure = entropy(ood_preds_proba)
                values["TU_auc"] = ood_detect(domain_labels, in_measure, out_measure, mode="ROC")

            if mode == "ens":
                all_preds = [] # predictions of all models in ensemble
                all_preds_ood = []
                    
                for i in range(num_models):
                    model = load_model(name, alg, i)
                    preds = model.predict_proba(test_pool)
                    all_preds.append(preds)
                    preds = model.predict_proba(ood_test_pool)
                    all_preds_ood.append(preds) 
                        
                all_preds = np.array(all_preds)
                preds_proba = np.mean(all_preds, axis=0)
                
                all_preds_ood = np.array(all_preds_ood)
                
                preds = np.argmax(preds_proba, axis=1)
                values["error"] = (preds != y_test_norm).astype(int)
                values["nll"] = nll_class(y_test_norm, preds_proba)
                
                TU = entropy_of_expected_class(all_preds)
                DU = expected_entropy_class(all_preds)
                KU = TU - DU
                
                TU_ood = entropy_of_expected_class(all_preds_ood)
                DU_ood = expected_entropy_class(all_preds_ood)
                KU_ood = TU_ood - DU_ood

                values["TU_prr"] = prr_class(y_test_norm, preds_proba, TU, False)
                values["KU_prr"] = prr_class(y_test_norm, preds_proba, KU, False)
                  
                values["TU_auc"] = ood_detect(domain_labels, TU, TU_ood, mode="ROC")
                values["KU_auc"] = ood_detect(domain_labels, KU, KU_ood, mode="ROC")
                        
            if mode == "virt":
                if alg in ["sgb", "sgb-fixed"]: # we do not evaluate virtual sgb model
                    continue
                    
                # generate virtual ensemble from 0th model
                model = load_model(name, alg, 0)

                all_preds = virtual_ensembles_predict(test_pool, model, alg)
                
                preds_proba = np.mean(all_preds, axis=0)
    
                preds = np.argmax(preds_proba, axis=1)
                values["error"] = (preds != y_test_norm).astype(int)
                values["nll"] = nll_class(y_test_norm, preds_proba)
                
                TU = entropy_of_expected_class(all_preds)
                DU = expected_entropy_class(all_preds)
                KU = TU - DU
                
                all_preds_ood = virtual_ensembles_predict(ood_test_pool, model, alg)
                TU_ood = entropy_of_expected_class(all_preds_ood)
                DU_ood = expected_entropy_class(all_preds_ood)
                KU_ood = TU_ood - DU_ood

                values["TU_prr"] = prr_class(y_test_norm, preds_proba, TU, False)
                values["KU_prr"] = prr_class(y_test_norm, preds_proba, KU, False)
                  
                values["TU_auc"] = ood_detect(domain_labels, TU, TU_ood, mode="ROC")
                values["KU_auc"] = ood_detect(domain_labels, KU, KU_ood, mode="ROC")
                        
            if mode == "virt" and alg in ["sgb", "sgb-fixed"]: # we do not evaluate virtual sgb model
                continue
            
            results.append(values)

    return np.array(results)
    
def make_table_element(mean, textbf, idx):
    table = ""
    if np.isnan(mean[idx]):
        table += "--- & "
        return table
    if idx in textbf:
        table += "\\textbf{" + str(int(np.rint(mean[idx]))) + "} "
    else:    
        table += str(int(np.rint(mean[idx]))) + " "
    table += "& "
    return table
                  
table_type = sys.argv[1]
                  
if table_type == "nll_error":
    print("===NLL and Error Table===")
        
    for name in datasets:
        
        values = aggregate_results(name)
        
        table = convert_name[name] + " & "
        
        values_nll = np.array([values[i]["nll"] for i in range(len(values))])
        values_error = np.array([values[i]["error"] for i in range(len(values))])
        
        table += make_table_entry(values_nll, "nll", round=3)
        table += make_table_entry(values_error*100, "error", round=1)
        print(table.rstrip("& ") + " \\\\")
        
if table_type == "prr_auc":
    print("===PRR and AUC-ROC Table===")
    
    for name in datasets:

        values = aggregate_results(name)
        
        prr_TU = np.array([values[i]["TU_prr"] for i in range(len(values))])
        prr_KU = np.array([values[i]["KU_prr"] for i in range(len(values))])
        prr = np.concatenate((prr_TU, prr_KU), axis=0)

        textbf_prr = compute_best(prr, minimize=False)
    
        auc_TU = np.array([values[i]["TU_auc"] for i in range(len(values))])
        auc_KU = np.array([values[i]["KU_auc"] for i in range(len(values))])
        auc = 100*np.concatenate((auc_TU, auc_KU), axis=0)
        
        textbf_auc = compute_best(auc, minimize=False)

        num = len(auc_TU)
    
        table = "\multirow{2}{*} {" + convert_name[name] + "} & TU & "
        for idx in range(num):
            table += make_table_element(prr, textbf_prr, idx)

        for idx in range(num):
            table += make_table_element(auc, textbf_auc, idx)
            
        print(table.rstrip("& ") + " \\\\")
        
        table = " & KU & "
        for idx in range(num, 2*num):
            table += make_table_element(prr, textbf_prr, idx)
            
        for idx in range(num, 2*num):
            table += make_table_element(auc, textbf_auc, idx)
        print(table.rstrip("& ") + " \\\\")
        
        print("\midrule")
 
if table_type == "rf_nll_error": 

    print("===Comparison with random forest, NLL and Error===")
    for name in datasets:

        values = aggregate_results(name, algorithms=["sglb-fixed", "rf"], modes=["single", "ens"])
        
        table = convert_name[name] + " & "
        
        values_nll = np.array([values[i]["nll"] for i in range(len(values))])
        values_error = np.array([values[i]["error"] for i in range(len(values))])
        
        table += make_table_entry(values_nll, "nll", round=3)
        table += make_table_entry(values_error*100, "error", round=1)
        
        print(table.rstrip("& ") + " \\\\")
        

if table_type == "rf_prr_auc":
    print("===Comparison with random forest, PRR and AUC-ROC===")
        
    for name in datasets:

        values = aggregate_results(name, algorithms=["sglb-fixed", "rf"], modes=["virt", "ens"])
        
        prr_TU = np.array([values[i]["TU_prr"] for i in range(len(values))])
        prr_KU = np.array([values[i]["KU_prr"] for i in range(len(values))])
        prr = np.concatenate((prr_TU, prr_KU), axis=0)

        textbf_prr = compute_best(prr, minimize=False)
    
        auc_TU = np.array([values[i]["TU_auc"] for i in range(len(values))])
        auc_KU = np.array([values[i]["KU_auc"] for i in range(len(values))])
        auc = 100*np.concatenate((auc_TU, auc_KU), axis=0)
        textbf_auc = compute_best(auc, minimize=False)

        num = len(auc_TU)
    
        table = "\multirow{2}{*} {" + convert_name[name] + "} & TU & "
        for idx in range(num):
            table += make_table_element(prr, textbf_prr, idx)

        for idx in range(num):
            table += make_table_element(auc, textbf_auc, idx)
            
        print(table.rstrip("& ") + " \\\\")
        
        table = " & KU & "
        for idx in range(num, 2*num):
            table += make_table_element(prr, textbf_prr, idx)
            
        for idx in range(num, 2*num):
            table += make_table_element(auc, textbf_auc, idx)
        print(table.rstrip("& ") + " \\\\")
        
        print("\midrule")

    