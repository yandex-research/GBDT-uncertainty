import numpy as np
from catboost import Pool, CatBoostRegressor
from gbdt_uncertainty.data import load_regression_dataset, make_train_val_test
from scipy.stats import ttest_rel
from gbdt_uncertainty.assessment import prr_regression, nll_regression, calc_rmse, ens_nll_regression, ood_detect, ens_rmse
from gbdt_uncertainty.uncertainty import ensemble_uncertainties_regression
import math
import joblib
import sys
from collections import defaultdict

datasets = ["bostonHousing", "concrete", "energy", "kin8nm", "naval-propulsion-plant",
            "power-plant", "protein-tertiary-structure", "wine-quality-red", "yacht", 
            "YearPredictionMSD"]
algorithms = ['sgb-fixed', 'sglb-fixed'] 

# for proper tables
convert_name = {"bostonHousing": "BostonH", "concrete": "Concrete", "energy": "Energy", 
                "kin8nm": "Kin8nm", "naval-propulsion-plant": "Naval-p", "power-plant": "Power-p",
                "protein-tertiary-structure": "Protein", "wine-quality-red": "Wine-qu", 
                "yacht": "Yacht", "YearPredictionMSD": "Year"}
     
def load_and_predict(X, name, alg, fold, i):
    if alg == "rf":
        model = joblib.load("results/models/" + name + "_" + alg + "_f" + str(fold) + "_" + str(i))
        preds = model.predict(X)
        preds = np.array([(p, 1) for p in preds]) # 1 for unknown variance
    else:
        model = CatBoostRegressor()
        model.load_model("results/models/" + name + "_" + alg + "_f" + str(fold) + "_" + str(i)) 
        preds = model.predict(X)
    return preds, model
    
def predict(X, model, alg):
    preds = model.predict(X)
    if alg == "rf":
        preds = np.array([(p, 1) for p in preds])
    return preds
    
def rf_virtual_ensembles_predict(model, X, count=10):
    trees = model.estimators_
    num_trees = len(trees)
    ens_preds = []
    for i in range(count):
        indices = range(int(i*num_trees/count), int((i+1)*num_trees/count))
        all_preds = []
        for ind in indices:
            all_preds.append(trees[ind].predict(X))
        all_preds = np.array(all_preds)
        preds = np.mean(all_preds, axis=0)
        preds = np.array([(p, 1) for p in preds]) # 1 for unknown variance
        ens_preds.append(preds)
    ens_preds = np.array(ens_preds)

    return np.swapaxes(ens_preds, 0, 1)
    
def virtual_ensembles_load_and_predict(X, name, alg, fold, i, num_models=10):
    if alg == "rf":
        model = joblib.load("results/models/" + name + "_" + alg + "_f" + str(fold) + "_" + str(i))
        all_preds = rf_virtual_ensembles_predict(model, X)
    else:
        model = CatBoostRegressor()
        model.load_model("results/models/" + name + "_" + alg + "_f" + str(fold) + "_" + str(i)) 
        all_preds = model.virtual_ensembles_predict(X, prediction_type='VirtEnsembles', virtual_ensembles_count=num_models)
    return np.swapaxes(all_preds, 0, 1), model
  
def virtual_ensembles_predict(X, model, alg, num_models=10):
    if alg == "rf":
        all_preds = rf_virtual_ensembles_predict(model, X)
    else:
        all_preds = model.virtual_ensembles_predict(X, prediction_type='VirtEnsembles', virtual_ensembles_count=num_models)
    return np.swapaxes(all_preds, 0, 1)
    
def compute_significance(values_all, metric, minimize=True, raw=False):

    if raw:
        values_all = values_all[:, 0, :]

    values_mean = np.mean(values_all, axis=1) # mean wrt folds or elements
    
    if raw and metric == "rmse":
        values_mean = np.sqrt(values_mean)
    
    # choose best algorithm
    if minimize:
        best_idx = np.nanargmin(values_mean)
    else:
        best_idx = np.nanargmax(values_mean)
        
    textbf = {best_idx} # for all algorithms insignificantly different from the best one
    # compute statistical significance on test or wrt folds

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
    
def make_table_entry(values_all, metric, minimize=True, round=2, raw=True):
    
    num_values = len(values_all)
    
    values_mean, textbf = compute_significance(values_all, metric, minimize=minimize, raw=raw)

    # prepare all results in latex format

    table = ""

    for idx in range(num_values):
        if idx in textbf:
            table += "\\textbf{" + str(np.round(values_mean[idx], round)) + "} "
        else:    
            table += str(np.round(values_mean[idx], round)) + " "
        table += "& " 
            
    return table
            
def aggregate_results(name, modes = ["single", "ens", "virt"], 
                      algorithms = ['sgb-fixed', 'sglb-fixed'], num_models = 10, 
                      raw=False):
    
    X, y, index_train, index_test, n_splits = load_regression_dataset(name)
    
    results = [] # metric values for all algorithms and all folds
    
    # for ood evaluation
    ood_X_test = np.loadtxt("datasets/ood/" + name)
    if name == "naval-propulsion-plant":
        ood_X_test = ood_X_test[:, :-1]
    ood_size = len(ood_X_test)
        
    for mode in modes:
        for alg in algorithms:
        
            values = defaultdict(lambda: []) # metric values for all folds for given algorithm

            for fold in range(n_splits):
                
                X_train_all, y_train_all, X_train, y_train, X_validation, y_validation, X_test, y_test = make_train_val_test(
                                                                                        X, y, index_train, index_test, fold)
                
                
                test_size = len(X_test)
                domain_labels = np.concatenate([np.zeros(test_size), np.ones(ood_size)])

                if mode == "single":
                    # use 0th model from ensemble as a single model
                    preds, model = load_and_predict(X_test, name, alg, fold, 0)
                    
                    values["rmse"].append(calc_rmse(preds[:, 0], y_test, raw=raw))
                    values["nll"].append(nll_regression(y_test, preds[:, 0], preds[:, 1], raw=raw))
                    values["TU_prr"].append(prr_regression(y_test, preds[:, 0], preds[:, 1]))
                    values["KU_prr"].append(float("nan"))
                    values["KU_auc"].append(float("nan"))
                    
                    ood_preds = predict(ood_X_test, model, alg)
                    in_measure = preds[:, 1]
                    out_measure = ood_preds[:, 1]
                    values["TU_auc"].append(ood_detect(domain_labels, in_measure, out_measure, mode="ROC"))

                if mode == "ens":
                    all_preds = [] # predictions of all models in ensemble
                    all_preds_ood = []
                    
                    for i in range(num_models):
                        preds, model = load_and_predict(X_test, name, alg, fold, i)
                        all_preds.append(preds)
                        preds = predict(ood_X_test, model, alg)
                        all_preds_ood.append(preds)   
                    all_preds = np.array(all_preds)
                    
                    values["rmse"].append(ens_rmse(y_test, all_preds, raw=raw))
                    values["nll"].append(ens_nll_regression(y_test, all_preds, raw=raw)) 
                    
                    TU = ensemble_uncertainties_regression(np.swapaxes(all_preds, 0, 1))["tvar"]
                    KU = ensemble_uncertainties_regression(np.swapaxes(all_preds, 0, 1))["varm"]

                    mean_preds = np.mean(all_preds[:, :, 0], axis=0)

                    values["TU_prr"].append(prr_regression(y_test, mean_preds, TU))
                    values["KU_prr"].append(prr_regression(y_test, mean_preds, KU))
                    
                    all_preds_ood = np.array(all_preds_ood)
                    TU_ood = ensemble_uncertainties_regression(np.swapaxes(all_preds_ood, 0, 1))["tvar"]
                    KU_ood = ensemble_uncertainties_regression(np.swapaxes(all_preds_ood, 0, 1))["varm"]
                    values["TU_auc"].append(ood_detect(domain_labels, TU, TU_ood, mode="ROC"))
                    values["KU_auc"].append(ood_detect(domain_labels, KU, KU_ood, mode="ROC"))
                        
                if mode == "virt":
                    if alg in ["sgb", "sgb-fixed"]: # we do not evaluate virtual sgb model
                        continue
                    # generate virtual ensemble from 0th model
                    all_preds, model = virtual_ensembles_load_and_predict(X_test, name, alg, fold, 0)

                    values["rmse"].append(ens_rmse(y_test, all_preds, raw=raw))
                    values["nll"].append(ens_nll_regression(y_test, all_preds, raw=raw)) 
                    
                    TU = ensemble_uncertainties_regression(np.swapaxes(all_preds, 0, 1))["tvar"]
                    KU = ensemble_uncertainties_regression(np.swapaxes(all_preds, 0, 1))["varm"]
                    
                    mean_preds = np.mean(all_preds[:, :, 0], axis=0)

                    values["TU_prr"].append(prr_regression(y_test, mean_preds, TU))
                    values["KU_prr"].append(prr_regression(y_test, mean_preds, KU))
                    
                    all_preds_ood = virtual_ensembles_predict(ood_X_test, model, alg)
                    all_preds_ood = np.array(all_preds_ood)
                    
                    TU_ood = ensemble_uncertainties_regression(np.swapaxes(all_preds_ood, 0, 1))["tvar"]
                    KU_ood = ensemble_uncertainties_regression(np.swapaxes(all_preds_ood, 0, 1))["varm"]
                    
                    values["TU_auc"].append(ood_detect(domain_labels, TU, TU_ood, mode="ROC"))
                    values["KU_auc"].append(ood_detect(domain_labels, KU, KU_ood, mode="ROC"))

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
                  
if table_type == "std":
                      
    print("===Results with std===") 
    # results with std

    for name in datasets:
        print(name)

        values = aggregate_results(name, modes = ["single"], 
                                   algorithms = ['sgb-fixed'], raw=False)
        
        #print(values)
        #exit(0)
        
        mean = np.mean(values[0]["rmse"])
        std = np.std(values[0]["rmse"])
        print("rmse:", np.round(mean, 2), "$\pm$", np.round(std,2)),

        mean = np.mean(values[0]["nll"])
        std = np.std(values[0]["nll"])
        print("nll:", np.round(mean, 2), "$\pm$", np.round(std,2))
    
if table_type == "nll_rmse":
    print("===NLL and RMSE Table===")
        
    for name in datasets:
        
        raw = False
        if name == "YearPredictionMSD":
            raw = True

        values = aggregate_results(name, raw=raw)
        
        table = convert_name[name] + " & "
        
        values_nll = np.array([values[i]["nll"] for i in range(len(values))])
        values_rmse = np.array([values[i]["rmse"] for i in range(len(values))])
        
        table += make_table_entry(values_nll, "nll", round=2, raw=raw)
        table += make_table_entry(values_rmse, "rmse", round=2, raw=raw)
        print(table.rstrip("& ") + " \\\\")
        
if table_type == "prr_auc":
    print("===PRR and AUC-ROC Table===")
    
    datasets = ["naval-propulsion-plant"]
        
    for name in datasets:

        values = aggregate_results(name, raw=False)
        
        prr_TU = np.array([values[i]["TU_prr"] for i in range(len(values))])
        prr_KU = np.array([values[i]["KU_prr"] for i in range(len(values))])
        prr = 100*np.concatenate((prr_TU, prr_KU), axis=0)

        mean_prr, textbf_prr = compute_significance(prr, "prr", minimize=False)
    
        auc_TU = np.array([values[i]["TU_auc"] for i in range(len(values))])
        auc_KU = np.array([values[i]["KU_auc"] for i in range(len(values))])
        auc = 100*np.concatenate((auc_TU, auc_KU), axis=0)
        mean_auc, textbf_auc = compute_significance(auc, "auc", minimize=False)

        num = len(auc_TU)
    
        table = "\multirow{2}{*} {" + convert_name[name] + "} & TU &"
        for idx in range(num):
            table += make_table_element(mean_prr, textbf_prr, idx)

        for idx in range(num):
            table += make_table_element(mean_auc, textbf_auc, idx)
            
        print(table.rstrip("& ") + " \\\\")
        
        table = " & KU & "
        for idx in range(num, 2*num):
            table += make_table_element(mean_prr, textbf_prr, idx)
            
        for idx in range(num, 2*num):
            table += make_table_element(mean_auc, textbf_auc, idx)
        print(table.rstrip("& ") + " \\\\")
        
        print("\midrule")
 
if table_type == "rf_rmse": 

    print("===Comparison with random forest, RMSE===")
    for name in datasets:
        
        raw = False
        if name == "YearPredictionMSD":
            raw = True

        values = aggregate_results(name, algorithms=["sglb-fixed", "rf"], modes=["single", "ens"], raw=raw)
        
        table = convert_name[name] + " & "
        
        values_rmse = np.array([values[i]["rmse"] for i in range(len(values))])
        
        table += make_table_entry(values_rmse, "rmse", round=2, raw=raw)
        print(table.rstrip("& ") + " \\\\")
        
if table_type == "rf_prr_auc":
    print("===Comparison with random forest, PRR and AUC-ROC===")
        
    for name in datasets:

        values = aggregate_results(name, algorithms=["sglb-fixed", "rf"], modes=["virt", "ens"], raw=False)
        
        prr_TU = np.array([values[i]["TU_prr"] for i in range(len(values))])
        prr_KU = np.array([values[i]["KU_prr"] for i in range(len(values))])
        prr = 100*np.concatenate((prr_TU, prr_KU), axis=0)

        mean_prr, textbf_prr = compute_significance(prr, "prr", minimize=False)
    
        auc_TU = np.array([values[i]["TU_auc"] for i in range(len(values))])
        auc_KU = np.array([values[i]["KU_auc"] for i in range(len(values))])
        auc = 100*np.concatenate((auc_TU, auc_KU), axis=0)
        mean_auc, textbf_auc = compute_significance(auc, "auc", minimize=False)

        num = len(auc_TU)
    
        table = "\multirow{2}{*} {" + convert_name[name] + "} & TU &"
        for idx in range(num):
            table += make_table_element(mean_prr, textbf_prr, idx)

        for idx in range(num):
            table += make_table_element(mean_auc, textbf_auc, idx)
            
        print(table.rstrip("& ") + " \\\\")
        
        table = " & KU & "
        for idx in range(num, 2*num):
            table += make_table_element(mean_prr, textbf_prr, idx)
            
        for idx in range(num, 2*num):
            table += make_table_element(mean_auc, textbf_auc, idx)
        print(table.rstrip("& ") + " \\\\")
        
        print("\midrule")

    