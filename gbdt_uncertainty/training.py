import os
import numpy as np
import joblib
from catboost import Pool, CatBoostRegressor, CatBoostClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from gbdt_uncertainty.data import make_train_val_test, process_classification_dataset

def tune_parameters_regression(X, y, index_train, index_test, n_splits, alg='sgb'):

    params = []
    seed = 1000 # starting random seed for hyperparameter tuning
    
    for fold in range(n_splits):

        # make catboost pools
        X_train_all, y_train_all, X_train, y_train, X_validation, y_validation, X_test, y_test = make_train_val_test(X, y, index_train, index_test, fold)
        full_train_pool = Pool(X_train_all, y_train_all)
        train_pool = Pool(X_train, y_train)
        validation_pool = Pool(X_validation, y_validation)
        test_pool = Pool(X_test, y_test)
        
        # list of hyperparameters for grid search
        # we do not tune the number of trees, it is important for virtual ensembles
        depths = [3, 4, 5, 6] # tree depth
        lrs = [0.001, 0.01, 0.1] # learning rate 
        if alg == "sgb" or alg == "sglb": # by default, we tune sample rate
            samples = [0.25, 0.5, 0.75]
        if alg == "sgb-fixed": # sgb without sample rate tuning
            samples = [0.5]
        if alg == "sglb-fixed": # sglb without sample rate tuning
            samples = [1.0]
        shape = (len(depths), len(lrs), len(samples))

        # perform grid search
        results = np.zeros(shape)
        for d, depth in enumerate(depths):
            for l, lr in enumerate(lrs):
                for s, sample in enumerate(samples):
                    if alg == 'sgb' or alg == 'sgb-fixed':
                        model = CatBoostRegressor(loss_function='RMSEWithUncertainty',
                                                  learning_rate=lr, depth=depth, 
                                                  subsample=sample, bootstrap_type='Bernoulli', verbose=False, 
                                                  random_seed=seed)                      
                    if alg == 'sglb' or alg == 'sglb-fixed':
                        model = CatBoostRegressor(loss_function='RMSEWithUncertainty',
                                                  learning_rate=lr, depth=depth, 
                                                  subsample=sample, 
                                                  bootstrap_type='Bernoulli', 
                                                  verbose=False, random_seed=seed, posterior_sampling=True)
                    
                    model.fit(train_pool, eval_set=validation_pool, use_best_model=False)
                    
                    # compute nll
                    results[d, l, s] = model.evals_result_['validation']['RMSEWithUncertainty'][-1]
                    
                    seed += 1 # update seed
        
        # get best parameters
        argmin = np.unravel_index(np.argmin(results), shape)
        depth = depths[argmin[0]]
        lr = lrs[argmin[1]]
        sample = samples[argmin[2]]
        
        current_params = {'depth': depth, 'lr': lr, 'sample': sample}
        params.append(current_params)
    
    return params
    
def tune_parameters_classification(dataset_name, alg='sgb'):

    # load and prepare data
    data_dir = os.path.join('datasets', dataset_name)
    train_file = os.path.join(data_dir, 'train')
    validation_file = os.path.join(data_dir, 'validation')
    test_file = os.path.join(data_dir, 'test')
    cd_file = os.path.join(data_dir, 'pool.cd')
    
    train_pool = Pool(data=train_file, column_description=cd_file)
    validation_pool = Pool(data=validation_file, column_description=cd_file)
    test_pool = Pool(data=test_file, column_description=cd_file)
    

    seed = 1000 # starting random seed for hyperparameter tuning
    
    # list of hyperparameters for grid search
    # we do not tune the number of trees, it is important for virtual ensembles
    depths = [3, 4, 5, 6] # tree depth
    lrs = [0.001, 0.01, 0.1] # learning rate 
    if alg == "sgb" or alg == "sglb": # by default, we tune sample rate
        samples = [0.25, 0.5, 0.75]
    if alg == "sgb-fixed": # sgb without sample rate tuning
        samples = [0.5]
    if alg == "sglb-fixed": # sglb without sample rate tuning
        samples = [1.0]
    shape = (len(depths), len(lrs), len(samples))

    # perform grid search
    results = np.zeros(shape)
    for d, depth in enumerate(depths):
        for l, lr in enumerate(lrs):
            for s, sample in enumerate(samples):
                if alg == 'sgb' or alg == 'sgb-fixed':
                    model = CatBoostClassifier(loss_function='Logloss',learning_rate=lr, depth=depth, subsample=sample, bootstrap_type='Bernoulli', verbose=False, random_seed=seed)                      
                if alg == 'sglb' or alg == 'sglb-fixed':
                    model = CatBoostClassifier(loss_function='Logloss',learning_rate=lr, depth=depth, subsample=sample, bootstrap_type='Bernoulli', verbose=False, random_seed=seed, posterior_sampling=True)
                    
                model.fit(train_pool, eval_set=validation_pool, use_best_model=False)
                    
                # compute nll
                results[d, l, s] = model.evals_result_['validation']['Logloss'][-1]
                    
                seed += 1 # update seed
        
    # get best parameters
    argmin = np.unravel_index(np.argmin(results), shape)
    depth = depths[argmin[0]]
    lr = lrs[argmin[1]]
    sample = samples[argmin[2]]
        
    params = {'depth': depth, 'lr': lr, 'sample': sample}
    
    return params
    
def generate_ensemble_regression(dataset_name, X, y, index_train, index_test, n_splits, params, alg="sgb", num_models=10):

    for fold in range(n_splits):

        # make catboost pools
        X_train_all, y_train_all, X_train, y_train, X_validation, y_validation, X_test, y_test = make_train_val_test(X, y, index_train, index_test, fold)
        full_train_pool = Pool(X_train_all, y_train_all)
        test_pool = Pool(X_test, y_test)
    
        # params contains optimal parameters for each fold
        depth = params[fold]['depth']
        lr = params[fold]['lr']
        sample = params[fold]['sample']

        seed = 10 * fold # fix different starting random seeds for all folds
        for i in range(num_models):
            if alg == 'sgb' or alg == 'sgb-fixed':
                model = CatBoostRegressor(loss_function='RMSEWithUncertainty', verbose=False, 
                                          learning_rate=lr, depth=depth, subsample=sample,
                                          bootstrap_type='Bernoulli', custom_metric='RMSE', 
                                          random_seed=seed)   
            if alg == 'sglb' or alg == 'sglb-fixed':
                model = CatBoostRegressor(loss_function='RMSEWithUncertainty', verbose=False, 
                                          learning_rate=lr, depth=depth, subsample=sample, 
                                          bootstrap_type='Bernoulli', posterior_sampling=True, 
                                          custom_metric='RMSE', random_seed=seed)
            seed += 1 # new seed for each ensemble element

            model.fit(full_train_pool, eval_set=test_pool, use_best_model=False) # do not use test pool for choosing best iteration
            model.save_model("results/models/" + dataset_name + "_" + alg + "_f" + str(fold) + "_" + str(i), format="cbm")
            

            
def generate_ensemble_classification(dataset_name, params, alg="sgb", num_models=10):

    # load and prepare data
    data_dir = os.path.join('datasets', dataset_name)
    full_train_file = os.path.join(data_dir, 'full_train')
    test_file = os.path.join(data_dir, 'test')
    cd_file = os.path.join(data_dir, 'pool.cd')
    
    full_train_pool = Pool(data=full_train_file, column_description=cd_file)
    test_pool = Pool(data=test_file, column_description=cd_file)

    # parameters
    depth = params['depth']
    lr = params['lr']
    sample = params['sample']
        
    seed = 0
    for i in range(num_models):
        if alg == 'sgb' or alg == 'sgb-fixed':
            model = CatBoostClassifier(loss_function='Logloss', verbose=False, 
                                       learning_rate=lr, depth=depth, subsample=sample,
                                       bootstrap_type='Bernoulli', custom_metric='ZeroOneLoss', 
                                       random_seed=seed)   
        if alg == 'sglb' or alg == 'sglb-fixed':
            model = CatBoostClassifier(loss_function='Logloss', verbose=False, 
                                       learning_rate=lr, depth=depth, subsample=sample, 
                                       bootstrap_type='Bernoulli', posterior_sampling=True, 
                                       custom_metric='ZeroOneLoss', random_seed=seed)
        seed += 1 # new seed for each ensemble element

        model.fit(full_train_pool, eval_set=test_pool, use_best_model=False) # do not use test pool for choosing best iteration
        model.save_model("results/models/" + dataset_name + "_" + alg + "_" + str(i), format="cbm")

def generate_rf_ensemble_regression(dataset_name, X, y, index_train, index_test, n_splits,
                                    num_models=10, n_estimators = 1000, compress=3, 
                                    n_jobs=-1, max_depth=10):
                                    
    for fold in range(n_splits):

        # make pools
        X_train_all, y_train_all, X_train, y_train, X_validation, y_validation, X_test, y_test = make_train_val_test(X, y, index_train, index_test, fold)

        seed = 10 * fold # fix different starting random seeds for all folds
        for i in range(num_models):
            model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, 
                                          n_jobs=n_jobs, random_state=seed)
            seed += 1 # new seed for each ensemble element

            model.fit(X_train_all, y_train_all) 
            joblib.dump(model, "results/models/" + dataset_name + "_" + "rf" + "_f" + str(fold) + "_" + str(i), compress=compress)
        
def generate_rf_ensemble_classification(dataset_name, num_models=10, 
                                        n_estimators = 1000, compress=3, 
                                        n_jobs=-1, max_depth=10):

    X_train, y_train, X_test, y_test, _ = process_classification_dataset(dataset_name)

    seed = 0
    for i in range(num_models):
    
        model = RandomForestClassifier(n_estimators=n_estimators, 
                                       max_depth=max_depth, n_jobs=n_jobs, 
                                       random_state=seed)

        seed += 1 # new seed for each ensemble element

        model.fit(X_train, y_train) 
        joblib.dump(model, "results/models/" + dataset_name + "_" 
                    + "rf" + "_" + str(i), compress=compress)
        