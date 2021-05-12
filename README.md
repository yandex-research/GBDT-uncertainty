# Uncertainty in Gradient Boosting via Ensembles

This is a supplementary code for our paper:

[Uncertainty in Gradient Boosting via Ensembles](https://openreview.net/pdf?id=1Jv6b0Zq3qi)
<br>
Andrey Malinin, Liudmila Prokhorenkova, Aleksei Ustimenko (ICLR 2021)

Datasets can be found here: https://drive.google.com/file/d/1btIDCqubKZsPNcB7KKJmFcNj8LQZa-4w/view?usp=sharing

#### Training models

```python train_models.py regression 1```

First argument options: ```regression```, ```classification```, ```regression_rf```, ```classification_rf```
<br>
Second argument (for CatBoost only): 0 or 1 indicates whether to tune hyperparameters (or use already obtained ones)

#### Aggregating results and getting tables in latex format.

Regression:

```python aggregate_results_regression.py prr_auc``` 

Options: ```std```, ```nll_rmse```, ```prr_auc```, ```rf_rmse```, ```rf_prr_auc```

Classification:

```python aggregate_results_classification.py prr_auc```

Options: ```nll_error```, ```prr_auc```, ```rf_nll_error```, ```rf_prr_auc```

#### Synthetic experiments

```synthetic_regression.ipynb```

```synthetic_classification.ipynb```

#### Additional experiment on KDD-99 Intrusion Detection dataset
(not included in the paper)

```gbdt_uncertainty/kdd/kdd.sh```
