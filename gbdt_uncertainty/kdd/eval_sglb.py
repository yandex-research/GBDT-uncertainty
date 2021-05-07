import context
import argparse
import os
import sys
import numpy as np

from gbdt_uncertainty.data import load_KDD_dataset
from gbdt_uncertainty.ensemble import ClassificationEnsembleSGLB
from gbdt_uncertainty.uncertainty import ensemble_uncertainties_classification as euc
from gbdt_uncertainty.assessment import ood_detect
from scipy.special import softmax
from gbdt_uncertainty.uncertainty import entropy

parser = argparse.ArgumentParser(description='Train a confidence score estimation BiLSTM model.')
parser.add_argument('data_path', type=str,
                    help='absolute path to training data.')
parser.add_argument('models', type=str,
                    help='absolute directory path where to save model and associated data.')
parser.add_argument('--n_models', type=int, default=10,
                    help='Batch size for training.')
parser.add_argument('--virtual', action='store_true',
                    help='Specify which GPUs to to run on.')

def main():
    args = parser.parse_args()
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/eval_gbdt_sglb.cmd', 'a') as f:
        f.write(' '.join(sys.argv) + '\n')
        f.write('--------------------------------\n')

    _, test, ood = load_KDD_dataset(args.data_path, eval=True)

    ens = ClassificationEnsembleSGLB(esize=args.n_models,
                                 verbose=True,
                                 load_path=args.models)

    class_map = {}
    for i, c in enumerate(ens.ensemble[0].classes_):
        class_map[c] = i

    test_labels = [class_map[l] for l in test.get_label()]


    if args.virtual:
        #for m in range(args.n_models):
        id_probs = ens.ensemble[0].virtual_ensembles_predict(test, prediction_type='VirtEnsembles', virtual_ensembles_count=args.n_models)
        id_probs = softmax(id_probs, axis=2).transpose([1,0,2])
        ood_probs = ens.ensemble[0].virtual_ensembles_predict(ood, prediction_type='VirtEnsembles',
                                                     virtual_ensembles_count=args.n_models)
        ood_probs = softmax(ood_probs, axis=2).transpose([1,0,2])
        # MUST GET SOFTMAX
        print(id_probs.shape, ood_probs.shape)



    else:
        id_probs = ens.predict(test)
        ood_probs = ens.predict(ood)


    ens_probs = np.mean(id_probs, axis=0)
    ens_preds = np.argmax(ens_probs, axis=1)

    assert len(ens_preds) == len(test_labels)
    error = np.mean(np.asarray(ens_preds != test_labels, np.float)) * 100

    ind_errors = []
    for i in range(args.n_models):
        preds = np.argmax(id_probs[i], axis=1)
        assert preds.shape == ens_preds.shape
        ind_errors.append(np.mean(np.asarray(preds != test_labels, np.float)) *100)

    print(f"Singe Model Error rate: {np.round(np.mean(ind_errors))} +/- {np.round(2*np.std(ind_errors), 2)}")
    print(f"Ensemble Error rate: {np.round(error, 1)}")

    id_uncertainties = euc(id_probs)
    ood_uncertaintyies = euc(ood_probs)

    id_labels = np.zeros(id_probs.shape[1])
    ood_labels = np.ones(ood_probs.shape[1])
    domain_labels = np.concatenate([id_labels, ood_labels])

    for measure in id_uncertainties.keys():
        auc = ood_detect(domain_labels, id_uncertainties[measure], ood_uncertaintyies[measure], mode='ROC')
        print(f"{measure}: {auc}")

    # Eval single models
    id_probs = id_probs.transpose([1,2,0])
    ood_probs = ood_probs.transpose([1,2,0])

    id_entropy = entropy(id_probs)
    ood_entropy = entropy(ood_probs)

    aucs = []
    for m in range(args.n_models):
        auc = ood_detect(domain_labels, id_entropy[:,m], ood_entropy[:,m], mode='ROC')
        aucs.append(auc)
    print(f"Single Model {np.round(np.mean(aucs),4)} +/- {2*np.round(np.std(aucs),4)}")

if __name__ == "__main__":
    main()
