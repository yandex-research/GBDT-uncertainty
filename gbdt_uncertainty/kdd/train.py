import context
import argparse
import os
import sys

from gbdt_uncertainty.data import load_KDD_dataset
from gbdt_uncertainty.ensemble import ClassificationEnsemble

parser = argparse.ArgumentParser(description='Train a confidence score estimation BiLSTM model.')
parser.add_argument('data_path', type=str,
                    help='absolute path to data.')
parser.add_argument('save_path', type=str,
                    help='absolute path where to save model.')
parser.add_argument('--lr', type=float, default=0.1,
                    help='Batch size for training.')
parser.add_argument('--n_models', type=int, default=10,
                    help='Batch size for training.')
parser.add_argument('--n_iters', type=int, default=50,
                    help='Specify which GPUs to to run on.')
parser.add_argument('--tree_depth', type=int, default=6,
                    help='How often to save a checkpoint.')
parser.add_argument('--random_strength', type=int, default=None,
                    help='Batch size for training.')
parser.add_argument('--border_count', type=int, default=None,
                    help='Batch size for training.')
parser.add_argument('--max_ctr_complexity', type=int, default=None,
                    help='Batch size for training.')
parser.add_argument('--use_best_model', action='store_true',
                    help='Specify which GPUs to to run on.')
parser.add_argument('--posterior_sampling', action='store_true',
                    help='Specify which GPUs to to run on.')
parser.add_argument('--seed', type=int, default=0,
                    help='Batch size for training.')
parser.add_argument('--gpu', type=int, action='append',
                    help='Specify which GPUs to to run on.')


def main():
    args = parser.parse_args()
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/train_gbdt.cmd', 'a') as f:
        f.write(' '.join(sys.argv) + '\n')
        f.write('--------------------------------\n')

    if args.gpu is None:
        args.gpu = []
    assert len(args.gpu) < 2
    if len(args.gpu) > 0:
        task_type = 'GPU'
        devices = f'args.gpu[0]'
    else:
        task_type = None
        devices = None

    train, test, ood = load_KDD_dataset(args.data_path)

    ens = ClassificationEnsemble(esize=args.n_models,
                                 iterations=args.n_iters,
                                 lr=args.lr,
                                 depth=args.tree_depth,
                                 seed=args.seed,
                                 verbose=True,
                                 posterior_sampling=args.posterior_sampling,
                                 random_strength=args.random_strength,
                                 max_ctr_complexity=args.max_ctr_complexity,
                                 task_type=task_type,
                                 devices=devices)

    ens.fit(train, eval_set=test, save_path=args.save_path)


if __name__ == "__main__":
    main()
