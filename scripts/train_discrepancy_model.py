"""
This script is used to train a discrepancy model by solving a MIP. It can be
run in PyCharm or called in the Terminal
"""
import os
import sys
import psutil
import dill

# add the default settings for this script to the top
settings = {
    'data_name': 'breastcancer',
    'fold_id': 'K05N01',
    'fold_num': 1,
    'delta': 0.2,
    'epsilon': 0.1,
    'time_limit': 300,
    'max_gap': 0.1,
    'regularization_param': 1e6,
    'random_seed': 109,
    'load_baseline': 1,
    'w_max': 25.0
    }

# parse arguments when script is called on cluster
ppid = os.getppid()  # get parent process id
process_type = psutil.Process(ppid).name()  # ex pycharm, bash
# if process_type in ('bash', 'zsh'):

import argparse
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument('--data_name', type=str, default = settings['data_name'], help='name of dataset being used')
parser.add_argument('--fold_id', type=str, default ='K05N01', help='Fold ID')
parser.add_argument('--fold_num', type=int, default=1, help='test fold number')
parser.add_argument('--delta', type=float, default=0.2, help='delta to evaluate discrepancy')
parser.add_argument('--epsilon', type=float,  default=0.1, help='level set for loss function')
parser.add_argument('--time_limit', type=int, default=600, help='time limit in seconds')
parser.add_argument('--max_gap', type=float, default=0.1, help='optimality gap')
parser.add_argument('--random_seed', type=int, default=109, help='random seed')
parser.add_argument('--load_baseline', type=int, default=1, help='set to 1 to load baseline')
parser.add_argument('--w_max', type=float, default=25.0, help='max coef bound in mip')
args, unknown = parser.parse_known_args()
settings.update(vars(args))

# set up paths – note that the next snippet will only work if:
# 1. this file is "in 'prm/scripts/"
# 2. this file is run from "prm/"
# add import cplex and exception



try:
    import prm.paths
except ImportError as e:
    repo_dir = Path(__file__).absolute().parent.parent
    sys.path.append(str(repo_dir))
    print(sys.path)



############ normal script starts here #################
from prm.paths import get_processed_data_file, get_results_file_disc, get_results_file_baseline
from prm.data import BinaryClassificationDataset
from prm.path_classifier import PathologicalClassifier
from prm.utils import compute_log_loss, print_log, predict_prob
from prm.disc_mip import DiscrepancyMIP


# import dataset and split into train/test
data = BinaryClassificationDataset.load(file = get_processed_data_file(**settings))
data.split(fold_id = settings['fold_id'], fold_num_test = settings['fold_num'])

results_file = get_results_file_disc(**settings)
baseline_file = get_results_file_baseline(**settings)

# load coefficients of baseline classifier from disk or train a new one
if settings['load_baseline'] and baseline_file.exists():
    with open(baseline_file, 'rb') as infile:
        out = dill.load(infile)
    settings['baseline_coefs'] = out['clf'].w
    settings['baseline_file'] = str(baseline_file)
else:
    clf = PathologicalClassifier(X = data.training.X, y = data.training.y, C = settings['regularization_param'], print_flag = False)
    settings['baseline_coefs'] = clf.w
    settings['baseline_file'] = ''

# get baseline loss and probabilities
settings['baseline_probs'] = predict_prob(X = data.training.X, w = settings['baseline_coefs'], add_intercept = True)
settings['baseline_loss'] = compute_log_loss(X = data.training.X, y = data.training.y, w = settings['baseline_coefs'], add_intercept = True)

# create discrepancy mip
dmip = DiscrepancyMIP(
        X = data.training.X,
        y = data.training.y,
        baseline_loss = settings['baseline_loss'],
        baseline_probs = settings['baseline_probs'],
        delta = settings['delta'],
        epsilon = settings['epsilon'],
        W_max = settings['w_max'],
        W_min = -1.0* settings['w_max']
        )

# todo: initialize MIP before solution

# solve discrepancy MINLP
stats = dmip.solve(time_limit = settings['time_limit'], max_gap = settings['max_gap'])

# print status of MIP for good bookkeeping
print_log(dmip)
try:
    assert dmip.check_mip_solution()
except Exception as e:
    print_log(e)

# save output to disk
output = dict(settings)
output.update({
    'coefs': dmip.coefficients,
    'loss': dmip.loss,
    'discrepancy': dmip.discrepancy,
    'results_file': get_results_file_disc(**settings),
    'mip_stats': stats,
    'mip_pool': dmip.loss_cb.stats['intermediate_solutions'],
    })

with open(results_file, 'wb') as outfile:
    dill.dump(output, outfile, protocol = dill.HIGHEST_PROTOCOL, recurse = True)


print_log(f'saved output to {results_file}')
