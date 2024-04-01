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
    'regularization_param': 1e6,
    'random_seed': 109,
    'start_point_idx': 0,
    'load_baseline': 1
    }

# parse arguments when script is called on cluster
ppid = os.getppid()  # get parent process id
process_type = psutil.Process(ppid).name()  # ex pycharm, bash
if process_type in ('bash', 'zsh'):

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
    args, unknown = parser.parse_known_args()
    settings.update(vars(args))

    # set up paths – note that the next snippet will only work if:
    # 1. this file is "in 'prm/scripts/"
    # 2. this file is run from "prm/"
    try:
        import prm.paths
    except ImportError as e:
        repo_dir = Path(__file__).absolute().parent.parent
        sys.path.append(str(repo_dir))
        print(sys.path)

############ normal script starts here #################
from prm.paths import get_processed_data_file, get_results_file_path, get_results_file_baseline
import pandas as pd
import numpy as np
from prm.utils import compute_log_loss, print_log, predict_prob
from prm.path_classifier import PathologicalClassifier
from prm.data import BinaryClassificationDataset


results_file = get_results_file_path(**settings)

# import the data and then divide into test and train datasets
data = BinaryClassificationDataset.load(file = get_processed_data_file(**settings))
data.split(fold_id = settings['fold_id'], fold_num_test = settings['fold_num'])

# Get Baseline Probabilities
baseline_file = get_results_file_baseline(**settings)

# train the baseline classifier and save results for discrepancy algorithm to use
if settings['load_baseline'] and baseline_file.exists():
    with open(baseline_file, 'rb') as infile:
        out = dill.load(infile)
    plf = out['clf']
    settings['baseline_coefs'] = out['clf'].w
    settings['baseline_file'] = str(baseline_file)
else:
    plf = PathologicalClassifier(X = data.training.X, y = data.training.y, C = settings['regularization_param'], print_flag = False)
    settings['baseline_coefs'] = plf.w
    settings['baseline_file'] = ''

# get baseline loss and probabilities
settings['baseline_probs'] = predict_prob(X = data.training.X, w = settings['baseline_coefs'], add_intercept = True)
settings['baseline_loss'] = compute_log_loss(X = data.training.X, y = data.training.y, w = settings['baseline_coefs'], add_intercept = True)

if results_file.exists():
    # If the results file already exists, start pathological algorithm from the last index ran
    # Load the existing results file, then extract the last index "i"
    try:
        with open(results_file, 'rb') as infile:
            results = dill.load(infile)
    except ValueError:
        import pickle5 as pickle

        with open(results_file, 'rb') as infile:
            results = pickle.load(infile)

    previous_results = pd.DataFrame(results['stats_df'])

    previous_coefs = []
    for key, value in results['all_coefs'].items():
        previous_coefs.append(value)

    previous_results['coefs'] = previous_coefs
    all_results = previous_results.to_dict('records')

    n_pts_done = np.max(previous_results['i']) + 1
    n_pts_total = data.training.X.shape[0]

    # if there are still points to step through, cut baseline probability array to reflect proper index
    if (n_pts_done < n_pts_total):
        last_idx = np.max(previous_results['i'])
        print_log("Beginning pathological algorithm from index: {}".format(last_idx))
        settings['baseline_probs'] = settings['baseline_probs'][last_idx:]
        settings['start_point_idx'] = last_idx
        results_file = get_results_file_path(**settings)
    all_results = []
else:
    all_results = []




for i, p_baseline in enumerate(settings['baseline_probs']):
    path_output = plf.fit_path(xt = data.training.X[i])

    for d in path_output:
        d["i"] = i
        all_results.append(d)

    # save data as we go by keep updating results file
    # merge all results into a data frame
    df = pd.DataFrame(data = all_results)
    df['model_id'] = ['M%07d' % i for i in df.index.to_list()]

    # store stats in data.frame
    stats_df = df.drop(columns = ['coefs'])
    stats_df = stats_df.rename(columns={"pt": "threshold_probability"})[["i", "baseline_probability", "threshold_probability", "predicted_probability", "model_id"]]
    # store coefficients in dictionary
    all_coefs = pd.DataFrame(df['coefs'])
    all_coefs.index = df['model_id']
    all_coefs = all_coefs.to_dict()
    all_coefs = all_coefs['coefs']

    # save output in report
    output = dict(settings)
    output.update({
        'stats_df': stats_df,
        'all_coefs': all_coefs,
        'baseline_coefs': settings['baseline_coefs'],
        'baseline_loss': settings['baseline_loss'],
        'output_filename': results_file,
        'data_filename': get_processed_data_file(**settings)
        })

    # save to disk
    with open(results_file, 'wb') as outfile:
        dill.dump(output, outfile, protocol = dill.HIGHEST_PROTOCOL, recurse = True)

print_log(f'saved output to {results_file}')
