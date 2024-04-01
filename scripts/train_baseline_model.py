"""
This script is used to train a discrepancy model by solving a MIP. It can be
run in PyCharm or called in the Terminal
"""
import os
import sys
import psutil
import warnings
import dill

# add the default settings for this script to the top
settings = {
    'data_name': 'breastcancer',
    'fold_id': 'K05N01',
    'fold_num': 1,
    'regularization_param': 1e6,
    'random_seed': 109
    }

# parse arguments when script is called on cluster
ppid = os.getppid()  # get parent process id
process_type = psutil.Process(ppid).name()  # ex pycharm, bash
if process_type in ('bash', 'zsh'):

    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', type=str, default='heart', help='name of dataset being used')
    parser.add_argument('--fold_id', type=str, default='K05N01', help='Fold ID')
    parser.add_argument('--fold_num', type=int, default=1, help='test fold number')
    parser.add_argument('--regularization_param', type=float, default=1e6, help='regularization parameter')
    args = parser.parse_known_args()
    settings.update(vars(args))

    # set up paths - this assumes that this file is "in prm/scripts"
    try:
        import prm.paths
    except ImportError as e:
        repo_dir = Path(__file__).absolute().parent.parent
        repo_name = repo_dir.name
        code_dir = repo_dir / repo_dir.name
        sys.path.append(code_dir)
        print(sys.path)

############ normal script starts here #################
from prm.paths import get_processed_data_file, get_results_file_baseline
from prm.data import BinaryClassificationDataset
from prm.path_classifier import PathologicalClassifier

# import the data and then divide into test and train datasets
data = BinaryClassificationDataset.load(file = get_processed_data_file(**settings))
data.split(fold_id = settings['fold_id'], fold_num_test = settings['fold_num'])
plf = PathologicalClassifier(X = data.training.X, y = data.training.y, C = settings['regularization_param'], print_flag = False)

output = {
    'clf': plf,
    'coefs': plf.w,
    'loss': plf.get_loss(data.training.X, data.training.y),
    }
output.update(settings)

# save to disk
results_file = get_results_file_baseline(**settings)
with open(results_file, 'wb') as outfile:
    dill.dump(output, outfile, protocol = dill.HIGHEST_PROTOCOL, recurse = True)
