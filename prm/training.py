import dill
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from prm.path_classifier import PathologicalClassifier
from prm.calibration import compute_calibration
from prm.data import BinaryClassificationDataset
from prm.data import load_data_from_csv
from prm.paths import data_dir, results_dir

default_inputs = {
    'delta_increment': 0.01,
    'regularization_param': 1.0/1e-6,
    'n_values': 20,
    'p_eps': 1e-2,
    'random_seed': 109
    }

def train_pathological_models_overview(**input):

    # update input
    input.update(default_inputs)

    data_filename = data_dir / '{}_processed.pickle'.format(input['data_name'])
    output_filename = results_dir / ('%s_prm.results' % input['data_name'])

    # import the data and then divide into test and train datasets
    data = BinaryClassificationDataset.load(file = data_filename)
    data.split(fold_id = input['fold_id'], fold_num_test = input['fold_num'])

    plf = PathologicalClassifier(X=data.training.X, y=data.training.y, C=input['regularization_param'])

    # Get Baseline Probabilities
    baseline_coefs = plf.w
    baseline_probs = plf.predict_proba(X=data.training.X)
    baseline_loss = plf.get_loss(data.training.X, data.training.y)

    # Compare probs to sklearn python function
    # sklearn_probs = plf.compare_to_sklearn(X_train, data['y_train'], baseline_probs, dif_threshold = 0.01)

    # Try to load results file
    # todo: turn the loading script into a function and add it to prm/utils.py
    if output_filename.exists():
        # load from dataset
        try:
            with open(output_filename, 'rb') as infile:
                results = dill.load(infile)
        except ValueError:
            import pickle5 as pickle

            with open(output_filename, 'rb') as infile:
                results = pickle.load(infile)

        previous_results = pd.DataFrame(results['stats_df'])

        previous_coefs = []
        for key, value in results['all_coefs'].items():
            previous_coefs.append(value)

        previous_results['coefs'] = previous_coefs

        all_results = previous_results.to_dict('records')

        n_pts_done = np.max(previous_results['i']) + 1
        n_pts_total = data.training.X.shape[0]

        if (n_pts_done < n_pts_total):
            last_idx = np.max(previous_results['i'])
            baseline_probs = baseline_probs[last_idx:]
            print("Starting training at idx: {}".format(last_idx))
            output_filename = results_dir / ('%s_prm%d.results' % (input['data_name'], last_idx))
        all_results = []
    else:
        all_results = []

    # baseline_probs = baseline_probs[0:3]
    for i, p_baseline in enumerate(baseline_probs):
        path_output = plf.fit_path(xt=data.training.X[i])

        # todo: add missing fields to path_output
        # note that you can convert too a pd dataframe
        for d in path_output:
            d["i"] = i
            all_results.append(d)

        # pt_idx = i, X_test = data.test.X, y_test = data.test.y

        # save data as we go by keep updating results file
        # merge all results into a data frame
        df = pd.DataFrame(data=all_results)
        df['model_id'] = ['M%07d' % i for i in df.index.to_list()]

        # store stats in data.frame
        stats_df = df.drop(columns=['coefs'])
        stats_df = stats_df.rename(columns={"pt": "threshold_probability"})[
            ["i", "baseline_probability", "threshold_probability", "predicted_probability", "model_id"]]
        # store coefficients in dictionary
        all_coefs = pd.DataFrame(df['coefs'])
        all_coefs.index = df['model_id']
        all_coefs = all_coefs.to_dict()
        all_coefs = all_coefs['coefs']

        # save output in report
        output = dict(input)
        output.update({
            'stats_df': stats_df,
            'all_coefs': all_coefs,
            'baseline_coefs': baseline_coefs,
            'baseline_loss': baseline_loss,
            'output_filename': output_filename,
            'data_filename': data_filename
        })

        # save to disk
        with open(output_filename, 'wb') as outfile:
            dill.dump(output, outfile, protocol=dill.HIGHEST_PROTOCOL, recurse=True)

    return output
