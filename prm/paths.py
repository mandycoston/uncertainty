from pathlib import Path

repo_dir = Path(__file__).absolute().parent.parent
data_dir = repo_dir / "data/"
results_dir = repo_dir / "results/"
reports_dir = repo_dir / "reports/"
templates_dir = repo_dir / "templates/"

# create directories to store results
results_dir.mkdir(exist_ok = True)

# create a directories to store reports
reports_dir.mkdir(exist_ok = True)

# Naming Functions
def get_processed_data_file(data_name,  **kwargs):
    """
    :param data_name: string containing name of the dataset
    :param kwargs: used to catch other args when unpacking dictionaries
                   this allows us to call this function as get_results_file_name(**settings)
    :return:
    """
    assert isinstance(data_name, str) and len(data_name) > 0
    f = data_dir / '{}_processed.pickle'.format(data_name)
    return f

def get_results_file_baseline(data_name, fold_id, fold_num, **kwargs):
    """
    returns file name for pickle files used to store the results of a training job (e.g., in `train_classifier`)
    :param data_name: string containing name of the dataset
    :param fold_id: string specifying fold_id of cross-validation indices
    :param fold_num: string specifying test fold in cross-validation_indices
    :param kwargs: used to catch other args when unpacking dictionaries
                   this allows us to call this function as get_results_file_name(**settings)
    :return:
    """
    assert isinstance(data_name, str) and len(data_name) > 0
    assert isinstance(fold_id, str) and len(fold_id) > 0
    assert isinstance(fold_num, int) and fold_num >= 0
    f = results_dir / '{}_{}_{}_baseline.results'.format(data_name, fold_id, '%02d' % fold_num)
    return f

def get_results_file_disc(data_name, fold_id, fold_num, delta, epsilon, w_max, **kwargs):
    """
    returns file name for pickle files used to store the results of a training job (e.g., in `train_classifier`)
    :param epsilon:
    :param delta:
    :param data_name: string containing name of the dataset
    :param fold_id: string specifying fold_id of cross-validation indices
    :param fold_num: string specifying test fold in cross-validation_indices
    :param kwargs: used to catch other args when unpacking dictionaries
                   this allows us to call this function as get_results_file_name(**settings)
    :return:
    """
    assert isinstance(data_name, str) and len(data_name) > 0
    assert isinstance(fold_id, str) and len(fold_id) > 0
    assert isinstance(fold_num, int) and fold_num >= 0
    assert isinstance(delta, float) and 0.0 <= delta <= 1.0
    assert isinstance(epsilon, float) and 0.0 <= epsilon <= 1.0

    #print delta and epsilon in 4 significant digits and strip the periods
    delta_str = ('%0.4f' % delta)[2:]
    epsilon_str = ('%0.4f' % epsilon)[2:]

    f = results_dir / '{}_{}_{}_{}_{}_wmax{}_disc.results'.format(data_name, fold_id, '%02d' % fold_num, delta_str, epsilon_str, int(w_max) )
    return f

def get_results_file_path(data_name, fold_id, fold_num, start_point_idx, **kwargs):
    """
    returns file name for pickle files used to store the results of a training job (e.g., in `train_classifier`)
    :param data_name: string containing name of the dataset
    :param fold_id: string specifying fold_id of cross-validation indices
    :param fold_num: string specifying test fold in cross-validation_indices
    :param kwargs: used to catch other args when unpacking dictionaries
                   this allows us to call this function as get_results_file_name(**settings)
    :return:
    """
    assert isinstance(data_name, str) and len(data_name) > 0
    assert isinstance(fold_id, str) and len(fold_id) > 0
    assert isinstance(fold_num, int) and fold_num >= 0
    f = results_dir / '{}_{}_{}_{}_path.results'.format(data_name, fold_id, '%02d' % fold_num, start_point_idx)
    return f
