import numpy as np
import cvxpy as cp
from prm.utils import log_loss_value
# from dev.debug import ipsh


class PathologicalClassifier(object):

    default_print_flag = True
    default_n_values = 20

    def __init__(self, X, y, C, **kwargs):

        #todo: sanity checks on Xf, y, C
        assert X.ndim == 2
        assert X.shape[0] > 0
        assert np.isfinite(X).all()
        assert np.isfinite(y).all()
        assert np.isin(y, (-1, 1)).all()
        assert np.greater_equal(C, 0.0)
        # todo: check that X does not contain a column of ones in intercept idx
        self._C = float(C)

        # set print flag
        self._print_flag = PathologicalClassifier.default_print_flag
        self.print_flag = kwargs.get('print_flag', PathologicalClassifier.default_print_flag)

        # attach dataset to classifier
        self.n = X.shape[0]
        self.d = X.shape[1]
        self.X = X
        self.y = y

        # compute new terms
        self.intercept_idx = 0
        self.coefficient_idx = np.arange(1, self.d + 1)

        # setup auxiliary for prediction
        self.Xf = np.insert(X, self.intercept_idx, 1.0, axis= 1)
        self.Z = self.y[:, np.newaxis] * self.Xf

        # setup elements of ERM problem

        # ERM Variables
        self._w = cp.Variable(self.d + 1)

        # ERM Objective
        self._objective = cp.sum(cp.logistic(-self.Z @ self._w)) / float(self.n)
        if np.greater(self._C, 0.0):
            self._objective += (cp.norm(self._w[self.coefficient_idx], 2) ** 2) / self._C
        self._objective = cp.Minimize(self._objective)

        # Fit a Baseline Classifier
        erm_baseline = self._setup_erm_problem(xt = None)
        sol_baseline = self._solve_cvx_problem(problem = erm_baseline, warm_start = False)
        self.print_flag = False

        # from prm.debug import ipsh
        # ipsh()
        self._w_baseline = self.w

    # helper functions
    def _setup_erm_problem(self, xt = None, pt = None):
        """
        :param xt:
        :param pt:
        :return:
        """

        # initialize list that will contain the elements of the optimization problem
        problem_args = [self._objective]

        # add prediction constraint to problem args
        if (xt is not None) and (pt is not None):
            self._logit_value.value = np.log(pt) - np.log(1.0 - pt)
            if np.less_equal(pt, self.baseline_probability):
                constraint = [(xt @ self._w) <= self._logit_value]
            else:
                constraint = [(xt @ self._w) >= self._logit_value]
            problem_args.append(constraint)

        # initialize problem
        problem = cp.Problem(*problem_args)

        return problem

    def _solve_cvx_problem(self, problem, warm_start = True, debug = True):
        """
        solves an optimization problem using CVX
        :param problem:
        :param warm_start:
        :param debug:
        :return:
        """

        if debug:
            try:
                problem.solve(warm_start = warm_start)
                # todo: catch the exact type of Exception that occurs
            except Exception:
                print("""******* Solver failed!""")
                print("Point is {}, baseline probability is {}, threshold probability is {}".format(self.xt,
                                                                                                    self.baseline_probability,
                                                                                                    self.pt))
                # self._print_solution_status(self,  )
        else:
            problem.solve(warm_start = warm_start)

        if self.print_flag:
            print("CVXPY solver status:", problem.status)
            # todo: only print these when the problem is feasible
            print("objective value: ", problem.value)
            print("solution: ", problem.solution)

        return problem.solution

    def compute_baseline_probability(self, xt):
        assert len(xt) == self.d
        xp_baseline = np.insert(arr = xt, obj = self.intercept_idx, values = 1.0)
        sp = self._w_baseline.dot(xp_baseline)
        if np.greater_equal(sp, 0.0):
            self.baseline_probability = 1.0 / (1.0 + np.exp(-sp))
        else:
            self.baseline_probability = np.exp(sp) / (1.0 + np.exp(sp))

    @staticmethod
    def probspace(n = 10, p_min = 0.01, p_max = None):
        """
        construct an array of n equally-spaced floats from 0 to 1 that represent
        valid risk predictions from a classifier. Since classifiers cannot assign
        risks of 0 and 1, end points of the array are small values near 0, 1

        n: # of values needed
        p_min: smallest probability value, i.e., the value of the left endpoint
        p_max: largest probability value, i.e., the value of the left endpoint
               set as 1 - p_min if it isn't specified

        :return: array of probability values

        """

        assert isinstance(n, int) and n >= 1
        assert isinstance(p_min, float)
        assert np.greater(p_min, 0.0) and np.less(p_min, 0.5)
        p_max = 1.0 - p_min if p_max is None else p_max
        assert np.greater(p_max, 0.5) and np.less(p_max, 1.0)

        arr = np.linspace(0.0, 1.0, num = n + 1, endpoint = True)

        # double check that the array is still sorted
        assert p_min < arr[1]
        # assert p_max > arr[-2]
        arr[np.flatnonzero(arr == 0.0)] = p_min
        arr[np.flatnonzero(arr == 1.0)] = 1.0 - p_min

        return arr

    # main methods
    def fit(self, xt = None, pt = None):
        """
        :param xt: feature vector with d features used for a prediction constraint
        :param pt: threshold probability for a prediction constraint
        :return:
        """
        # todo: write this function
        # ERM Variables
        self._w = cp.Variable(self.d + 1)

        # ERM Objective
        self._objective = cp.sum(cp.logistic(-self.Z @ self._w)) / float(self.n)
        if np.greater(self._C, 0.0):
            self._objective += (cp.norm(self._w[self.coefficient_idx], 2) ** 2) / self._C
        self._objective = cp.Minimize(self._objective)

        # Fit a Baseline Classifier
        erm = self._setup_erm_problem(xt=None)
        problem = self._solve_cvx_problem(problem = erm, warm_start = False)

    def fit_path(self, xt, n = 20, p_min = 0.01, p_max = None):
        """
        fit a set of n classifiers that are required to assign a specific
        risk prediction to a point with features xt

        xt: feature vector with d features used for a prediction constraint
        n: number of points on the path
        p_min: smallest probability value, i.e., the value of the left endpoint
        p_max: largest probability value, i.e., the value of the left endpoint
               set as 1 - p_min if it isn't specified

        :return: list of n models
        """

        assert xt is not None
        self.xt = xt  # for debugging

        self.compute_baseline_probability(xt)
        pt_baseline = self.baseline_probability
        xt_f = np.insert(arr=xt, obj = self.intercept_idx, values=1.0)

        # build probability path
        path_probabilities = self.probspace(n = n, p_min = p_min)
        path_left = path_probabilities[np.less_equal(path_probabilities, pt_baseline)]
        path_right = path_probabilities[np.greater(path_probabilities, pt_baseline)]

        # setup an ERM problem parameterized by the left-hand value of the prediction constraint
        logit_pt = cp.Parameter()
        erm_left = cp.Problem(self._objective, [(xt_f @ self._w) <= logit_pt])
        erm_right = cp.Problem(self._objective, [(xt_f @ self._w) >= logit_pt])

        # initialize output
        path_output = []
        output_infeasible = {
            'baseline_probability': pt_baseline,
            'pt': float('nan'),
            'predicted_probability': float('nan'),
            'coefs': np.repeat(np.nan, self.d+1),
            }

        for pt in path_left:
            logit_pt.value = np.log(pt) - np.log(1.0 - pt)
            self.pt = pt # for debugging
            self._solve_cvx_problem(erm_left, warm_start = True)

            # process output
            out = dict(output_infeasible)
            out['pt'] = pt
            if self.has_solution:
                out.update({'coefs': self.w, 'predicted_probability': self.predict_proba(xt).astype(np.float64).item() })
            path_output.append(out)

        for pt in path_right:
            logit_pt.value = np.log(pt) - np.log(1.0 - pt)
            self.pt = pt # for debugging
            self._solve_cvx_problem(erm_right, warm_start = True)

            # process output
            out = dict(output_infeasible)
            out['pt'] = pt
            if self.has_solution:
                out.update({'coefs': self.w, 'predicted_probability': self.predict_proba(xt).astype(np.float64).item()})
            path_output.append(out)

        return path_output

    #### ERM Properties ####
    @property
    def print_flag(self):
        return self._print_flag

    @print_flag.setter
    def print_flag(self, value):
        assert isinstance(value, bool)
        self._print_flag = value

    @property
    def objective(self):
        return self._objective

    @property
    def has_solution(self):
        return self._w.value is not None

    @property
    def w(self):
        if self.has_solution:
            return self._w.value
        else:
            return np.repeat(np.nan, self.d + 1)

    #### Classifier Properties ###
    @property
    def loss(self):
        """return logistic loss value of fitted coefficient vector over dataset"""
        return self.get_loss(X = self.X, y = self.y)

    @property
    def intercept(self):
        if self.has_solution:
            return self._w[self.intercept_idx].value
        else:
            return np.nan

    @property
    def coefs(self):
        if self.has_solution:
            return self._w[self.coefficient_idx].value
        else:
            return np.repeat(np.nan, self.d)

    #### prediction API ####
    def decision_function(self, X):
        """
        returns confidence score
        :param X:
        :return: vector of confidence scores for each row in X
        """
        scores = X.dot(self.coefs) + self.intercept
        return scores

    def predict_proba(self, X):
        """
        predicts probability of positive class
        :param X: feature matrix with d columns features
        :return: vector of predicted probabilities for each row in X
        """
        scores = self.decision_function(X)
        probs = np.zeros_like(scores)
        pos_idx = np.greater_equal(scores, 0)
        neg_idx = np.logical_not(pos_idx)
        probs[pos_idx] = 1.0 / (1.0 + np.exp(-scores[pos_idx]))
        probs[neg_idx] = np.exp(scores[neg_idx]) / (1.0 + np.exp(scores[neg_idx]))
        return probs

    def predict(self, X):
        """
        predicts label for each row in X
        :param X: feature matrix with d columns features
        :return: vector of confidence scores for each row in X
        """
        probs = self.predict_proba(X)
        yhat = np.greater(probs, 0.5)
        return yhat

    # todo: make this into a standalone method (maybe move to prm/utils.py) -  there's no need to include anymore
    def get_loss(self, X, y, w = None):
        """
        :param X: feature matrix (n rows x d columns)
        :param y: label vector with n elements
        :param w: coefficient vector with (d + 1) elements
        :return: logistic loss of feature matrix over coefficient vector
        """
        if w is None:
            w = self.w
        assert X.shape[1] == self.d
        assert np.isin(y, (-1, 1)).all()
        Xf = np.insert(X, obj = self.intercept_idx, values = 1.0, axis = 1)
        #ipsh()
        Z = np.multiply(y[:, None], Xf)
        return log_loss_value(Z, w)
