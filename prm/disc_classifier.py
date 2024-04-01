import numpy as np
import cvxpy as cp
from prm.utils import log_loss_value, convert_Xy_to_Z





class DiscrepancyClassifier(object):

    default_print_flag = True
    default_delta = 0.05
    default_epsilon = 0.05

    def __init__(self, X, y, baseline_probabilities, **kwargs):

        #todo: add baseline probabilities
        assert X.ndim == 2
        assert X.shape[0] > 0
        assert np.isfinite(X).all()
        assert np.isfinite(y).all()
        assert np.isin(y, (-1, 1)).all()

        # print flag
        self._print_flag = DiscrepancyClassifier.default_print_flag
        self.print_flag = kwargs.get('print_flag', DiscrepancyClassifier.default_print_flag)

        # classifier parameters
        self.delta = kwargs.get('delta', DiscrepancyClassifier.default_delta)
        self.epsilon = kwargs.get('epsilon', DiscrepancyClassifier.default_epsilon)

        # dataset properties
        self.n = X.shape[0]
        self.d = X.shape[1]
        self.X = X
        self.y = y
        self.Z = convert_Xy_to_Z(X, y)

        # coefficient indices
        self._w = cp.Variable(self.d + 1)
        self.intercept_idx = 0
        self.coefficient_idx = np.arange(1, self.d + 1)

        # setup ERM probblem
        self.problem = self._setup_problem_cvx(baseline_probabilities, epsilon = self.epsilon, delta = self.delta)

    @property
    def epsilon(self):
        """
        :return: suboptimaliity ration
        """
        return self._epsilon

    @epsilon.setter
    def epsilon(self, value):
        assert np.isfinite(value)
        if np.greater(value, 0.0):
            self._epsilon = value

    @property
    def delta(self):
        """
        :return: suboptimaliity ration
        """
        return self._delta


    # main methods
    def fit(self, delta, epsilon):
        """
        :param xt: feature vector with d features used for a prediction constraint
        :param pt: threshold probability for a prediction constraint
        :return:
        """
        # todo: write this function
        # ERM Variables
        self.problem = self._setup_problem_cvx(epsilon = epsilon, delta = delta)
        self.solution = self._solve_problem_cvx(problem = self.problem, warm_start = False)

        # Set Coefficients


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
