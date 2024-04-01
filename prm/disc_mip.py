import numpy as np
import warnings
import sys
try:
    import cplex
    print('---cplex loaded without error')

from cplex import Cplex, SparsePair
from cplex.callbacks import LazyConstraintCallback
from scipy.special import logit
from prm.utils import convert_Xy_to_Z, log_loss_value, predict_prob, print_log
from prm.cpx_utils import StatsCallback, get_mip_stats, set_mip_time_limit, set_mip_max_gap, has_solution

# checking functions
is_geq_or_close = lambda a, b: np.logical_or(np.greater_equal(a, b), np.isclose(a, b)).astype(int)
is_leq_or_close = lambda a, b: np.logical_or(np.less_equal(a, b), np.isclose(a, b)).astype(int)

def compute_log_loss(X, y, w, add_intercept = True):
    """
    :param X: feature matrix (n rows x d columns)
    :param y: label vector with n elements
    :param w: coefficient vector with (d + 1) elements
    :return: logistic loss of feature matrix over coefficient vector
    """
    Z = convert_Xy_to_Z(X, y, add_intercept = add_intercept)
    return log_loss_value(Z, w)


def compute_discrepancy_indicators(X, y, w, p0, delta, add_intercept = True):
    """
    :param p0: baseline probabilities (n rows)
    :param y: label vector with n elements (n rows)
    :param w: coefficient vector with (d + 1) elements
    :return: logistic loss of feature matrix over coefficient vector
    """
    assert X.shape[0] == len(y)
    assert X.shape[0] == len(p0)
    assert np.isin(y, (-1, 1)).all()
    assert np.isfinite(X).all()
    assert np.isfinite(y).all()
    assert np.isfinite(w).all()
    assert np.isfinite(p0).all()
    assert np.isfinite(delta)
    p = predict_prob(X = X, w = w, add_intercept = add_intercept)
    dp = np.abs(p0 - p)
    return is_geq_or_close(dp, delta)


class LossCallback(LazyConstraintCallback):
    """
    This callback has to be initialized after construnction with initialize().
    LossCallback is called when CPLEX finds an integer feasible solution. By default, it will add a cut at this
    solution to improve the cutting-plane approximation of the loss function. The cut is added as a 'lazy' constraint
    into the surrogate LP so that it is evaluated only when necessary.

    Optimality Cut:
    --------------
    L ≥ loss_value^k + sum(loss_gradient^k[j] * (w[j] - w^k[j]))
    L ≥ loss_value^k + sum(loss_gradient^k[j] * w[j]) - sum(loss_gradient^k[j] * w^k[j])
    L + loss_gradient^k[0] * w[0] + ....  loss_gradient^k[d] * w[d]  ≥ loss_value^k - <loss_gradient^k, w^k>

    Feasibility Cut:
    ----------------
    0 ≥ loss_value^k + sum(loss_gradient^k[j] * (w[j] - w^k[j]))
    0 ≥ loss_value^k + sum(loss_gradient^k[j] * w[j]) - sum(loss_gradient^k[j] * w^k[j])
    0*L + loss_gradient^k[0] * w[0] + ....  loss_gradient^k[d] * w[d]  ≥ loss_value^k - <loss_gradient^k, w^k>

    """

    def initialize(self, X, y, loss_idx, coef_idx, threshold_loss, purge_cuts = False, **kwargs):
        """
        :param X: feature matrix
        :param y: label vector
        :param loss_index: index of the loss variable in the MIP
        :param coef_index: index of the coefficient variables in the MIP
        :param purge_cuts: set to True to drop cutting planes that are not used.
        """

        assert isinstance(loss_idx, list)
        assert isinstance(coef_idx, list)
        assert isinstance(purge_cuts, bool)
        assert len(coef_idx) == X.shape[1] + 1
        threshold_loss = float(threshold_loss)
        assert np.greater(threshold_loss, 0.0)
        self.Z = convert_Xy_to_Z(X, y)

        # indices
        self.cut_idx = loss_idx + coef_idx
        self.coef_idx = coef_idx

        # initial cutting planes
        self.initial_cuts = kwargs.get('initial_cuts')
        self.threshold_loss = threshold_loss

        # purge loss cuts
        self.purge_cuts = self.use_constraint.purge if purge_cuts else self.use_constraint.force

        # initialize dictionary of stats
        self.stats = {
            'n_feasibility_cuts': 0,
            'n_optimality_cuts': 0,
            'intermediate_solutions': [],
            'lowerbound': float('inf'),
            'incumbent': np.repeat(float('nan'), len(self.coef_idx)),
            }

        return

    def __call__(self):
        """
        this function is called whenever CPLEX finds an integer feasible solutin
        :return:
        """

        # add initial cuts first time the callback is used
        if self.initial_cuts is not None:
            for cut, lhs in zip(self.initial_cuts['coefs'], self.initial_cuts['lhs']):
                self.add(constraint = cut, sense = "G", rhs = lhs, use = self.purge_cuts)
            self.initial_cuts = None

        # get coefficient values
        coefs = np.array(self.get_values(self.coef_idx))

        # get parameters off loss cut
        loss_value, loss_slope = self._get_loss_cut_parameters(coefs)
        excess_loss = loss_value - self.threshold_loss

        # add loss cut
        # 0 ≥ loss_value^k - loss_threshold + sum(loss_gradient^k[j] * w[j]) - sum(loss_gradient^k[j] * w^k[j])
        # -sum(loss_gradient^k[j] * w[j]) ≥ loss_value^k - loss_threshold  - sum(loss_gradient^k[j] * w^k[j])
        if np.greater(excess_loss, 0.0):
            #print_log('**loss exceeds baseline loss...**', 'adding feasibility cut', 'loss: {}'.format(loss_value))
            cut_args = {
                'constraint': [self.coef_idx, (-loss_slope).tolist()],
                'sense': 'G',
                'rhs': excess_loss - loss_slope.dot(coefs)
                }
            self.stats['n_optimality_cuts'] += 1
        else:
            #print_log('**loss is within epsilon of baseline loss...**', 'adding optimality cut', 'loss: {}'.format(loss_value))
            cut_args = {
                'constraint': [self.cut_idx, [1.0] + (-loss_slope).tolist()],
                'sense': 'G',
                'rhs': excess_loss - loss_slope.dot(coefs)
                }
            self.stats['n_feasibility_cuts'] += 1

        self.add(**cut_args)

        # update stats

        self.stats['intermediate_solutions'].append(coefs)
        incumbent_update = np.less_equal(loss_value, self.threshold_loss)
        if incumbent_update:
            self.stats.update({
                'lowerbound': self.get_objective_value(),
                'incumbent': coefs,
                })

        return

    def _get_loss_cut_parameters(self, coefs):
        """
        computes the value and slope of the logistic loss in a numerically stable way
        this function should only be used when generating cuts in cutting-plane algorithms
        (computing both the value and the slope at the same time is slightly cheaper)
        see also: http://stackoverflow.com/questions/20085768/
        Parameters
        ----------
        Z           numpy.array containing training data with shape = (n_rows, n_cols)
        coefs         numpy.array of coefficients with shape = (n_cols,)
        Returns
        -------
        loss_value  scalar = 1/n_rows * sum(log( 1 .+ exp(-Z*coefs))
        loss_slope: (n_cols x 1) vector = 1/n_rows * sum(-Z*coefs ./ (1+exp(-Z*coefs))
        """

        scores = self.Z.dot(coefs)
        pos_idx = scores > 0
        exp_scores_pos = np.exp(-scores[pos_idx])
        exp_scores_neg = np.exp(scores[~pos_idx])

        # compute loss value
        loss_value = np.empty_like(scores)
        loss_value[pos_idx] = np.log1p(exp_scores_pos)
        loss_value[~pos_idx] = -scores[~pos_idx] + np.log1p(exp_scores_neg)
        loss_value = loss_value.mean()

        # compute loss slope
        log_probs = np.empty_like(scores)
        log_probs[pos_idx] = 1.0 / (1.0 + exp_scores_pos)
        log_probs[~pos_idx] = exp_scores_neg / (1.0 + exp_scores_neg)
        loss_slope = self.Z.T.dot(log_probs - 1.0) / self.Z.shape[0]
        return loss_value, loss_slope

def build_discrepancy_mip_cpx(X, y, baseline_probs, delta, **params):
    """
    Sets up a Cplex object to train a logistic classifier with loss guarantees using a cutting-plane algorithm
    :param X:
    :param y:
    :param baseline_loss:
    :param baseline_probs:
    :param coefs_max:
    :return: cplex, indices
    ------------------------------------------------------------------------------------------------------------------------
    The coefficients of the classifier are fit by solving an ERM of the form:
    min loss(data)
    st. L[w] <= L[w_0] + epsilon

    :param kwargs:
    :return: cplex object
    -----------------------------------------------------------
    Discrepancy Classifier with Epsilon-Level-Set MIP Formulation
    -----------------------------------------------------------
    maximize np.sum(d[i])
    such that
    L <= loss_value[w_0] + epsilon
    d[i] = v[i] + z[i]
    MZ[i] * (1 - z[i]) ≥ - (score[i] - scores_ub[i]) for i in 1,...,n –---- score[i] < U[i] => z[i] = 0
    MV[i] * (1 - v[i]) ≥ (sf[i] - scores_lb[i])   for i in 1,...,n –---- score[i] ≥ B[i] => v[i] = 0

    L    in [0, infinity]
    w[j] in [coefs_lb[j], coefs_ub[j]]    for j = 1, 2, ..., d
    d[i] in {0, 1}                        for i in 1,...,n
    v[i] in {0, 1}                        for i in 1,...,n
    z[i] in {0, 1}                        for i in 1,...,n
    ---------------------
    MIP Parameters:
    ---------------------
    MZ[i] =
    MV[i] =
    """
    n_samples, n_variables = X.shape
    assert np.isfinite(X).all()
    assert np.isfinite(y).all()
    assert len(y) == n_samples
    intercept_idx = 0
    Xf = np.insert(X, intercept_idx, 1.0, axis= 1)

    # X_norm = np.abs(Xf).sum(axis = 1)
    X_norm = np.abs(X).sum(axis = 1)
    W_max = params.get('W_max', DiscrepancyMIP.DEFAULT_VALUES['W_max'])
    W_min = params.get('W_min', DiscrepancyMIP.DEFAULT_VALUES['W_min'])
    print_log("coef max: {}".format(W_max))
    print_log("coef min: {}".format(W_min))
    score_max = W_max * X_norm
    score_min = W_min * X_norm

    # intercept_idx = 0
    # Xf = np.insert(X, intercept_idx, 1.0, axis= 1)

    # create cplex object
    cpx = Cplex()
    cpx.objective.set_sense(cpx.objective.sense.maximize)
    vars = cpx.variables
    cons = cpx.linear_constraints

    # initialize variables
    #X_norm = np.abs(Xf).sum(axis = 1)
    # X_norm = np.abs(X).sum(axis = 1)
    # W_max = 25.0
    # W_min = -W_max
    # score_max = W_max * X_norm
    # score_min = W_min * X_norm

    # loss
    loss_names = ['loss']
    vars.add(obj = [0.0],
             ub = [float('inf')],
             lb = [0.0],
             types = 'C',
             names = loss_names)

    # coefficients
    n_coefs = n_variables + 1
    coef_names = ['w_%d' % j for j in range(n_coefs)]
    vars.add(names = coef_names,
             obj = [0.0] * n_coefs,
             lb =[W_min] * n_coefs,
             ub =[W_max] * n_coefs,
             types = 'C' * n_coefs)

    # discrepany indicators
    d_names = ['d_%d' % i for i in range(n_samples)]
    vars.add(names = d_names,
             obj = [1.0] * n_samples,
             ub = [1.0] * n_samples,
             lb = [0.0] * n_samples,
             types = 'B' * n_samples)


    # z indicators
    z_names = ['z_%d' % i for i in range(n_samples)]
    vars.add(names = z_names,
             obj = [0.0] * n_samples,
             ub = [1.0] * n_samples,
             lb = [0.0] * n_samples,
             types = 'B' * n_samples)

    # v indicators
    v_names = ['v_%d' % i for i in range(n_samples)]
    vars.add(names = v_names,
             obj = [0.0] * n_samples,
             ub = [1.0] * n_samples,
             lb = [0.0] * n_samples,
             types = 'B' * n_samples)



    # constraint to set d[i] = v[i] + z[i]  for each i
    # d_i = v_i + z_i
    # v_i + z_i - d_i = 0
    expr_names = [[v, z, d] for v, z, d in zip(z_names, v_names, d_names)]
    expr_values = [[1.0, 1.0, -1.0] for i in range(n_samples)]
    cons.add(names = ['dvz[%i]' % i for i in range(n_samples)],
             lin_expr = [SparsePair(ind, val) for ind, val in zip(expr_names, expr_values)],
             senses = ["E"] * n_samples,
             rhs = [0.0] * n_samples)

    # disc_threshold_min = np.nan_to_num(disc_threshold_min, nan= -1000.0 ) # replace all NAN values with large negative number
    disc_threshold_min = logit(baseline_probs - delta)
    disc_threshold_max = logit(baseline_probs + delta)

    """
    Big-M constraint for z[i] = 1[s[i] < U[i]]
    ------------------------------------------

    We want

    z[i] = 1[s[i] < U[i]]

    This requires:

    1. setting z[i] = 1 when s[i] < U[i]
    2. setting z[i] = 0 when s[i] ≥ U[i]

    Since we are solving a maximization problem, the optimization process
    will naturally set z[i] = 1. Thus we only need a constraint to ensure
    that z[i] = 0 when s[i] ≥ U[i].

    The following constraint sets z[i] = 0 when B[i] - s[i] > 0

    MZ * (1 - z[i]) ≥ (s[i] - U[i]) <- this constraint sets z[i] = 0 when (s[i] - U[i]) > 0

    where

    MZ = max(s_i - U_i)
    MZ = max(s_i) - U_i

    MZ * (1 - z[i]) ≥ (s[i] - U[i])
    MZ - MZ*z[i] ≥ s[i] - U[i]
    MZ - MZ*z[i] ≥ s[i] - U[i]
    MZ + U[i] ≥ MZ*z[i] + s[i]
    """
    MZ = score_max - disc_threshold_min
    for i, U in enumerate(disc_threshold_min):
        if np.isnan(U): # if out of bounds set to zero
            expr_names = ['z_%d' % i]
            expr_values = [1.0]
            cons.add(names=['set_z_off[%i]' % i],
                     lin_expr=[SparsePair(expr_names, expr_values)],
                     senses=["E"],
                     rhs=[0.0])
        else:
            expr_names = ['z_%d' % i] + coef_names
            expr_values = [MZ[i]] + Xf[i].tolist()
            cons.add(names=['set_z_off[%i]' % i],
                     lin_expr = [SparsePair(expr_names, expr_values)],
                     senses = ["L"],
                     rhs = [U + MZ[i]])

    """
    Big-M constraint for v[i] = 1[s[i] > B[i]]
    ------------------------------------------

    We want

    v[i] = 1[s[i] > B[i]]

    This requires:

    1. setting v[i] = 1 when s[i] > B[i]
    2. setting v[i] = 0 when s[i] ≤ B[i]

    Since we are solving a maximization problem, the optimization process
    will naturally set v[i] = 1. Thus we only need a constraint to ensure
    that v[i] = 0 when s[i] ≤ B[i]

    The following constraint  sets v[i] = 0 when B[i] - s[i] > 0

    MV * (1 - v[i]) ≥ (B[i] - s[i])

    Here, we can set the Big-M parameter MV as:

    MV = max(B_i - s_i)
    MV = B_i + max(- s_i)
    MV = B_i - min(s_i)

    The CPLEX implementation is:

    MV * (1 - v[i]) ≥ B[i] - s[i]
    MV - MV*v[i]    ≥ B[i] - s[i]
    -MV * v[i] + s[i] ≥ B[i] -Mv
    -MV * v[i] -sum(w[j] *x[i,j]) ≥ B[i] - MV
    """
    MV = disc_threshold_max - score_min
    for i, B in enumerate(disc_threshold_max):
        if np.isnan(B): # if out of bounds set to zero
            expr_names = ['v_%d' % i]
            expr_values = [1.0]
            cons.add(names=['set_v_off[%i]' % i],
                     lin_expr=[SparsePair(expr_names, expr_values)],
                     senses=["E"],
                     rhs=[0.0])
        else:
            expr_names = ['v_%d' % i] + coef_names
            expr_values = [-MV[i]] + Xf[i].tolist()
            cons.add(names=['set_v_off[%i]' % i],
                     lin_expr = [SparsePair(expr_names, expr_values)],
                     senses = ["G"],
                     rhs = [B - MV[i]])

    indices = {
        'n_variables': vars.get_num(),
        'n_constraints': cons.get_num(),
        'names': vars.get_names(),
        #
        'loss_name': loss_names,
        'coef_names': coef_names,
        'd_names': d_names,
        'v_names': v_names,
        'z_names': z_names,
        #
        'loss': vars.get_indices(loss_names),
        'coefs': vars.get_indices(coef_names),
        'd': vars.get_indices(d_names),
        'v': vars.get_indices(v_names),
        'z': vars.get_indices(z_names),
        }

    indices.update(params)
    return cpx, indices

class DiscrepancyMIP(object):
    """
    Convenience class to create, solve, and check the integrity of the ERM
    to fit a logistic regression classifier with accessibility constraints
    """
    DEFAULT_VALUES = {
        'W_max': 5.0,
        'W_min': -5.0,
        }

    PRINT_FLAG = True
    PARALLEL_FLAT = True

    def __init__(self, X, y, baseline_loss, baseline_probs, delta, epsilon, random_seed = 2338, **kwargs):
        """
        :param X:
        :param y:
        :param baseline_loss:
        :param baseline_probs:
        :param delta:
        :param epsilon:
        :param random_seed:
        :param kwargs:
        """

        # attach parameters
        default_params = DiscrepancyMIP.DEFAULT_VALUES
        default_params.update(kwargs)

        # set flags
        self._print_flag = DiscrepancyMIP.PRINT_FLAG
        self._parallel_flag = DiscrepancyMIP.PARALLEL_FLAT

        # attach data
        assert X.ndim == 2
        assert X.shape[0] > 0
        assert np.isfinite(X).all()
        assert np.isfinite(y).all()
        assert np.isin(y, (-1, 1)).all()
        assert X.shape[0] == len(y)
        self.X = X
        self.y = y
        self.n_samples = X.shape[0]
        self.n_variables = X.shape[1]
        self._intercept_idx = 0
        self._coefficient_idx = np.arange(1, self.n_variables + 1)

        # attach baseline loss and probabilites
        assert np.greater_equal(baseline_loss, 0.0)
        assert np.isfinite(baseline_probs.all())
        assert np.greater_equal(baseline_probs, 0.0).all() and np.less_equal(baseline_probs, 1.0).all()
        assert len(baseline_probs) == self.n_samples
        self.baseline_loss = baseline_loss
        self.baseline_probs = baseline_probs

        # attach parameters
        assert np.greater_equal(delta, 0.0)
        assert np.greater_equal(epsilon, 0.0)
        self.delta = delta
        self.epsilon = epsilon
        self.threshold_loss = self.baseline_loss + self.epsilon

        # random seed
        self.random_seed = random_seed

        # setup mip and callback
        cpx, indices = build_discrepancy_mip_cpx(X = self.X, y = self.y, baseline_loss = baseline_loss, baseline_probs = baseline_probs, delta = delta, **default_params)
        loss_cb = cpx.register_callback(LossCallback)
        loss_cb.initialize(X = X, y = y, loss_idx = indices['loss'], coef_idx = indices['coefs'], threshold_loss = baseline_loss + epsilon)
        cpx = self._set_mip_parameters(cpx, self.random_seed)

        # attach CPLEX object
        self.mip = cpx
        self.indices = indices
        self.loss_cb = loss_cb
        self.vars = self.mip.variables
        self.cons = self.mip.linear_constraints
        self.parameters = self.mip.parameters

    def solve(self, time_limit = 60.0, max_gap = 0.1, return_stats = False, return_incumbents = False):
        """
        solves MIP
        #
        :param time_limit: max # of seconds to run before stopping the B & B.
        :param return_stats: set to True to record basic profiling information as the B&B algorithm runs (warning: this may slow down the B&B)
        :param return_incumbents: set to True to record all imcumbent solution processed during the B&B search (warning: this may slow down the B&B)
        :return:
        """

        if (return_stats or return_incumbents):
            self._add_stats_callback(store_solutions = return_incumbents)

        if time_limit is not None:
            self.mip = set_mip_time_limit(self.mip, time_limit)

        if max_gap is not None:
            self.mip = set_mip_max_gap(self.mip, max_gap)

        self.mip.solve()

        info = self.solution_info
        if (return_stats or return_incumbents):
            info['progress_info'], info['progress_incumbents'] = self._stats_callback.get_stats()

        return info

    #### key values ####
    @property
    def coefficients(self):
        """
        :return: coefficients of the linear classifier
        """
        s = self.solution
        if self.has_solution:
            coefs = np.array(s.get_values(self.indices['coefs']))
        else:
            coefs = np.repeat(np.nan, self.n_variables)
        return coefs

    @property
    def loss(self):
        """return logistic loss value of fitted coefficient vector over dataset"""
        loss = float('nan')
        if self.has_solution:
            loss = compute_log_loss(X = self.X, y = self.y, w = self.coefficients)
        return loss

    @property
    def discrepancy(self):
        disc = float('nan')
        if self.has_solution:
            disc = compute_discrepancy_indicators(X = self.X, y = self.y, w = self.coefficients, p0 = self.baseline_probs, delta = self.delta)
        return sum(disc)

    def check_mip_solution(self):
        """
        todo: runs basic tests to make sure that the MIP contains a suitable solution
        :return:
        """
        s = self.solution
        loss_true = self.loss
        indices = self.indices

        # check that v and z are mutually exclusive
        v_cpx = s.get_values(indices['z'])
        z_cpx = s.get_values(indices['v'])
        assert np.less_equal(v_cpx + z_cpx, 1.0).all()

        # todo: check that v is set correctly
        # todo: check that z is set correctly

        # todo: check that z[i] = 1 when p[i] < p_baseline[i] - delta
        d_cpx = np.greater(s.get_values(indices['d']), 0.0).astype(int)
        d_true = compute_discrepancy_indicators(X = self.X, y = self.y, w = self.coefficients, p0 = self.baseline_probs, delta = self.delta)
        assert np.array_equal(d_cpx, d_true), 'discrepancy indicators do not match discrepancy values'
        try:
            msg = 'discrepancy indicators do not match discrepancy values'
            assert np.array_equal(d_cpx, d_true), msg
        except AssertionError as e:
            warnings.warn(msg)
            print_log(msg)

        # check that loss of model is within loss from cplex
        loss_cpx = s.get_values(indices['loss'])[0]
        try:
            msg = f'loss {loss_true:.6f} does not match cplex loss {loss_cpx:.6f}'
            assert np.isclose(loss_true, loss_cpx), msg
        except AssertionError as e:
            warnings.warn(msg)
        return True

    def __repr__(self):
        s = [
            f'<DiscrepancyMIP(n={self.n_samples}, d={self.n_variables}, delta={self.delta}, epsilon={self.epsilon})>',
            f'discrepancy: {self.discrepancy}',
            f'loss: {self.loss:.3f}',
            f'threshold_loss: {self.threshold_loss:.4f} (={self.baseline_loss:.4f}+{self.epsilon:.4f})'
            ]

        # add MIP info
        info = self.solution_info
        s.extend([f'{k}: {info[k]}' for k in ('has_solution', 'status')])
        s.extend([f'{k}: {info[k]:.2%}' for k in ('gap',)])
        s.extend([f'{k}: {info[k]:.4f}' for k in ('objval', 'upperbound', 'lowerbound')])
        s = '\n'.join(s)
        return s


    #### generic mip methods and properties ####
    @property
    def has_solution(self):
        """returns true if mip has a feasible solution"""
        return has_solution(self.mip)

    @property
    def solution(self):
        """
        :return: handle to CPLEX solution
        """
        return self.mip.solution

    @property
    def solution_info(self):
        """returns information associated with the current best solution for the mip"""
        return get_mip_stats(self.mip)

    def add_initial_solution(self, coefs):
        """
        adds initial solutions to MIP
        :param coefs:
        :return:
        """
        coefs = np.array(coefs).flatten()
        assert np.isfinite(coefs).all()
        assert len(coefs) == self.n_variables + 1

        # add solution to MIP start pool
        sol = coefs.tolist()
        idx = self.indices['coefs']
        self.mip.MIP_starts.add(SparsePair(val = sol, ind = idx), self.mip.MIP_starts.effort_level.solve_MIP)

    def _set_mip_parameters(self, cpx, random_seed):
        """
        sets CPLEX parameters
        :param cpx:
        :return:
        """
        p = cpx.parameters
        p.randomseed.set(random_seed)

        # annoyances
        p.paramdisplay.set(False)
        p.output.clonelog.set(0)
        p.mip.tolerances.mipgap.set(0.0)
        return cpx

    def _add_stats_callback(self, store_solutions = False):
        if not hasattr(self, '_stats_callback'):
            sol_idx = self.indices['coefs']
            min_idx, max_idx = min(sol_idx), max(sol_idx)
            assert np.array_equal(np.array(sol_idx), np.arange(min_idx, max_idx + 1))
            cb = self.mip.register_callback(StatsCallback)
            cb.initialize(store_solutions, solution_start_idx = min_idx, solution_end_idx = max_idx)
            self._stats_callback = cb

    @property
    def print_flag(self):
        """
        set as True in order to print output information of the MIP
        :return:
        """
        return self._print_flag

    @print_flag.setter
    def print_flag(self, flag):
        if flag is None:
            self._print_flag = DiscrepancyMIP.PRINT_FLAG
        elif isinstance(flag, bool):
            self._print_flag = bool(flag)
        else:
            raise ValueError('print_flag must be boolean or None')

        # toggle flag
        if self._print_flag:
            self.parameters.mip.display.set(self.parameters.mip.display.default())
            self.parameters.simplex.display.set(self._print_flag)
        else:
            self.parameters.mip.display.set(False)
            self.parameters.simplex.display.set(False)

    @property
    def parallel_flag(self):
        """
        set as True in order to print output information of the MIP
        :return:
        """
        return self._parallel_flag

    @parallel_flag.setter
    def parallel_flag(self, flag):

        if flag is None:
            self._parallel_flag = DiscrepancyMIP.PARALLEL_FLAG
        elif isinstance(flag, bool):
            self._parallel_flag = bool(flag)
        else:
            raise ValueError('parallel_flag must be boolean or None')

        # toggle parallel
        p = self.mip.parameters
        if self._parallel_flag:
            p.threads.set(0)
            p.parallel.set(0)
        else:
            p.parallel.set(1)
            p.threads.set(1)
