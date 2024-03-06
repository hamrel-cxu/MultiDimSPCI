import pandas as pd
import numpy as np
import math
import time as time
from helpers import utils_SPCI as utils
from sklearn.neighbors import NearestNeighbors
import warnings
from sklearn_quantile import RandomForestQuantileRegressor, SampleRandomForestQuantileRegressor
from numpy.lib.stride_tricks import sliding_window_view
warnings.filterwarnings("ignore")


class SPCI_and_EnbPI():
    def __init__(self, X_train, X_predict, Y_train, Y_predict, fit_func):
        self.regressor = fit_func
        self.X_train = X_train
        self.X_predict = X_predict
        self.Y_train = Y_train
        self.Y_predict = Y_predict
        # Predicted training data centers by EnbPI
        n, n1 = len(self.X_train), len(self.X_predict)
        self.d = self.Y_train.shape[1] # dimension of output
        self.Ensemble_train_interval_centers = np.ones((n,self.d))*np.inf
        # Predicted test data centers by EnbPI
        self.Ensemble_pred_interval_centers = np.ones((n1,self.d))*np.inf
        self.Ensemble_online_resid = np.ones((n+n1,self.d))*np.inf  # LOO scores
        self.beta_hat_bins = []
        self.cov_matrix_ls = []
        #### Other hyperparameters for training
        # QRF training & how it treats the samples
        self.weigh_residuals = False # Whether we weigh current residuals more.
        self.c = 0.995 # If self.weight_residuals, weights[s] = self.c ** s, s\geq 0
        self.n_estimators = 10 # Num trees for QRF
        self.max_d = 2 # Max depth for fitting QRF
        self.criterion = 'squared_error' # {'absolute_error', 'squared_error', 'friedman_mse', 'poisson'}
        # search of \beta^* \in [0,\alpha]
        self.bins = 5 # break [0,\alpha] into bins to minimize width
        # how many LOO training residuals to use for training current QRF 
        self.T1 = None # None = use all
        # Extra: possible low-rank approximation
        self.r = None
        # Extra: local ellipsoid
        self.use_local_ellipsoid = False
        self.local_ellipsoid_idx = 0

    def one_boot_prediction(self, Xboot, Yboot, Xfull):
        model = self.regressor
        model.fit(Xboot, Yboot)
        return model.predict(Xfull)

    def fit_bootstrap_models_online_multistep(self, B, stride=1):
        '''
          Train B bootstrap estimators from subsets of (X_train, Y_train), compute aggregated predictors, and compute the residuals

          stride: int. If > 1, then we perform multi-step prediction, where we have to fit stride*B boostrap predictors.
            Idea: train on (X_i,Y_i), i=1,...,n-stride
            Then predict on X_1,X_{1+s},...,X_{1+k*s} where 1+k*s <= n+n1
            Note, when doing LOO prediction thus only care above the points above
        '''
        n = self.X_train.shape[0]
        n1 = len(self.X_predict)
        N = n-stride+1  # Total training data each one-step predictor sees
        # We make prediction every s step ahead, so these are feature the model sees
        train_pred_idx = np.arange(0, n, stride)
        # We make prediction every s step ahead, so these are feature the model sees
        test_pred_idx = np.arange(n, n+n1, stride)
        self.train_idx = train_pred_idx
        self.test_idx = test_pred_idx
        # Only contains features that are observed every stride steps
        Xfull = np.vstack([self.X_train[train_pred_idx], self.X_predict[test_pred_idx-n]])
        nsub, n1sub = len(train_pred_idx), len(test_pred_idx)
        for s in range(stride):
            ''' 1. Create containers for predictions '''
            # hold indices of training data for each f^b
            boot_samples_idx = utils.generate_bootstrap_samples(N, N, B)
            # for i^th column, it shows which f^b uses i in training (so exclude in aggregation)
            in_boot_sample = np.zeros((B, N), dtype=bool)
            # hold predictions from each f^b for fX and sigma&b for sigma
            boot_predictionsFX = np.zeros((B, nsub+n1sub, self.d))
            # We actually would just use n1sub rows, as we only observe this number of features
            out_sample_predictFX = np.zeros((n, n1sub, self.d))

            ''' 2. Start bootstrap prediction '''
            start = time.time()
            for b in range(B):
                self.b = b
                Xboot, Yboot = self.X_train[boot_samples_idx[b],
                                            :], self.Y_train[s:s+N][boot_samples_idx[b], ]
                in_boot_sample[b, boot_samples_idx[b]] = True
                boot_fX_pred = self.one_boot_prediction(Xboot, Yboot, Xfull)
                boot_predictionsFX[b] = boot_fX_pred
            print(
                f'{s+1}/{stride} multi-step: finish Fitting {B} Bootstrap models, took {time.time()-start} secs.')

            ''' 3. Obtain LOO residuals (train and test) and prediction for test data '''
            start = time.time()
            # Consider LOO, but here ONLY for the indices being predicted
            for j, i in enumerate(train_pred_idx):
                # j: counter and i: actual index X_{0+j*stride}
                if i < N:
                    b_keep = np.argwhere(
                        ~(in_boot_sample[:, i])).reshape(-1)
                    if len(b_keep) == 0:
                        # All bootstrap estimators are trained on this model
                        b_keep = 0  # More rigorously, it should be None, but in practice, the difference is minor
                else:
                    # This feature is not used in training, but used in prediction
                    b_keep = range(B)
                pred_iFX = boot_predictionsFX[b_keep, j].mean(axis=0)
                pred_testFX = boot_predictionsFX[b_keep, nsub:].mean(axis=0)
                # Populate the training prediction
                # We add s because of multi-step procedure, so f(X_t) is for Y_t+s
                true_idx = min(i+s, n-1)
                self.Ensemble_train_interval_centers[true_idx] = pred_iFX
                resid_LOO = self.Y_train[true_idx] - pred_iFX
                out_sample_predictFX[i] = pred_testFX
                self.Ensemble_online_resid[true_idx] = resid_LOO
            sorted_out_sample_predictFX = out_sample_predictFX[train_pred_idx].mean(
                0)  # length ceil(n1/stride)
            pred_idx = np.minimum(test_pred_idx-n+s, n1-1)
            self.Ensemble_pred_interval_centers[pred_idx] = sorted_out_sample_predictFX
            pred_full_idx = np.minimum(test_pred_idx+s, n+n1-1)
            resid_out_sample = self.Y_predict[pred_idx] - sorted_out_sample_predictFX
            self.Ensemble_online_resid[pred_full_idx] = resid_out_sample
            print(f'Leave-one-out residuals computed, took {time.time()-start} secs.')
        # Sanity check
        num_inf = (self.Ensemble_online_resid == np.inf).sum()
        if num_inf > 0:
            print(
                f'Something can be wrong, as {num_inf}/{n+n1} residuals are not all computed')
            print(np.where(self.Ensemble_online_resid == np.inf))
        # Get non-conformity scores by estimating
        # covariance matrix on training residuals
        self.get_test_et = False
        self.train_et = self.get_et(self.Ensemble_online_resid[:n])
        self.get_test_et = True
        self.test_et = self.get_et(self.Ensemble_online_resid[n:])
        self.all_et = np.concatenate([self.train_et, self.test_et])

    def get_local_ellipsoid(self):
        if self.use_local_ellipsoid and self.get_test_et:
            idx = self.local_ellipsoid_idx
            X_prev = np.vstack([self.X_train[idx:], self.X_predict[:idx]])
            max_past = min(1000, len(X_prev))
            X_prev = X_prev[-max_past:]
            n_neighbors = int(0.1*max_past)
            knn = NearestNeighbors(n_neighbors=n_neighbors).fit(X_prev)
            neighbors = knn.kneighbors(self.X_predict[idx].reshape(1, -1), return_distance=False).reshape(-1)
            Cov_neighbor = np.cov(self.Ensemble_online_resid[idx:][neighbors].T)
            lamb = 0.95
            local_cov = lamb * Cov_neighbor + (1-lamb) * self.global_cov
            cov_now, inv_cov_now = self.get_rank_approx(local_cov) 
            self.cov_matrix_ls.append(cov_now)
            self.local_ellipsoid_idx += 1
            if self.local_ellipsoid_idx % 25 == 0:
                print(X_prev.shape)
                print(f'Local Ellipsoid {self.local_ellipsoid_idx} computed')
        return inv_cov_now

    def get_rank_approx(self, A):
        r = self.r
        if r is not None:
            # Rank r approximation
            u, s, v = np.linalg.svd(A, full_matrices=False)
            Ur = u[:, :r]; Sr = np.diag(s[:r]); Vr = v[:r, :]
            Ar = np.dot(Ur, np.dot(Sr, Vr))
            S_inv = np.diag(1 / s[:r])  
            Ar_pseudo_inverse = np.dot(Vr.T, np.dot(S_inv, Ur.T))
        else:
            Ar = A; Ar_pseudo_inverse = np.linalg.inv(A)
        return Ar, Ar_pseudo_inverse

    def get_et(self, residuals):
        # There are shape: length-by-d, where d = dimension of Y
        if self.get_test_et is False:
            global_cov, global_inv = self.get_rank_approx(np.cov(residuals.T))
            self.global_cov = global_cov
            self.global_cov_inv = global_inv
        # Get the non-conformity scores
        nonconform_scores = []
        for i in range(len(residuals)):
            if self.use_local_ellipsoid is False:
                cov_mat_est_inv = self.global_cov_inv
            else:
                if self.get_test_et is False:
                    cov_mat_est_inv = self.global_cov_inv
                else:
                    cov_mat_est_inv = self.get_local_ellipsoid()
            nonconform_scores.append(np.sqrt(
                np.matmul(residuals[i], np.matmul(cov_mat_est_inv, residuals[i].T))))
        return np.array(nonconform_scores)

    def compute_Widths_Ensemble_online(self, alpha, stride=1, smallT=True, past_window=100, use_SPCI=False, quantile_regr='RF'):
        '''
            stride: control how many steps we predict ahead
            smallT: if True, we would only start with the last n number of LOO residuals, rather than use the full length T ones. Used in change detection
                NOTE: smallT can be important if time-series is very dynamic, in which case training MORE data may actaully be worse (because quantile longer)
                HOWEVER, if fit quantile regression, set it to be FALSE because we want to have many training pts for the quantile regressor
            use_SPCI: if True, we fit conditional quantile to compute the widths, rather than simply using empirical quantile
        '''
        self.alpha = alpha
        n1 = len(self.X_train)
        # For SPCI, this is the "lag" for predicting quantile (i.e., feature dimension)
        # For EnbPI, this is how many past non-conformity scores we take the quantile over
        self.past_window = past_window 
        if smallT:
            # Namely, for special use of EnbPI, only use at most past_window number of LOO residuals.
            n1 = min(self.past_window, len(self.X_train))
        # Now f^b and LOO residuals have been constructed from earlier
        out_sample_predict = self.Ensemble_pred_interval_centers
        start = time.time()
        # Matrix, where each row is a UNIQUE slice of residuals with length stride.
        if use_SPCI:
            s = stride
            stride = 1
        # NOTE, NOT ALL rows are actually "observable" in multi-step context, as this is rolling
        resid_strided = utils.strided_app(self.all_et[len(self.X_train) - n1:-1], n1, stride)
        # NEW: compute the non-conformity scores
        print(f'Shape of slided e_t lists is {resid_strided.shape}')
        num_unique_resid = resid_strided.shape[0]
        width_left = np.zeros(num_unique_resid)
        width_right = np.zeros(num_unique_resid)
        # NOTE: 'max_features='log2', max_depth=2' make the model "simpler", which improves performance in practice
        self.QRF_ls = []
        self.i_star_ls = []
        for i in range(num_unique_resid):
            if use_SPCI:
                remainder = i % s
                if remainder == 0:
                    # Update QRF
                    past_resid = resid_strided[i, :]
                    n2 = self.past_window
                    resid_pred = self.multi_step_QRF(past_resid, i, s, n2)
                # Use the fitted regressor.
                # NOTE, residX is NOT the same as before, as it depends on
                # "past_resid", which has most entries replaced.
                rfqr= self.QRF_ls[remainder]
                i_star = self.i_star_ls[remainder]
                wid_all = rfqr.predict(resid_pred)
                num_mid = int(len(wid_all)/2)
                wid_left = wid_all[i_star]
                wid_right = wid_all[num_mid+i_star]
                width_left[i] = wid_left
                width_right[i] = wid_right
            else:
                past_resid = resid_strided[i, :]
                # Naive empirical quantile, where we use the SAME residuals for multi-step prediction
                # The number of bins will be determined INSIDE binning
                cov_mat = self.global_cov if self.use_local_ellipsoid is False else self.cov_matrix_ls[i]
                beta_hat_bin = utils.binning(past_resid, cov_mat, alpha, self.bins)
                self.beta_hat_bins.append(beta_hat_bin)
                width_left[i] = np.percentile(
                    past_resid, math.ceil(100 * beta_hat_bin))
                width_right[i] = np.percentile(
                    past_resid, math.ceil(100 * (1 - alpha + beta_hat_bin)))
            num_print = int(num_unique_resid / 20)
            if num_print == 0:
                print(
                        f'Radius of Ellipsoid at test {i} is {width_right[i]-width_left[i]}')
            else:
                if i % num_print == 0:
                    print(
                        f'Radius of Ellipsoid at test {i} is {width_right[i]-width_left[i]}')
        print(
            f'Finish Computing {num_unique_resid} unique Prediction Intervals, took {time.time()-start} secs.')
        Ntest = len(out_sample_predict)
        # This is because |width|=T1/stride.
        width_left = np.repeat(width_left, stride)[:Ntest]
        # This is because |width|=T1/stride.
        width_right = np.repeat(width_right, stride)[:Ntest]
        Width_Ensemble = pd.DataFrame(np.c_[width_left,width_right], columns=['lower', 'upper'])
        self.Width_Ensemble = Width_Ensemble

    def get_results(self):
        # Also report average prediction region area using 
        covered_or_not, rolling_size = [], []
        for i in range(len(self.test_et)):
            et = self.test_et[i]
            lower, upper = self.Width_Ensemble.iloc[i, 0], self.Width_Ensemble.iloc[i, 1]
            covered_or_not.append((et <= upper) and (et >= lower))
            cov_mat = self.global_cov if self.use_local_ellipsoid is False else self.cov_matrix_ls[i]
            upper_v = utils.ellipsoid_volume(cov_mat, upper)
            lower_v = utils.ellipsoid_volume(cov_mat, lower)
            rolling_size.append(upper_v - lower_v)
        self.coverages_all = covered_or_not
        self.width_all = rolling_size
        mean_cov, mean_size = np.mean(covered_or_not), np.mean(rolling_size)
        print(f'Average Coverage is {mean_cov:.3f}, Average Ellipsoid Volume is {mean_size:.2e}')
        return mean_cov, mean_size

    '''
        Get Multi-step QRF
    '''

    def multi_step_QRF(self, past_resid, i, s, n2):
        '''
            Train multi-step QRF with the most recent residuals
            i: prediction index
            s: num of multi-step, same as stride
            n2: past window w
        '''
        # 1. Get "past_resid" into an auto-regressive fashion
        # This should be more carefully examined, b/c it depends on how long \hat{\eps}_t depends on the past
        # From practice, making it small make intervals wider
        num = len(past_resid)
        resid_pred = past_resid[-n2:].reshape(1, -1)
        residX = sliding_window_view(past_resid[:num-s+1], window_shape=n2)
        self.cov_matrix = self.global_cov if self.use_local_ellipsoid is False else self.cov_matrix_ls[i]
        for k in range(s):
            residY = past_resid[n2+k:num-(s-k-1)]
            self.train_QRF(residX, residY)
            if i == 0:
                # Initial training, append QRF to QRF_ls
                self.QRF_ls.append(self.rfqr)
                self.i_star_ls.append(self.i_star)
            else:
                # Retraining, update QRF to QRF_ls
                self.QRF_ls[k] = self.rfqr
                self.i_star_ls[k] = self.i_star
        return resid_pred

    def train_QRF(self, residX, residY):
        alpha = self.alpha
        beta_ls = np.linspace(start=0, stop=alpha, num=self.bins)
        full_alphas = np.append(beta_ls, 1 - alpha + beta_ls)
        self.common_params = dict(n_estimators = self.n_estimators,
                                  max_depth = self.max_d,
                                  criterion = self.criterion,
                                  n_jobs = -1)
        if residX[:-1].shape[0] > 10000:
            # see API ref. https://sklearn-quantile.readthedocs.io/en/latest/generated/sklearn_quantile.RandomForestQuantileRegressor.html?highlight=RandomForestQuantileRegressor#sklearn_quantile.RandomForestQuantileRegressor
            # NOTE, should NOT warm start, as it makes result poor
            self.rfqr = SampleRandomForestQuantileRegressor(
                **self.common_params, q=full_alphas)
        else:
            self.rfqr = RandomForestQuantileRegressor(
                **self.common_params, q=full_alphas)
        # 3. Find best \hat{\beta} via evaluating many quantiles
        # rfqr.fit(residX[:-1], residY)
        sample_weight = None
        if self.weigh_residuals:
            sample_weight = self.c ** np.arange(len(residY), 0, -1)
        if self.T1 is not None:
            self.T1 = min(self.T1, len(residY)) # Sanity check to make sure no errors in training
            self.i_star, _, _, _ = utils.binning_use_RF_quantile_regr(
                self.rfqr, self.cov_matrix, residX[-(self.T1+1):-1], residY[-self.T1:], residX[-1], beta_ls, sample_weight)
        else:
            self.i_star, _, _, _ = utils.binning_use_RF_quantile_regr(
                self.rfqr, self.cov_matrix, residX[:-1], residY, residX[-1], beta_ls, sample_weight)
    
    