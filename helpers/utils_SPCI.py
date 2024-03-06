import numpy as np
import math

def ellipsoid_volume(covariance_matrix, r):
    # Only compute volume of the ellipsoid along the first r dimensions
    # where r is the number of dimensions with non-zero eigenvalues
    eigenvalues = np.linalg.eigvals(covariance_matrix)
    eigenvalues = np.real(np.sort(eigenvalues)[::-1])
    eps = 1e-6
    num_r = np.sum(eigenvalues > eps)
    det_sigma = np.prod(eigenvalues[:num_r])
    constant_cd = np.pi**(num_r/2) / np.math.gamma(num_r/2 + 1)  # Volume constant for d-dimensional sphere
    volume = constant_cd * r**num_r * np.sqrt(det_sigma)
    return volume


#### From utils_EnbPI ####
def adjust_alpha_t(alpha_t, alpha, errs, gamma=0.005, method='simple'):
    if method == 'simple':
        # Eq. (2) of Adaptive CI
        return alpha_t+gamma*(alpha-errs[-1])
    else:
        # Eq. (3) of Adaptive CI with particular w_s as given
        t = len(errs)
        errs = np.array(errs)
        w_s_ls = np.array([0.95**(t-i) for i in range(t)]
                          )  # Furtherest to Most recent
        return alpha_t+gamma*(alpha-w_s_ls.dot(errs))


def ave_cov_width(df, Y):
    coverage_res = ((np.array(df['lower']) <= Y) & (
        np.array(df['upper']) >= Y)).mean()
    print(f'Average Coverage is {coverage_res}')
    width_res = (df['upper'] - df['lower']).mean()
    print(f'Average Width is {width_res}')
    return [coverage_res, width_res]

#### Miscellaneous ####


window_size = 300


def rolling_avg(x, window=window_size):
    return np.convolve(x, np.ones(window)/window)[(window-1):-window]


def generate_bootstrap_samples(n, m, B):
    '''
      Return: B-by-m matrix, where row b gives the indices for b-th bootstrap sample
    '''
    samples_idx = np.zeros((B, m), dtype=int)
    for b in range(B):
        sample_idx = np.random.choice(n, m)
        samples_idx[b, :] = sample_idx
    return(samples_idx)


def strided_app(a, L, S):
    nrows = ((a.shape[0] - L) // S) + 1
    shape = (nrows, L) + a.shape[1:]
    strides = (S * a.strides[0],) + a.strides
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)



def binning(past_resid, cov_mat_est, alpha, bins = 5):
    '''
    Input:
        past residuals: evident
        alpha: signifance level
    Output:
        beta_hat_bin as argmin of the difference
    Description:
        Compute the beta^hat_bin from past_resid, by breaking [0,alpha] into bins (like 20). It is enough for small alpha
        number of bins are determined rather automatic, relative the size of whole domain
    '''
    beta_ls = np.linspace(start=0, stop=alpha, num=bins)
    sizes = np.zeros(bins)
    for i in range(bins):
        width = np.percentile(past_resid, math.ceil(100 * (1 - alpha + beta_ls[i]))) - \
            np.percentile(past_resid, math.ceil(100 * beta_ls[i]))
        sizes[i] = ellipsoid_volume(cov_mat_est, width)
    i_star = np.argmin(sizes)
    return beta_ls[i_star]


def binning_use_RF_quantile_regr(quantile_regr, cov_mat_est, Xtrain, Ytrain, feature, beta_ls, sample_weight=None):
    # API ref: https://sklearn-quantile.readthedocs.io/en/latest/generated/sklearn_quantile.RandomForestQuantileRegressor.html
    feature = feature.reshape(1, -1)
    low_high_pred = quantile_regr.fit(Xtrain, Ytrain, sample_weight).predict(feature)
    num_mid = int(len(low_high_pred)/2)
    low_pred, high_pred = low_high_pred[:num_mid], low_high_pred[num_mid:]
    width = (high_pred-low_pred).flatten()
    width = [ellipsoid_volume(cov_mat_est, w) for w in width]
    i_star = np.argmin(width)
    wid_left, wid_right = low_pred[i_star], high_pred[i_star]
    return i_star, beta_ls[i_star], wid_left, wid_right

