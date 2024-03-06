import numpy as np
from scipy.linalg import sqrtm
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

def plot_uncertainty_set(Yte, Yhat, sets, idx,
                         coverage, size,
                         xlim = None, ylim = None,
                         box = False):
    '''
    idx: obtained via `find_furtherest_point_idxes` based on SPCI
    xlim, ylim: for setting aspect_ratio
    '''
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    if box:
        y1_l, y1_u, y2_l, y2_u = sets
    else:
        x, y = sets
    num_plot = len(idx)
    s = 35
    label_set ='Prediction set'
    c1, c2 = 'blue', 'orange'
    fsize = 24
    # Uncertainty sets
    for i in range(num_plot):
        if box: 
            if i == 0:
                ax.vlines(x = y1_l[idx[i]], ymin = y2_l[idx[i]], ymax =  y2_u[idx[i]], color=c1, label = label_set)
            else:
                ax.vlines(x = y1_l[idx[i]], ymin = y2_l[idx[i]], ymax =  y2_u[idx[i]], color=c1)
            ax.vlines(x = y1_u[idx[i]], ymin = y2_l[idx[i]], ymax =  y2_u[idx[i]], color=c1)
            ax.hlines(y = y2_l[idx[i]], xmin = y1_l[idx[i]], xmax =  y1_u[idx[i]], color=c1)
            ax.hlines(y = y2_u[idx[i]], xmin = y1_l[idx[i]], xmax =  y1_u[idx[i]], color=c1)
        else:
            if i == 0:
                ax.plot(x[idx[i]] + Yhat[idx[i], 0], y[idx[i]] + Yhat[idx[i], 1], color=c1, label = label_set)
            else:
                ax.plot(x[idx[i]] + Yhat[idx[i], 0], y[idx[i]] + Yhat[idx[i], 1], color=c1)
    # Yhat and Ytrue
    ax.scatter(Yhat[idx, 0], Yhat[idx, 1], s=s, color=c1, label = r'$\hat{Y}$')
    ax.scatter(Yte[idx, 0], Yte[idx, 1], s=s, color=c2, label = r'$Y$')
    ax.set_title(f'Coverage: {coverage:.2f}% & Size: {np.mean(size):.2f}', fontsize = fsize)
    ax.set_aspect('equal', adjustable='box')
    # Legend
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.01), fontsize=20, ncol = 3)
    if xlim is not None:
        ax.set_xlim(xlim); ax.set_ylim(ylim)
        ax.set_xticks([])
        ax.set_yticks([]) 
    return fig
            
def return_ellipsoid(cov_mat, r):
    # Ellipsoid: x.T @ cov_mat @ x = r
    theta = np.linspace(0, 2*np.pi, 500)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    ellipsoid = sqrtm(np.linalg.inv(cov_mat)).dot(np.array([x, y])).T
    return ellipsoid[:, 0], ellipsoid[:, 1]

def find_furthest_point_idxes(points, n1):
    # Compute the pairwise distances between all points
    distances = cdist(points, points)
    # Initialize a list to store the furthest points
    furthest_points_idxes = []
    furthest_points = []
    # Find the first furthest point
    first_point_idx = np.argmax(np.sum(distances, axis=1))
    furthest_points_idxes.append(first_point_idx)
    furthest_points.append(points[first_point_idx])
    # Find the remaining furthest points
    for _ in range(1, n1):
        # Compute the distances from the current furthest points to all other points
        furthest_distances = cdist(furthest_points, points)      
        # Find the point with the minimum distance to the current furthest points
        new_point_idx = np.argmax(np.min(furthest_distances, axis=0))
        # Add the new furthest point to the list
        furthest_points_idxes.append(new_point_idx)
        furthest_points.append(points[new_point_idx])
    return furthest_points_idxes