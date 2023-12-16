import numpy as np
import matplotlib.pyplot as plt

def read_image(path):
    """
    A wrapper around matplotlib's image loader that deals with
    images that are grayscale or which have an alpha channel

    Parameters
    ----------
    path: string
        Path to file
    
    Returns
    -------
    ndarray(M, N, 3)
        An RGB color image in the range [0, 1]
    """
    img = plt.imread(path)
    if np.issubdtype(img.dtype, np.integer):
        img = np.array(img, dtype=float)/255
    if len(img.shape) == 3:
        if img.shape[1] > 3:
            # Cut off alpha channel
            img = img[:, :, 0:3]
    if img.size == img.shape[0]*img.shape[1]:
        # Grayscale, convert to rgb
        img = np.concatenate((img[:, :, None], img[:, :, None], img[:, :, None]), axis=2)
    return img

def get_weights(I, thresh, p=1, grad_sigma=1, edge_thresh=0.5):
    """
    Create pre-pixel weights based on image brightness

    Parameters
    ----------
    I: ndarray(M, N)
        Grayscale image
    thresh: float
        Amount above which to make a point 1
    p: float
        Contrast boost, apply weights^(1/p)
    grad_sigma: float
        If >0, incorporate gradients into weights using a smoothing
        parameter of grad_sigma
    edge_thresh: float
        Threshold below which to ignore edges.  Lower thresholds will
        include more edges
    
    Returns
    -------
    ndarray(M, N)
        The weights of each pixel, in the range [0, 1]
    """
    weights = np.array(I)
    if np.max(weights) > 1:
        weights /= 255
    weights = np.minimum(weights, thresh)
    weights -= np.min(weights)
    weights /= np.max(weights)
    weights = 1-weights
    weights = weights**(1/p)
    if grad_sigma > 0:
        from scipy import signal
        w = int(np.ceil(grad_sigma*4))
        x = np.arange(-w, w+1)
        dgauss = x*np.exp(-x**2/(2*grad_sigma**2))
        dgauss /= np.sum(dgauss*np.arange(dgauss.size))
        Ix = signal.convolve2d(I, dgauss[None, :], mode='same', boundary='symm')
        Iy = signal.convolve2d(I, dgauss[:, None], mode='same', boundary='symm')
        grad = np.sqrt(Ix**2 + Iy**2)
        grad /= np.max(grad)
        grad = grad**0.2
        grad[grad < edge_thresh] = 0
        weights = np.maximum(weights, grad)
    return weights

def stochastic_universal_sample(weights, target_points, jitter=0.1):
    """
    Sample pixels according to a particular density using 
    stochastic universal sampling

    Parameters
    ----------
    ndarray(M, N)
        The weights of each pixel, in the range [0, 1]
    target_points: int
        The number of desired samples
    jitter: float
        Perform a jitter with this standard deviation of a pixel
    
    Returns
    -------
    ndarray(N, 2)
        Location of point samples
    """
    choices = np.zeros(target_points, dtype=np.int64)
    w = np.zeros(weights.size+1)
    order = np.random.permutation(weights.size)
    w[1::] = weights.flatten()[order]
    w = w/np.sum(w)
    w = np.cumsum(w)
    p = np.random.rand() # Cumulative probability index, start off random
    idx = 0
    for i in range(target_points):
        while idx < weights.size and not (p >= w[idx] and p < w[idx+1]):
            idx += 1
        idx = idx % weights.size
        choices[i] = order[idx]
        p = (p + 1/target_points) % 1
    X = np.array(list(np.unravel_index(choices, weights.shape)), dtype=float).T
    if jitter > 0:
        X += jitter*np.random.randn(X.shape[0], 2)
    return X

def get_centroids_edt(X, weights):
    """
    Compute weighted centroids of Voronoi regions of points in X

    Parameters
    ----------
    X: ndarray(n_points, 2)
        Points locations
    weights: ndarray(M, N)
        Weights to use at each pixel in the Voronoi image
    
    Returns
    -------
    ndarray(<=n_points, 2)
        Points moved to their centroids.  Note that some points may die
        off if no pixel is nearest to them
    """
    from scipy.ndimage import distance_transform_edt
    from scipy import sparse
    ## Step 1: Comput Euclidean Distance Transform
    mask = np.ones_like(weights)
    X = np.array(np.round(X), dtype=np.int64)
    mask[X[:, 0], X[:, 1]] = 0
    _, inds = distance_transform_edt(mask, return_indices=True)
    ## Step 2: Take weighted average of all points that have the same
    ## label in the euclidean distance transform, using scipy's sparse
    ## to quickly add up weighted coordinates of all points with the same label
    inds = inds[0, :, :]*inds.shape[2] + inds[1, :, :]
    inds = inds.flatten()
    N = len(np.unique(inds))
    idx2idx = -1*np.ones(inds.size)
    idx2idx[np.unique(inds)] = np.arange(N)
    inds = idx2idx[inds]
    ii, jj = np.meshgrid(np.arange(weights.shape[0]), np.arange(weights.shape[1]), indexing='ij')
    cols_i = (weights*ii).flatten()
    cols_j = (weights*jj).flatten()
    num_i = sparse.coo_matrix((cols_i, (inds, np.zeros(inds.size, dtype=np.int64))), shape=(N, 1)).toarray().flatten()
    num_j = sparse.coo_matrix((cols_j, (inds, np.zeros(inds.size, dtype=np.int64))), shape=(N, 1)).toarray().flatten()
    denom = sparse.coo_matrix((weights.flatten(), (inds, np.zeros(inds.size, dtype=np.int64))), shape=(N, 1)).toarray().flatten()
    num_i = num_i[denom > 0]
    num_j = num_j[denom > 0]
    denom = denom[denom > 0]
    return np.array([num_i/denom, num_j/denom]).T

def voronoi_stipple(I, thresh, target_points, p=1, grad_sigma=1, edge_thresh=0.5, n_iters=10, do_plot=False):
    """
    An implementation of the method of [2]

    [2] Adrian Secord. Weighted Voronoi Stippling
    
    Parameters
    ----------
    I: ndarray(M, N, 3)
        An RGB/RGBA or grayscale image
    thresh: float
        Amount above which to make a point 1
    p: float
        Contrast boost, apply weights^(1/p)
    grad_sigma: float
        If >0, use a canny edge detector with this standard deviation
    edge_thresh: float
        Threshold below which to ignore edges.  Lower thresholds will
        include more edges
    n_iters: int
        Number of iterations
    do_plot: bool
        Whether to plot each iteration
    
    Returns
    -------
    ndarray(<=target_points, 2)
        An array of the stipple pattern, with x coordinates along the first
        column and y coordinates along the second column.
        Note that the number of points may be slightly less than the requested
        points due to density filtering or resolution limits of the Voronoi computation
    """
    if np.max(I) > 1:
        I = I/255
    if len(I.shape) > 2:
        I = 0.2125*I[:, :, 0] + 0.7154*I[:, :, 1] + 0.0721*I[:, :, 2]
    ## Step 1: Get weights and initialize random point distribution
    ## via rejection sampling
    weights = get_weights(I, thresh, p, grad_sigma=grad_sigma, edge_thresh=edge_thresh)
    X = stochastic_universal_sample(weights, target_points)
    X = np.array(np.round(X), dtype=int)
    X[X[:, 0] >= weights.shape[0], 0] = weights.shape[0]-1
    X[X[:, 1] >= weights.shape[1], 1] = weights.shape[1]-1

    ## Step 2: Repeatedly re-compute centroids of Voronoi regions
    if do_plot:
        plt.figure(figsize=(10, 10))
    for it in range(n_iters):
        if do_plot:
            plt.clf()
            plt.scatter(X[:, 1], X[:, 0], 1)
            plt.gca().invert_yaxis()
            plt.xlim([0, weights.shape[1]])
            plt.ylim([weights.shape[0], 0])
            plt.savefig("Voronoi{}.png".format(it), facecolor='white')
        X = get_centroids_edt(X, weights)

    X[:, 0] = I.shape[0]-X[:, 0]
    return np.fliplr(X)

def density_filter(X, fac, k=1):
    """
    Filter out points below a certain density

    Parameters
    ----------
    X: ndarray(N, 2)
        Point cloud
    fac: float
        Percentile (between 0 and 1) of points to keep, by density
    k: int
        How many neighbors to consider
    
    Returns
    -------
    ndarray(N)
        Distance of nearest point
    """
    from scipy.spatial import KDTree
    tree = KDTree(X)
    dd, _ = tree.query(X, k=k+1)
    dd = np.mean(dd[:, 1::], axis=1)
    q = np.quantile(dd, fac)
    return X[dd < q, :]
