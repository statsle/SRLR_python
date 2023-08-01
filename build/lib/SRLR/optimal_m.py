import numpy as np
from .estimators import SketchedRidgelessRegressor

def optimal_m(alpha, sigma, n_train, n_features):
    snr = alpha / sigma
    phi = n_features / n_train

    if snr > 1 and phi > (1 - sigma/(2*alpha)) and phi <= alpha/(alpha - sigma):
        opt_m = phi * n_train * (alpha - sigma) / alpha
        case = 1

    elif snr <= 1 and phi > (alpha**2 / (alpha**2 + sigma**2)):
        opt_m = max(n_train / 40, 1)
        case = 2

    else:
        opt_m = n_train
        case = 3

    return int(opt_m), case

def optimal_m_empirical(x, y, m, x_val, beta):
    # fit ridgeless least square using sketching matrix
    ridgeless = SketchedRidgelessRegressor()
    ridgeless = ridgeless.fit(x, y, m)
    
    # calculate mse
    val_mse = np.mean((np.matmul(x_val, ridgeless.beta) - np.matmul(x_val, beta))**2)
    
    return val_mse
