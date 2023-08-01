from .estimators import RidgelessLinearRegressor
from .datasets import *
import numpy as np
from joblib import Parallel, delayed
from .optimal_m import *
from .asymptotics import *

def simulate(estimator, n_train, n_features, alpha, sigma, seed, n_test=1, m=20, correlated_features=False, x_sigma=None):
    
    if correlated_features:
        x, y, beta = gen_train_dat_correlated(n_train, n_features, alpha, x_sigma, sigma, seed)    
        x_oos, y_oos = gen_test_dat_correlated(n_test, n_features, beta, x_sigma, sigma, seed)  
    else:
        x, y, beta = gen_train_dat(n_train, n_features, alpha, sigma, seed)  
        x_oos, y_oos = gen_test_dat(n_test, n_features, beta, sigma, seed)

    # fit ridgeless least square
    if estimator._estimator_type == "sketched" and m < n_train:
        est = estimator.fit(x, y, m)
    else:
        estimator = RidgelessLinearRegressor()
        est = estimator.fit(x, y)
    
    # calculate mse
    test_mse = np.mean((np.matmul(x_oos, est.beta) - np.matmul(x_oos, beta))**2)

    return test_mse


def simulate_emp_m(estimator, n_train, n_features, alpha, sigma, seed, n_test=1, n_psi=40, n_val=100, correlated_features=False, x_sigma=None):
    
    if correlated_features:
        x, y, beta = gen_train_dat_correlated(n_train, n_features, alpha, x_sigma, sigma, seed)    
        x_val, y_val = gen_val_dat_correlated(n_val, n_features, beta, x_sigma, sigma, seed)    
        x_oos, y_oos = gen_test_dat_correlated(n_test, n_features, beta, x_sigma, sigma, seed)  
    else:
        x, y, beta = gen_train_dat(n_train, n_features, alpha, sigma, seed)  
        x_val, y_val = gen_val_dat(n_val, n_features, beta, sigma, seed)
        x_oos, y_oos = gen_test_dat(n_test, n_features, beta, sigma, seed)

    mse_optimal_m = []
    psi = list(np.linspace(0.1, 0.49, int(n_psi/2))) + list(np.linspace(0.51, 0.99, int(n_psi/2)))
    mm = [int(ppsi * n_train) for ppsi in psi]
    mse_optimal_m = Parallel(n_jobs=-1)(delayed(optimal_m_empirical)(x, y, mm[j], x_val, beta) for j in range(n_psi))

    # If decreasing in the tail but min not attain at the end (due to numerical instability), then get m = n
    if n_train - mm[np.nanargmin(mse_optimal_m)] < 50:
        opt_m = n_train
    else:
        opt_m = mm[np.nanargmin(mse_optimal_m)]

    # fit ridgeless least square
    if estimator._estimator_type == "sketched" and opt_m < n_train:
        est = estimator.fit(x, y, opt_m)
    else:
        estimator = RidgelessLinearRegressor()
        est = estimator.fit(x, y)
    
    # calculate mse
    test_mse = np.mean((np.matmul(x_oos, est.beta) - np.matmul(x_oos, beta))**2)

    return test_mse

def parallel_simulate_emp_m(estimator, n_train, n_sim, alpha, sigma, seed, n_test, n_pts, n_psi, n_val, correlated_features=False):
    pp = np.zeros(n_pts, dtype=int)
    avg_mse = np.zeros(n_pts)
    phi_range = list(np.logspace(-1, 1, n_pts))
    opt_m = np.zeros(n_pts, dtype=int)

    for i, pphi in enumerate(phi_range):
        pp[i] = int(pphi * n_train)
        mse_oos = []

        if correlated_features:
            ppp = int(pphi * n_train / 2)
            lambda_op = list(np.ones(ppp) * 2) + list(np.ones(ppp) * 1) 
            opt_m[i] = optimal_m_correlated(n_train, pp[i], alpha, sigma, True, lambda_op, index=i)
            sigma_x = np.diag(lambda_op)
            mse_oos = Parallel(n_jobs=-1)(delayed(simulate_emp_m)(estimator, n_train, pp[i], alpha, sigma, sigma_x, seed+j, n_test=n_test, n_psi=n_psi, n_val=n_val, correlated_features=True, x_sigma=sigma_x) for j in range(n_sim))
        else:
            opt_m[i], case = optimal_m(alpha, sigma, n_train, pp[i])
            mse_oos = Parallel(n_jobs=-1)(delayed(simulate_emp_m)(estimator, n_train, pp[i], alpha, sigma, seed+j, n_test=n_test, n_psi=n_psi, n_val=n_val) for j in range(n_sim))
        avg_mse[i] = np.mean(mse_oos)

    return avg_mse

def parallel_simulate_psi(estimator, n_train, n_features, n_sim, alpha, sigma, seed, n_test, n_pts, correlated_features=False):
    
    m = np.zeros(n_pts, dtype=int)  # points in the x-axis
    avg_mse = np.zeros(n_pts)
    psi_range = list(np.linspace(0.11, 0.49, int(n_pts/2))) + list(np.linspace(0.51, 1, int(n_pts/2)))

    for i, psi in enumerate(psi_range):
        m[i] = int(n_train * psi) 
        mse_oos = []
        if correlated_features:
            ppp = int(n_features / 2)
            lambda_op = list(np.ones(ppp) * 2) + list(np.ones(ppp) * 1) 
            sigma_x = np.diag(lambda_op)
            mse_oos = Parallel(n_jobs=-1)(delayed(simulate)(estimator, n_train, n_features, alpha, sigma, seed+j, n_test=n_test, m=m[i], correlated_features=True, x_sigma=sigma_x) for j in range(n_sim))
        else:
            mse_oos = Parallel(n_jobs=-1)(delayed(simulate)(estimator, n_train, n_features, alpha, sigma, seed+j, n_test=n_test, m=m[i]) for j in range(n_sim))
        avg_mse[i] = np.mean(mse_oos)

    return avg_mse

def parallel_simulate_phi(estimator, n_train, n_sim, alpha, sigma, seed, n_test, n_pts, correlated_features=False):
    pp = np.zeros(n_pts, dtype=int)
    avg_mse = np.zeros(n_pts)
    phi_range = list(np.logspace(-1, 1, n_pts))
    opt_m = np.zeros(n_pts, dtype=int)

    for i, pphi in enumerate(phi_range):
        pp[i] = int(pphi * n_train)
        mse_oos = []

        if correlated_features:
            ppp = int(pphi * n_train / 2)
            lambda_op = list(np.ones(ppp) * 2) + list(np.ones(ppp) * 1) 
            opt_m[i] = optimal_m_correlated(n_train, pp[i], alpha, sigma, True, lambda_op, index=i)
            sigma_x = np.diag(lambda_op)
            mse_oos = Parallel(n_jobs=-1)(delayed(simulate)(estimator, n_train, pp[i], alpha, sigma, sigma_x, seed+j, n_test=n_test, m=opt_m[i], correlated_features=True, x_sigma=sigma_x) for j in range(n_sim))
        else:
            opt_m[i], case = optimal_m(alpha, sigma, n_train, pp[i])
            mse_oos = Parallel(n_jobs=-1)(delayed(simulate)(estimator, n_train, pp[i], alpha, sigma, seed+j, n_test=n_test, m=opt_m[i]) for j in range(n_sim))
        avg_mse[i] = np.mean(mse_oos)

    return avg_mse

