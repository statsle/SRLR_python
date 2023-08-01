import numpy as np
from .optimal_m import *
from joblib import Parallel, delayed
from scipy.optimize import fsolve   # solve for the function

def asy_risk_nonsketch(n, p, alpha, sigma):

    phi = p / n

    # gamma < 1
    r_up = sigma**2 * phi / (1 - phi)

    # gamma > 1
    r_op = alpha**2 * (1 - 1/phi) + sigma**2 / (phi - 1)

    return (phi < 1) * r_up + (phi > 1) * r_op


def asy_risk_sketch(n, p, m, alpha, is_orthogonal=True, sigma=1):
    """
    gamma: p/n
    psi: m/n
    norm: alpha in Lemma 3.2
    """

    phi = p / n
    psi = m / n
    condition = phi / psi

    # condition < 1
    if is_orthogonal:
        r_up = sigma**2 * condition / (1 - condition)
    else:
        r_up = sigma**2 * phi / (1 - phi) + sigma**2 * condition / (1 - condition)

    # condition > 1
    r_op = alpha**2 * (1 - 1/condition) + sigma**2 / (condition - 1)

    r = (condition < 1) * r_up + (condition > 1) * r_op

    return r


def parallel_asy_phi(n_train, alpha, sigma, n_pts, sketching=False, correlated_features=False):
    pp = np.zeros(n_pts, dtype=int)
    mse_asy = np.zeros(n_pts)
    opt_m = np.zeros(n_pts)

    phi = list(np.logspace(-1, 1, n_pts))
    for i, pphi in enumerate(phi):
        pp[i] = int(pphi * n_train)

        if sketching:
            opt_m[i], case = optimal_m(alpha, sigma, n_train, pp[i])

            if opt_m[i] == n_train:
                mse_asy[i] = asy_risk_nonsketch(n_train, pp[i], alpha, sigma)
            else:
                mse_asy[i] = asy_risk_sketch(n_train, pp[i], alpha, sigma)
        else:
            if correlated_features:
                ppp = int(pphi * n_train / 2)
                lambda_op = list(np.ones(ppp) * 2) + list(np.ones(ppp) * 1) 
                mse_asy[i] = asy_risk_nonsketch_correlated(n_train, pp[i], alpha, lambda_op, sigma) 
            else:
                mse_asy[i] = asy_risk_nonsketch(n_train, pp[i], alpha, sigma)
    
    return mse_asy


def parallel_asy_psi(n_train, n_features, alpha, sigma, n_pts, is_orthogonal=True, correlated_features=False):
    mm = np.zeros(n_pts, dtype=int)
    mse_asy = np.zeros(n_pts)

    psi = list(np.linspace(0.01, 0.49, int(n_pts/2))) + list(np.linspace(0.51, 1, int(n_pts/2)))
    for i, ppsi in enumerate(psi):
        mm[i] = int(ppsi * n_train)

        if correlated_features:
            ppp = int(n_features / 2)
            lambda_op = list(np.ones(ppp) * 2) + list(np.ones(ppp) * 1) 
            mse_asy[i] = asy_risk_sketch_correlated(n_train, n_features, mm[i], alpha, is_orthogonal=is_orthogonal, lambda_op=lambda_op, sigma=sigma)
        else:
            mse_asy[i] = asy_risk_sketch(n_train, n_features, mm[i], alpha, is_orthogonal=is_orthogonal, sigma=sigma)

    return mse_asy


############################### Correlated case ####################################################
def fn(c, x, phi, psi):
    sum_ = [i/(-c + i * psi / phi) for i in x]
    return np.mean(sum_) - 1

def unique_c0_op(lambda1, phi, psi, init=-4):
    c0 = fsolve(fn, args=(lambda1, phi, psi), x0=init)
    if fn(c0, lambda1, phi, psi) < 1e-7:
        return c0
    else:
        print("Specify another starting point for the optimization method")

def gn(c, x, phi, psi):
    sum_ = [i/(-c + i * phi) for i in x]
    return np.mean(sum_) - 1/psi

def unique_c0_up(lambda1, phi, psi, init=4):
    c0 = fsolve(gn, args=(lambda1, phi, psi), x0=init)
    if gn(c0, lambda1, phi, psi) < 1e-7:
        return c0
    else:
        print("Specify another starting point for the optimization method")

def asy_risk_sketch_correlated(n, p, m, alpha, is_orthogonal, lambda_op, sigma=1):

    phi = p / n
    psi = m / n
    
    condition = phi / psi
    condition_inv = psi / phi

    c0_op = unique_c0_op(lambda_op, phi, psi, init=-1)[0]
    #c0_up = unique_c0_up(lambda_up, phi, psi, init=-1)[0] # Orthogonal s

    if m > p and condition_inv > 1: 
        b_up = 0
        # numerator_ = psi * np.mean([i**2 * phi / (c0_up - i * phi)**2 for i in lambda_up])
        # denominator_ = 1 - psi * np.mean([i**2 * phi / (c0_up - i * phi)**2 for i in lambda_up])
        # v_up = sigma**2 * numerator_ / denominator_
        
        if is_orthogonal:
            v_up = sigma**2 * (condition / (1-condition))
        else:
            v_up = sigma**2 * (phi/(1-phi) + condition / (1-condition))
        return b_up + v_up

    if m < p and condition > 1:
        b_op = -alpha**2 * c0_op
        numerator_ = np.mean([i**2 * condition_inv / (c0_op - i * condition_inv)**2 for i in lambda_op])
        denominator_ = 1 - np.mean([i**2 * condition_inv / (c0_op - i * condition_inv)**2 for i in lambda_op])
        v_op = sigma**2 * numerator_ / denominator_
        return b_op + v_op
    

def asy_risk_nonsketch_correlated(n, p, alpha, lambda_op, sigma):
    phi = p / n
    psi = 1
    condition = phi / psi
    condition_inv = psi / phi

    # gamma < 1
    if phi < 1:
        return sigma**2 * phi / (1 - phi)
    
    c0_op = unique_c0_op(lambda_op, phi, psi, init=-1)[0]

    # gamma > 1
    if phi > 1:
        b_op = -alpha**2 * c0_op
        numerator_ = np.mean([i**2 * condition_inv / (c0_op - i * condition_inv)**2 for i in lambda_op])
        denominator_ = 1 - np.mean([i**2 * condition_inv / (c0_op - i * condition_inv)**2 for i in lambda_op])
        v_op = sigma**2 * numerator_ / denominator_
        return b_op + v_op
    
def optimal_m_correlated(n_train, n_features, alpha, sigma, is_orthogonal, lambda_op, n_pts_asymp=100, index=0):
    # Theorem 4 
    mm = np.zeros(n_pts_asymp, dtype=int)
    mse_asy = []

    psi = list(np.linspace(0.1, 0.49, int(n_pts_asymp/2))) + list(np.linspace(0.51, 0.99, int(n_pts_asymp/2)))
    mm = [int(ppsi * n_train) for ppsi in psi]
    mse_asy = Parallel(n_jobs=-1)(delayed(asy_risk_sketch_correlated)(n_train, n_features, mm[j], alpha, is_orthogonal, lambda_op, sigma=sigma) for j in range(n_pts_asymp))
        
    # 1. Remove none or negative mse
    none_index = [i for i in range(len(mse_asy)) if mse_asy[i] == None or mse_asy[i] < 0]
    if len(none_index) > 0:
        mse_asy = np.delete(mse_asy, (np.r_[none_index]))
        psi = np.delete(psi, (np.r_[none_index]))
        mm = np.delete(mm, (np.r_[none_index]))

    # 2. Remove outliers that change dramastically 
    dx = psi[1] - psi[0]
    dy = np.diff(mse_asy) / dx
    # Find sign change
    sign_change = (np.diff(np.sign(dy)) != 0)*1
    change_index = [i for i in range(len(sign_change)) if sign_change[i] == 1]
    if len(change_index) > 0:
        mse_asy = np.delete(mse_asy, (np.r_[change_index]))
        psi = np.delete(psi, (np.r_[change_index]))
        mm = np.delete(mm, (np.r_[change_index]))
    
    # 3. If decreasing in the tail but min not attain at the end (due to numerical instability), then get m = n
    if n_train - mm[np.nanargmin(mse_asy)] < 50:
        opt_m = n_train
    else:
        opt_m = mm[np.nanargmin(mse_asy)]
    
    # plt.clf()        
    # ax = sns.lineplot(x=psi, y=mse_asy)
    # ax.set_ylim(-2, 60)
    # ax.set_title("n features: " + str(n_features) + ", optimal m: " + str(opt_m) + ", phi :" + str(n_features/n_train) + ", " + str(mse_asy[-1] - mse_asy[0]))
    # ax.set_ylabel("Out-of-sample Risk")
    # ax.set_xlabel(r"$\psi$ = m/n")
    # plt.show()
    
    return opt_m