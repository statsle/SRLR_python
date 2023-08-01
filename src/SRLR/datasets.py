import numpy as np

def gen_train_dat(n_train, n_features, alpha, sigma, seed):
    """
    n_samples: n
    n_features: p
    alpha: beta's norm
    sigma: sigma
    """

    #bet = norm / np.sqrt(p) * np.ones(p) 
    
    # seed for fixed X
    np.random.seed(1234)

    # Assumption 2.1 (model specification)
    x = np.random.normal(size=(n_train, n_features), scale=1)
    
    # seed for eps, beta
    np.random.seed(seed+3524)
    
    # Assumption 2.2 (random beta)
    bet = alpha / np.sqrt(n_features) * np.random.normal(size=n_features)  #from guassian prior
    eps = np.random.normal(size=n_train, scale=sigma)
    y = np.matmul(x, bet) + eps

    return x, y, bet

def gen_val_dat(n_val, n_features, beta, sigma=1, seed=1234):
        
    # seed for X 
    np.random.seed(3213)
    
    # Assumption 2.4 (correlated features)
    x = np.random.normal(size=(n_val, n_features), scale=1)
    
    # seed for eps
    np.random.seed(seed+1244)
    eps = np.random.normal(size=n_val, scale=sigma)
    y = np.matmul(x, beta) + eps

    return x, y

def gen_test_dat(n_test, n_features, beta, sigma, seed):
    
    # seed for x0 and eps
    np.random.seed(seed+1234)
    
    x0 = np.random.normal(size=(n_test, n_features), scale=1)
    eps = np.random.normal(size=n_test, scale=sigma)
    y = np.matmul(x0, beta) + eps

    return x0, y

def gen_train_dat_correlated(n_train, n_features, alpha, x_sigma, sigma, seed):

    # seed for fixed X
    np.random.seed(1234)

    # Assumption 2.4 (correlated features)
    x = np.random.multivariate_normal(mean=np.zeros(n_features), cov=x_sigma, size=(n_train))
    
    # seed for eps, beta
    np.random.seed(seed+3524)
    
    # Assumption 2.2 (random beta)
    bet = alpha / np.sqrt(n_features) * np.random.normal(size=n_features)  #from guassian prior
    eps = np.random.normal(size=n_train, scale=sigma)
    y = np.matmul(x, bet.T) + eps

    return x, y, bet

def gen_val_dat_correlated(n_val, n_features, beta, x_sigma, sigma, seed):

    np.random.seed(3213)
    # Assumption 2.4 (correlated features)
    x = np.random.multivariate_normal(mean=np.zeros(n_features), cov=x_sigma, size=(n_val))
    
    # seed for eps
    np.random.seed(seed+1244)
    eps = np.random.normal(size=n_val, scale=sigma)
    y = np.matmul(x, beta) + eps

    return x, y

def gen_test_dat_correlated(n_test, n_features, beta, x_sigma, sigma, seed):
    # seed for x0 and eps
    np.random.seed(seed+1234)

    x0 = np.random.multivariate_normal(mean=np.zeros(n_features), cov=x_sigma, size=(n_test))
    eps = np.random.normal(size=n_test, scale=sigma)
    y = np.matmul(x0, beta) + eps

    return x0, y
