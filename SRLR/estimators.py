from sklearn.base import RegressorMixin
from scipy.stats import ortho_group # othogonal matrix
from scipy.linalg import hadamard
import numpy as np
import numpy as np

class RidgelessLinearRegressor(RegressorMixin):
    """Ridgeless linear regression."""

    def __init__(self):
        self.beta = None
        self._estimator_type = "non-sketched"

    def fit(self, x, y):
        n = x.shape[0]
        p = x.shape[1]
        
        Ip = np.eye(p)
        xxinv = np.linalg.inv(np.matmul(x.T, x)/n + 1e-6 * Ip)
        xy = np.matmul(x.T, y)/n
        
        self.beta = np.matmul(xxinv, xy)
        return self

    def predict(self, x):
        return np.matmul(x, self.beta)


class SketchedRidgelessRegressor(RegressorMixin):
    """Sketched Ridgeless regression."""

    def __init__(self, is_orthogonal=True):
        self.beta = None
        self._estimator_type = "sketched"
        self.is_orthogonal = is_orthogonal

    def fit(self, x, y, m):
        n = x.shape[0]
        p = x.shape[1]
        
        # Given S
        np.random.seed(1234)
        
        if self.is_orthogonal:
            #S = generate_orthogonal_matrix(n)[0:m, 0:n]
            S = ortho_group.rvs(dim=n)[0:m, :]   # orthogonal schetching matrix
        else:
            S = np.random.normal(size=(m, n), scale=1/np.sqrt(n))    # iid schetching matrix

        sx = np.matmul(S, x)
        sy = np.matmul(S, y)

        # xxinv = np.linalg.pinv(np.matmul(sx.T, sx))
        # xy = np.matmul(sx.T, sy)
        
        Ip = np.eye(p)
        xxinv = np.linalg.inv(np.matmul(sx.T, sx)/n + 1e-6 * Ip)
        xy = np.matmul(sx.T, sy)/n
        
        self.beta = np.matmul(xxinv, xy)
        return self

    def predict(self, x):
        return np.matmul(x, self.beta)
    

def generate_hadamard_matrix(n):
    m = 1
    while 2 ** m < n:
        m += 1
    power_of_2 = 2 ** m
    return hadamard(power_of_2, dtype=np.float64) / np.sqrt(power_of_2)

def srht(X, r):
    n, _ = X.shape
    H = generate_hadamard_matrix(n)

    np.random.seed(1234)

    # Generate matrices D, P, and B
    D = np.diag(np.random.choice([-1, 1], n))  # random signs
    P = np.random.permutation(np.eye(n))  # permutation matrix
    B = np.diag(np.random.binomial(1, r/n, n))  # bernoulli diagonal

    # Perform the operation BHDP
    S = B @ H @ D @ P @ X

    # Remove rows with only zeros
    S = S[~np.all(S == 0, axis=1)]

    return S

def generate_orthogonal_matrix(n):
    X = generate_hadamard_matrix(n)
    S = srht(X, n)
    return S

