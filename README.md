# SRLR

Sketched Ridgeless Linear Regression

## Description

This repository presents numerical simulations that analyze the empirical risks of the sketched ridgeless estimator, aiming to enhance generalization performance. The simulations focus on determining optimal sketching sizes that minimize out-of-sample prediction risks. The results reveal that the optimally sketched estimator exhibits stable risk curves, effectively eliminating the peaks observed in the full-sample estimator. Additionally, we introduce a practical procedure to empirically identify the optimal sketching size.

Suppose we observe data vectors  (x<sub>i</sub>,y<sub>i</sub>) that follow a linear model y<sub>i</sub>=x<sub>i</sub><sup>T</sup>&beta;<sup>*</sup>+&epsilon;<sub>i</sub>, i=1,...n, where y<sub>i</sub> is a univariate response,  x<sub>i</sub> is a d-dimensional predictor, &beta;<sup>*</sup> denotes the vector of regression coefficients, and &epsilon;<sub>i</sub> is a random error. We consider the ridgeless least square estimator β̂=(X<sup>T</sup>X)<sup>+</sup>X<sup>T</sup>Y.

With this package, the simulation results in [this paper](https://arxiv.org/abs/2302.01088) can be reproduced.

## Examples

The sketched least square estimator is implemented as follows:

```Python
class SketchedRidgelessRegressor(RegressorMixin):
    """Sketched Ridgeless regression using Moore Penrose pseudoinverse."""

    def __init__(self):
        self.beta = None

    def fit(self, x, y, m, is_orthogonal=True):
        n = x.shape[0]
        p = x.shape[1]

        if is_orthogonal:
            S = ortho_group.rvs(dim=n)[0:m, :]  # orthogonal schetching matrix
        else:
            S = np.random.normal(size=(m, n), scale=1/np.sqrt(n))    # iid schetching matrix

        sx = np.matmul(S, x)
        sy = np.matmul(S, y)
        
        Ip = np.eye(p)
        xxinv = np.linalg.inv(np.matmul(sx.T, sx)/n + 1e-9 * Ip)
        xy = np.matmul(sx.T, sy)/n
        
        self.beta = np.matmul(xxinv, xy)
        return self

    def predict(self, x):
        return np.matmul(x, self.beta)
```

There are multiple approaches to generate an orthogonal matrix in Python. One method involves utilizing the ortho_group function from the scipy.stats module as shown above. Another faster alternative is to employ the Fast Fourier Transform (FFT) algorithm. In this notebook, we provide our own implementation to generate the orthogonal matrix using the FFT algorithm (See [sketched_estimator.ipynb](https://github.com/statsle/SRLR_python/blob/main/code/sketched_estimator.ipynb) for instance). 

## Simulations

Simulations from the paper include:

- [Figure 1](https://github.com/statsle/SRLR_python/blob/main/code/ridgeless_estimators.ipynb): Asymptotic risk curves for the ridgeless least square estimator, as functions of φ = p/n.
- [Figure 2](https://github.com/statsle/SRLR_python/blob/main/code/sketched_estimator.ipynb): Asymptotic risk curves for sketched ridgeless least square estimators with **isotropic features**, orthogonal or i.i.d. sketching, as functions of ψ.
- [Figure 3](https://github.com/statsle/SRLR_python/blob/main/code/optimal_m.ipynb): Optimal sketching size selected based on SNR and φ, as described in Theorem 3.3. Asymptotic risk curves for the full-sample (no sketching) and sketched ridgeless least square estimators with **isotropic features** and orthogonal or i.i.d. sketching, as functions of φ.
- [Figure 4](https://github.com/statsle/SRLR_python/blob/main/code/correlated_features_sketched_estimators.ipynb): Asymptotic risk curves for sketched ridgeless least square estimators with **correlated features**, orthogonal or i.i.d. sketching, as functions of ψ.
- [Figure 5](https://github.com/statsle/SRLR_python/blob/main/code/correlated_features_optimal_m.ipynb): Optimal sketching size selected based on theoretical risk curves. Asymptotic risk curves for the full-sample (no sketching) and sketched ridgeless least square estimators with **correlated features**, orthogonal sketching or i.i.d. sketching, as functions of φ.
- [Figure 6](https://github.com/statsle/SRLR_python/blob/main/code/practical_procedure_independent.ipynb): Practical procedure to pick the best possible sketching size. Asymptotic risk curves for the full-sample (no sketching) and sketched ridgeless least square estimators with isotropic and correlated features and orthogonal sketching as functions of φ. Simulations of correlated features can also be found in [this link](https://github.com/statsle/SRLR_python/blob/main/code/practical_procedure_independent.ipynb). 



## Reference

Chen, X., Zeng, Y., Yang, S. and Sun, Q. Sketched Ridgeless Linear Regression: The Role of Downsampling. [Paper](https://arxiv.org/abs/2302.01088)
