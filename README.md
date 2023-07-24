# SRLR

Sketched Ridgeless Linear Regression

## Description

This repository presents numerical simulations that analyze the empirical risks of the sketched ridgeless estimator, aiming to enhance generalization performance. The simulations focus on determining optimal sketching sizes that minimize out-of-sample prediction risks. The results reveal that the optimally sketched estimator exhibits stable risk curves, effectively eliminating the peaks observed in the full-sample estimator. Additionally, we introduce a practical procedure to empirically identify the optimal sketching size.

Suppose we observe data vectors  (x<sub>i</sub>,y<sub>i</sub>) that follow a linear model y<sub>i</sub>=x<sub>i</sub><sup>T</sup>&beta;<sup>*</sup>+&epsilon;<sub>i</sub>, i=1,...n, where y<sub>i</sub> is a univariate response,  x<sub>i</sub> is a d-dimensional predictor, &beta;<sup>*</sup> denotes the vector of regression coefficients, and &epsilon;<sub>i</sub> is a random error. We consider the ridgeless least square estimator β̂=(X<sup>T</sup>X)<sup>+</sup>X<sup>T</sup>Y.

With this package, the simulation results in [this paper](https://arxiv.org/abs/2302.01088) can be reporduced.

## Examples

The sketched least square estimator is implemented as follows:

```Python
sketched = estimators.SketchedRidgelessRegressor()
```

For further details, please see `tutorial.ipynb` as an example. 



## Reference

Chen, X., Zeng, Y., Yang, S. and Sun, Q. Sketched Ridgeless Linear Regression: The Role of Downsampling. [Paper](https://arxiv.org/abs/2302.01088)
