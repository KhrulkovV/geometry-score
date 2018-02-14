# Geometry Score: A Method For Evaluating Generative Adversarial Networks
Python implementation of the algorithms from the paper.
![manifolds](assets/manif.png)
## Prerequisites

- Python 2.7 or Python 3.3+
- [SciPy](http://www.scipy.org/install.html)
- [NumPy](http://www.numpy.org/)
- [matplotlib](https://matplotlib.org/users/installing.html)
- [GUDHI](http://gudhi.gforge.inria.fr/python/latest/installation.html)
- [Cython](http://cython.org/)

## Basic usage
```python
import numpy as np
import gs
X = np.random.rand(1000, 2)
rlt = gs.rlts(X, L_0=32, gamma=1.0/8, i_max=100, n=100)
mrlt = np.mean(rlt, axis=0)
```
For more details see the [MNIST example](https://github.com/geom-score/geometry-score/blob/master/example-mnist.ipynb) and
[toy examples](https://github.com/geom-score/geometry-score/blob/master/examples-basic.ipynb)
.

