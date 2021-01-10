import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from numpy.linalg import inv

# Controlling randomness
np.random.seed(0)

# Parameters
x_spread = 200
N = 200
sigma_2 = 70
tau_2 = 1000
granularity = 10*x_spread


# First, let's distribute our data (1-d) along the x-axis
# There are a couple options to play with the spread
# Uncomment as needed.

# x = np.random.uniform(0, x_spread, N)
x = np.random.normal(x_spread/2, x_spread/10, N)
# x = np.r_[np.random.normal(x_spread/4, x_spread/20, int(3*N/4)),
# 		  np.random.normal(3*x_spread/4, x_spread/20, int(N/4))]

# Next, we'll contrive our data.
e = np.random.normal(0, sigma_2, N)
y = -2*x + 0.05*x**2 + e


# Let's expand the basis we are trying to fit to capture
# the squaring effect.
X = np.c_[np.ones(N), x, x**2]

# We'll assume that our prior on our regression
# weights is N(0, tau_2). This is synonymous with
# ridge regression. Below are the posterior mean
# (V_N) and posterior variance (V_N)

D = X.shape[1]
V_N = sigma_2*inv(sigma_2/tau_2 * np.eye(D) + X.T.dot(X))
w_N = 1/sigma_2 * V_N.dot(X.T).dot(y-np.mean(y))

print(w_N)

# Our 'point estimate' is now the posterior mean.
# We can check that this roughly aligns with our
# contrived y's. However, the real benefit of Bayesian
# regression is in the ability to compute the error bars
# better.

pred_x = np.linspace(0, x_spread, granularity)
pred_x = np.c_[np.ones(granularity), pred_x, pred_x**2]

sigma_2_bayes = np.diag(sigma_2 + pred_x.dot(V_N).dot(pred_x.T))
print(sigma_2_bayes.shape)
l_bound = w_N.T.dot(pred_x.T) - 2*sigma_2_bayes
u_bound = w_N.T.dot(pred_x.T) + 2*sigma_2_bayes

plt.scatter(x=pred_x[:,1], y=w_N.dot(pred_x.T), s=0.1)
plt.scatter(x=x, y=y-np.mean(y), c='red', s=0.3)
plt.scatter(x=pred_x[:,1], y=l_bound, c='cyan', s=0.1)
plt.scatter(x=pred_x[:,1], y=u_bound, c='cyan', s=0.1)
plt.show()