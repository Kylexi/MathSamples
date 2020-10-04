import numpy as np
import matplotlib.pyplot as plt

# This demo script will demonstrate how to apply a locally weighted least squares fit

# First, we generate some sample data that follows a sine curve
# Since we are using a sine curve, we'll likely expect this locally weighted fitting
# to not quite capture the peaks and valleys. We should see the general wave still, however.
N = 500
n = 100
x = 2*np.pi*np.random.rand(N)
y = np.sin(x) + np.random.normal(0, 0.2, (N,))

# Let's define the points we want to evalute our fit predictions
x_0 = np.linspace(0, 2*np.pi, n)

def K(x_0, x):
	"""Simple Gaussian Kernel method"""
	ret = 1/np.sqrt(2*np.pi)*np.exp(-(x_0*np.ones(x.shape[0]) - x)**2/2)
	return ret

def fit(x_0, x, y, K):
	"""Since we need to define a fitting method for each data point in x_0,
	we first write a method to do it for a generic item in x_0.

	x_0: float
	x: [... floats ...]
	y: [... floats ...] same size as x
	K: kernel smoother, must take in two scalar inputs and return a distance
	"""

	W = np.diag(K(x_0, x))
	b = np.array((1, x_0)).T
	B = np.array((np.ones(x.shape[0]),x)).T
	Bt = B.T
	S = np.linalg.inv(Bt.dot(W).dot(B))
	ret = b.dot(S).dot(Bt).dot(W).dot(y)
	return ret

y_fit = [fit(x_i, x, y, K) for x_i in x_0] # Apply fitting method across all x_0


plt.scatter(x, y, s=2) 
plt.plot(x_0, y_fit, c="r")
plt.show()