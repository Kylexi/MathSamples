import numpy as np
from scipy.stats import beta

np.random.seed(0)

# Simple Monte Carlo method to determine "Which seller should I trust? 
# The one with 90 good/10 bad reviews, or the one with 2 good/0 bad reviews?" 
# Technique employs an uninformative Beta prior.


def MC_estimate(N, reviews, a=1, b=1):
	"""
	N: sample size
	reviews: counts of 'reviews', should be len(reviews) == 4
	a,b: prior distribution params, default to uninformative prior.
	"""
	if len(reviews) != 4:
		raise ValueError("reviews parameter should be of length 4")

	first_pos, first_neg, sec_pos, sec_neg = reviews

	samples = np.c_[beta(first_pos + a,first_neg + b).rvs(N),
					beta(sec_pos + a,sec_neg + b).rvs(N)]
	return samples[samples[:,0] > samples[:,1]].shape[0]/N


print(MC_estimate(50000, (90, 10, 2, 0)))
