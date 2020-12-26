import numpy as np
from scipy.stats import norm

def generate_data(N, params):
	mu_one, sigma_one, mu_two, sigma_two = params

	data_one = np.random.multivariate_normal(mu_one, sigma_one, N) # Class one
	data_two = np.random.multivariate_normal(mu_two, sigma_two, N) # Class two

	y = np.array([0] * N + [1] * N) # Adding labels

	data = np.concatenate([data_one, data_two])
	data = np.c_[data, y]
	return data

# Generate training data to work with
N = 500

mu_one = [0, 1]
sigma_one = [[1, 0], [0, 1]]

mu_two = [1, 2]
sigma_two = [[2, 0], [0, 2]]

params = (mu_one, sigma_one, mu_two, sigma_two)
data = generate_data(N, params)

# Estimate parameters for independent (Naive Bayes Assumption) distributions
# Since our data is real valued, we will assume Gaussian fits.
m1 = data[:N,0].sum()/N
s1 = ((data[:N,0] - m1)**2).sum()/(N-1)

m2 = data[N:,0].sum()/N
s2 = ((data[N:,0] - m2)**2).sum()/(N-1)

m11 = data[:N,1].sum()/N
s11 = ((data[:N,1] - m11)**2).sum()/(N-1)

m22 = data[N:,1].sum()/N
s22 = ((data[N:,1] - m22)**2).sum()/(N-1)


# Generate test data.
test_data = generate_data(round(N/10), params)

def predict(data, estimates):
	m1, s1, m2, s2, m11, m22 = estimates
	# Ignoring the prior distribution for now since we have balanced data
	prediction = []
	for i in range(data.shape[0]):
		res = norm(m2,s2).pdf(data[i,0]) * norm(m22,s2).pdf(data[i,1])/ (
			norm(m1,s1).pdf(data[i,0]) * norm(m11,s1).pdf(data[i,1]) + norm(m2,s2).pdf(data[i,0]) * norm(m22,s2).pdf(data[i,1])
			)
		prediction.append(round(res))
	return prediction

pred = predict(test_data, (m1, s1, m2, s2, m11, m22))
truth_and_prediction = np.c_[pred, test_data[:,2]]
res = truth_and_prediction

misclass = np.abs(res[:,0] - res[:,1]).sum() / res.shape[0]
print(f"Misclassification error over {round(N/10)} test data points: {misclass}")