import numpy as np
from scipy.stats import norm
from sklearn.naive_bayes import GaussianNB

np.random.seed(0)

def generate_data(N, params):
	mu_one, sigma_one, mu_two, sigma_two = params

	data_one = np.random.multivariate_normal(mu_one, sigma_one, N) # Class one
	data_two = np.random.multivariate_normal(mu_two, sigma_two, N) # Class two

	y = np.array([0] * N + [1] * N) # Adding labels

	data = np.concatenate([data_one, data_two])
	data = np.c_[data, y]
	return data

# Generate training data to work with
N = 500000

mu_one = [-1, 3]
sigma_one = [[3, 0], [0, 1]]

mu_two = [1, 2]
sigma_two = [[4, 0], [0, 2]]

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
	m1, s1, m2, s2, m11, s11, m22, s22 = estimates
	# Ignoring the prior distribution for now since we have balanced data
	res = norm(m2,s2).pdf(data[:,0]) * norm(m22,s22).pdf(data[:,1])/ (
			norm(m1,s1).pdf(data[:,0]) * norm(m11,s11).pdf(data[:,1]) + 
			norm(m2,s2).pdf(data[:,0]) * norm(m22,s22).pdf(data[:,1])
			)
	return np.round(res)

pred = predict(test_data, (m1, s1, m2, s2, m11, s11, m22, s22))
truth_and_prediction = np.c_[pred, test_data[:,2]]
res = truth_and_prediction

misclass = np.abs(res[:,0] - res[:,1]).sum() / res.shape[0]
print(f"Misclassification error over {round(N/10)} test data points: {misclass}")

gnb = GaussianNB()
gnb.fit(data[:, 0:2], data[:, 2])
y_pred = gnb.predict(test_data[:, 0:2])
misclass = (y_pred != test_data[:, 2]).sum() / res.shape[0]
print(f"Misclassification error over {round(N/10)} test data points: {misclass} (sklearn)")