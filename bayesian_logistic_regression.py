import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression

N1 = 15
N2 = 15

mu_1 = (1, 1.5)
mu_2 = (3, 1.9)

sigma_1 = (0.5, 0.6)
sigma_2 = (0.3, 0.5)

boots = 100

grain = 1000
pred_spread = 20 # Multiplies the variance out to get a window.
contour_count = 20
contour_accept_eps = 1E-3

X1 = np.c_[np.random.normal(mu_1[0], sigma_1[0], N1),
		   np.random.normal(mu_1[1], sigma_1[1], N1)]
X2 = np.c_[np.random.normal(mu_2[0], sigma_2[0], N2),
		   np.random.normal(mu_2[1], sigma_2[1], N2)]

X = np.r_[X1, X2]
Y = np.r_[0*np.ones(N1), 1*np.ones(N2)]

df = pd.DataFrame(np.c_[X,Y], columns = ('x1', 'x2', 'Y'))

coef_estimates = []
for _ in range(boots):
	model = LogisticRegression()
	boot_df = df.sample(df.shape[0], replace=True)
	model.fit(boot_df[['x1','x2']], boot_df['Y'])
	coef_estimates.append(np.c_[model.coef_, model.intercept_])

x1_preds = np.linspace(min(mu_1[0] - pred_spread*sigma_1[0], mu_2[0] - pred_spread*sigma_2[0]),
					  max(mu_1[0] + pred_spread*sigma_1[0], mu_2[0] + pred_spread*sigma_2[0]), grain)
x2_preds = np.linspace(min(mu_1[1] - pred_spread*sigma_1[1], mu_2[1] - pred_spread*sigma_2[1]),
					  max(mu_1[1] + pred_spread*sigma_1[1], mu_2[1] + pred_spread*sigma_2[1]), grain)

def sigmoid(a):
	return 1/(1 + np.exp(-1*a))

contour_values = np.linspace(0, 1, contour_count)
contour_values = contour_values[1:-1]
contour_plane = []
# print(coef_estimates)
for x1 in x1_preds:
	print(x1)
	for x2 in x2_preds:
		evals = [sigmoid(w[0][0]*x1 + w[0][1]*x2 + w[0][2]) for w in coef_estimates]
		prob = sum(evals) / boots
		for contour in contour_values:
			if abs(prob - contour) < contour_accept_eps:
				contour_plane.append([x1, x2, contour])

contour_df = pd.DataFrame(contour_plane, columns = ('x1', 'x2', 'p'))

plt.scatter(X[:,0], X[:,1], c=Y)
plt.scatter(contour_df['x1'], contour_df['x2'], s=0.1)
plt.show()