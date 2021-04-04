import matplotlib.pyplot as plt
import numpy as np
import pymc3 as pm
import theano

size = 100
true_intercept = 1
true_slope = 2

x = np.linspace(0, 1, size)
# y = a + b*x
true_regression_line = true_intercept + true_slope * x
# add noise
y = true_regression_line + np.random.normal(scale=0.5, size=size)

# Add outliers
x_out = np.append(x, [0.1, 0.15, 0.2])
y_out = np.append(y, [8, 6, 9])

data = dict(x=x_out, y=y_out)

fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(111, xlabel="x", ylabel="y", title="Generated data and underlying model")
ax.plot(x_out, y_out, "x", label="sampled data")
ax.plot(x, true_regression_line, label="true regression line", lw=2.0)
plt.legend(loc=0);
# plt.show()

with pm.Model() as model:
    pm.glm.GLM.from_formula("y ~ x", data)
    trace = pm.sample(2000, tune=2000, cores=16)

plt.figure(figsize=(7, 5))
plt.plot(x_out, y_out, "x", label="data")
pm.plot_posterior_predictive_glm(trace, samples=100, label="posterior predictive regression lines")
plt.plot(x, true_regression_line, label="true regression line", lw=3.0, c="y")

plt.legend(loc=0);

with pm.Model() as model_robust:
    family = pm.glm.families.StudentT()
    pm.glm.GLM.from_formula("y ~ x", data, family=family)
    trace_robust = pm.sample(2000, tune=2000, cores=16)

plt.figure(figsize=(7, 5))
plt.plot(x_out, y_out, "x")
pm.plot_posterior_predictive_glm(trace_robust, label="posterior predictive regression lines")
plt.plot(x, true_regression_line, label="true regression line", lw=3.0, c="y")
plt.legend();
plt.show()