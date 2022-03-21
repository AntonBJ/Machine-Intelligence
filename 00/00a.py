import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import math

x0 = np.arange(-8,8, 0.01)
mu = 0
variance = 1
sigma = math.sqrt(variance)

pdf = stats.norm.pdf(x0, mu, sigma)

print(len(pdf))

plt.plot(x0, pdf, color='blue', label='Standard Normal Distribution')

plt.ylabel('pdf(x)')
plt.xlabel('x')
plt.title('Exercise H0.1')

plt.show()


