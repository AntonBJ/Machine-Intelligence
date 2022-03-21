import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd

data = pd.read_csv('Data/applesOranges.csv')

n=0

for n in len(data):
	x[n]=data[]


x = data['x.1']
y = data['x.2']

plt.plot(x, y, 'o', color='black')

plt.ylabel('x.2')
plt.xlabel('x.1')
plt.title('Exercise H0.2')

plt.show()
