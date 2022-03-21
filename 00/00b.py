import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

# Our 2-dimensional distribution will be over variables X1 and X2
N = 60
X1 = np.linspace(-8, 8, N)
X2 = np.linspace(-8, 8, N)
X1, X2 = np.meshgrid(X1, X2)

# Mean vector and covariance matrix
mu = np.array([0, 0])
Sigma = np.array([[ 1 , 0.5], [0.5,  1]])

# Pack X and Y into a single 3-dimensional array
pos = np.empty(X1.shape + (2,))
pos[:, :, 0] = X1
pos[:, :, 0] = X2

def multivariate_gaussian(pos, mu, Sigma):
    """Return the multivariate Gaussian distribution on array pos.

    pos is an array constructed by packing the meshed arrays of variables
    x_1, x_2, x_3, ..., x_k into its _last_ dimension.

    """

    n = mu.shape[0]
    Sigma_det = np.linalg.det(Sigma)
    Sigma_inv = np.linalg.inv(Sigma)
    N = np.sqrt((2*np.pi)**n * Sigma_det)
    # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
    # way across all the input variables.
    fac = np.einsum('...k,kl,...l->...', pos-mu, Sigma_inv, pos-mu)

    return np.exp(-fac / 2) / N

# The distribution on the variables X, Y packed into pos.
Z = multivariate_gaussian(pos, mu, Sigma)

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.contourf(X1, X2, Z, zdir='z', cmap=cm.viridis)


plt.show()

# Create a surface plot and projected filled contour plot under it.
# fig = plt.figure()

# ax = fig.gca(projection='3d')
# ax.plot_surface(X1, X2, Z, rstride=3, cstride=3, linewidth=1, antialiased=True,
#                 cmap=cm.viridis)

# cset = ax.contourf(X1, X2, Z, zdir='z', offset=-0.15, cmap=cm.viridis)

# # Adjust the limits, ticks and view angle
# ax.set_zlim(-0.15,0.2)
# ax.set_zticks(np.linspace(0,0.2,5))
# ax.view_init(27, -21)

# plt.show()