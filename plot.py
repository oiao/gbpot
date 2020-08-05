import numpy as np
import matplotlib.pyplot as plt
from gbpot import gb
# %matplotlib inline

range = (-2.5, 2.5, .03)
kappa,kappapr,mu,nu = 1,1,1,1
u1 = np.array([-2,0,0])
u2 = np.array([1,0,0])

U = []
for x in np.arange(*range):
    for y in np.arange(*range):
        dr = np.array([x,y,0])
        U.append(gb(dr,u1,u2, kappa,kappapr,mu,nu))

U      = np.array(U)
U[U>0] = np.nan

X = Y = np.arange(*range)
plt.contourf(X,Y,U.reshape(len(X),len(X)))
plt.gca().set_aspect('equal', 'box')
plt.tight_layout()
