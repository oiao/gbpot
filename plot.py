import numpy as np
import matplotlib.pyplot as plt
from gbpot import gbpot

range = (-2.5, 2.5, .01)
u1 = np.array([1,0,0])
u2 = np.array([1,0,0])
xx, yy, zz = np.meshgrid(np.arange(*range),np.arange(*range),[0])
P          = np.array((xx,yy,zz)).T.reshape(-1,3)

params = (
    (1.5,1., 1., 1.),
    (.5, 1., 1., 1.),
    (.5, 5., 1., 1.),
    (.5, .5, 1., 1.),
    (1.2, .5,10.,1.),
    (1.2, .5,.1 ,1.),
    (.5, .5, 1.,10.),
    (.5, .5, 1., .1),
    )

fig, ax = plt.subplots(4,2, figsize=(9.5, 20))
ax = ax.flatten()
for i,p in enumerate(params):
    U       = gbpot(P,u1,u2, *p)
    U[U>0]  = 0

    im      = ax[i].contour(xx[:,:,0], yy[:,:,0], U.reshape(len(xx),len(xx)), levels=10)
    ax[i].text(0.01, 0.94, f"$\kappa={p[0]}, \kappa'={p[1]}, \mu={p[2]}, \\nu={p[3]}$", transform=ax[i].transAxes)
    ax[i].text(0.01, 0.88,  f"Umin = {U.min():.2e}", transform=ax[i].transAxes)
    ax[i].set_aspect('equal', 'box')
    ax[i].set_xticks([]); ax[i].set_yticks([])

plt.tight_layout()
plt.savefig('isopots.png', dpi=200)
