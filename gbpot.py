import numpy as np
np.seterr(all='raise')

def gb(dr, u1, u2, kappa=1, kappapr=1, mu=1, nu=1):
    eps0, sigma0, sigma_ff = 1., 1., 1.
    norm  = np.linalg.norm(dr)
    dr    = dr/norm
    u1    = u1 / np.linalg.norm(u1)
    u2    = u2 / np.linalg.norm(u2)
    u1r, u2r, u1u2 = u1.dot(dr), u2.dot(dr), u1.dot(u2)
    pur   = (u1r + u2r)**2
    nur   = (u1r - u2r)**2

    kappa2     = kappa**2
    kappapr_mu = kappapr**(1/mu)
    chi   = (kappa2     - 1) / (kappa2     + 1)
    chipr = (kappapr_mu - 1) / (kappapr_mu + 1)

    chi_u1u2 = chi*u1u2
    p, n  = pur / (1 + chi_u1u2), nur / (1 - chi_u1u2)
    sigma = sigma0 / np.sqrt(1 - .5*chi*(p+n))

    chipr_u1u2 = chipr*u1u2
    p, n  = pur / (1 + chipr_u1u2), nur / (1 - chipr_u1u2)
    eps1  = 1/np.sqrt(1 - (chi**2) * (u1u2**2))
    eps2  = 1 - .5*chipr*(p+n)
    eps   = eps0 * (eps1**nu) * (eps2**mu)

    rho   = sigma_ff / (norm - sigma + sigma_ff)
    rho6  = rho**6
    rho12 = rho6**2

    return 4*eps*(rho12-rho6)


x = np.random.randn(3,3)

np.linalg.norm(x,axis=-1)
