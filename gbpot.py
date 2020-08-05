import numpy as np

def gbpot(dr, u1, u2, kappa=1., kappapr=1., mu=1., nu=1.):
    """
    Implementation of the Gay-Berne potential [1]_, [2]_.

    Parameters
    ----------
    dr : ndarray
        A `(m,n)` array of pairwise distance vectors
    u1 : ndarray
        A `(m,n)` array of the `i`th particle orientation vectors
    u2 : ndarray
        A `(m,n)` array of the `j`th particle orientation vectors
    kappa: float > 0
        Shape anisotropy parameter
    kappapr: float > 0
        Interaction anisotropy paramter
    mu : float
        Anisotropic potential locality
    nu: float
        Anisotropic potential depth


    References
    ----------
    .. [1] Gay, J. G.; Berne, B. J. J. Chem. Phys. 1981, 74, 3316–3319
    .. [2] Bates, M. A.; Luckhurst, G. R. J. Chem. Phys. 1996, 104, 6696–6709
    """
    eps0, sigma0, sigma_ff = 1., 1., 1.
    dr    = np.atleast_2d(dr)
    u1    = np.atleast_2d(u1)
    u2    = np.atleast_2d(u2)
    norm  = np.linalg.norm(dr, axis=-1)
    dr    = dr/norm[:,None]
    u1    = u1 / np.linalg.norm(u1, axis=-1)[:,None]
    u2    = u2 / np.linalg.norm(u2, axis=-1)[:,None]
    u1r   = np.einsum('ij,ij->i', u1, dr)
    u2r   = np.einsum('ij,ij->i', u2, dr)
    u1u2  = np.einsum('ij,ij->i', u1, u2)
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
