#!/usr/bin/env python3
"""
Finds critical Rayleigh number.

Plots the Critical Ra as function of wave-number and finds the minimum.
Based on the matlab code provided by Thierry Alboussiere.
Can do both the no-slip and free-slip BCs, applying to both boundaries.
"""
import math
import numpy as np
from scipy import linalg
import dmsuite.dmsuite as dm
import matplotlib.pyplot as plt
import seaborn as sns
import sprintf

######## Options #######
# Free slip at both boudaries
COMPUTE_FREESLIP = False
# No slip at both boundaries
COMPUTE_NOSLIP = False
COMPUTE_FREERIGID = True
NCHEB = 10
FTSZ = 14
######################

def eigval_general(wnk, ranum, ncheb, **kwargs):
    """
    Eigenvalue for given wavenumber and Rayleigh number with general BCs

    The boundary conditions are
    a_1 f(1) + b_1 f'(1)  = c_1
    a_N f(-1) + b_N f'(-1) = c_N
    for each of the following variables: u (horizontal velocity),
    w (vertical velocity), t (temperature)
    The values are passed as part of the **kwargs, with keywords
    bcsX for variable X.
    Default values are the Dirichlet conditions for all variables at both
    boundaries, ie: bcsX = np.array([[1, 0, 0], [1, 0, 0]], float)

    INPUT
    wnk = horizontal wavenumber of the perturbation
    ranum = Rayleigh number
    ncheb = number of Chebyshev points in vertical direction

    OUTPUT
    eigv = eigenvalue with the largest real part (fastest growing if positive).
    """
    # Default boundary conditions. 
    bcsu = np.array([[1, 0, 0],[1, 0, 0]], float)
    bcsw = np.array([[1, 0, 0],[1, 0, 0]], float)
    bcst = np.array([[1, 0, 0],[1, 0, 0]], float)

    if kwargs != {}:
        for key, value in kwargs.items():
            if key == 'bcsu':
                bcsu = value
            elif key == 'bcsw':
                bcsw = value
            elif key == 'bcst':
                bcst = value
            else:
                print("kwarg value not understood %s == %s" %(key, value))
                print("ignored")


    # setup indices for each field depending on BCs
    iu0 = 0
    iun = ncheb
    if bcsu[0, 1] == 0:
        # Dirichlet at z=-1/2
        iu0 = 1
    if bcsu[1, 1] == 0:
        # Dirichlet ar z=1/2
        iun = ncheb-1

    iw0 = 0
    iwn = ncheb
    if bcsw[0, 1] == 0:
        # Dirichlet at z=-1/2
        iw0 = 1
    if bcsw[1, 1] == 0:
        # Dirichlet ar z=1/2
        iwn = ncheb-1

    it0 = 0
    itn = ncheb
    if bcst[0, 1] == 0:
        # Dirichlet at z=-1/2
        it0 = 1
    if bcst[1, 1] == 0:
        # Dirichlet ar z=1/2
        itn = ncheb-1

    # compute differentiation matrices
    # For horizontal velocity
    d2u, d1u, phipu, phimu = dm.cheb2bc(ncheb, bcsu)
    # For vertical velocity
    d2w, d1w, phipw, phimw = dm.cheb2bc(ncheb, bcsw)
    # For temperature
    d2t, d1t, phipw, phimw = dm.cheb2bc(ncheb, bcst)
    # For pressure. No BCs but side values needed or removed
    # depending on the BCs for W. number of lines need to be
    # the same as that of d2w and depends on bcsw.
    ddm = dm.chebdif(ncheb, 1)
    d1p = 2.*ddm[0, iw0:iwn, :]
    d1w = 2.*ddm[0, :, iw0:iwn]

    # identity
    ieye = np.eye(ncheb, ncheb)

    # scale to [-0.5, 0.5]
    d2u = 4.*d2u
    d2w = 4.*d2w
    d2t = 4.*d2t

    # Prandtl number
    pra = 1.
    # square of the wave number
    wn2 = wnk**2.
    np.set_printoptions(precision=2, suppress=True, formatter={'float': '{: 0.2f}'.format})
    # construction of the lhs matrix. Submatrices' dimensions must match
    # first line: p
    app = np.zeros((ncheb, ncheb))
    apu = -1j*wnk*ieye[:, iu0:iun]
    apw = -d1w
    apt = np.zeros((ncheb, itn-it0))
    # second line: u
    aup = -1j*wnk*ranum*ieye[iu0:iun, :]
    auu = pra*(d2u-wn2*ieye[iu0:iun, iu0:iun])
    auw = np.zeros((iun-iu0, iwn-iw0))
    aut = np.zeros((iun-iu0, itn-it0))
    # third line: w
    awp = -ranum*d1p
    awu = np.zeros((iwn-iw0, iun-iu0))
    aww = pra*(d2w-wn2*ieye[iw0:iwn, iw0:iwn])
    awt = -ranum*ieye[iw0:iwn, it0:itn]
    # fourth line: t
    atp = np.zeros((itn-it0, ncheb))
    atu = np.zeros((itn-it0, iun-iu0))
    # Assume here the basic state to be linear of z
    atw = -ieye[it0:itn, iw0:iwn]
    att = d2t-wn2*ieye[it0:itn, it0:itn]
    abig = np.concatenate((np.concatenate((app, apu, apw, apt), axis=1),
                            np.concatenate((aup, auu, auw, aut), axis=1),
                            np.concatenate((awp, awu, aww, awt), axis=1),
                            np.concatenate((atp, atu, atw, att), axis=1),
                            ))

# construction of the rhs matrix: identity almost everywhere
    nbig = abig.shape[0]
    nsmall = nbig-app.shape[0]
    bbig = np.eye(nbig)
    # 0 for pressure
    bbig[range(d1p.shape[1]), range(d1p.shape[1])] = 1.e-10
    # Find the eigenvalues
    egv, rvec = linalg.eig(abig, bbig, right=True)
    leig = np.argmax(np.real(egv))
    return egv[leig]


def eigval_freeslip(wnk, ranum, ncheb, **kwargs):
    """eigenvalue for given wavenumber and Rayleigh number with Freeslip BCs"""

    # second order derivative.
    ddm = dm.chebdif(ncheb+2, 2)
    # Freeslip BCs obtained by excluding boundary points.
    # factor 2 because reference interval is [-1,1]
    dd2 = 4.*ddm[1, 1:ncheb+1, 1:ncheb+1]
    # identity
    ieye = np.eye(dd2.shape[0])

    # square of the wave number
    wn2 = wnk**2.

    auzuz = dd2-wn2*ieye
    auzv = -ieye
    auzt = np.zeros(dd2.shape)

    avuz = np.zeros(dd2.shape)
    avv = dd2-wn2*ieye
    avt = ranum*wn2*ieye

    atuz = -ieye
    atv = np.zeros(dd2.shape)
    att = dd2-wn2*ieye

    abig = np.concatenate((np.concatenate((auzuz, auzv, auzt)),
                           np.concatenate((avuz, avv, avt)),
                           np.concatenate((atuz, atv, att))), axis=1)


    egv = linalg.eig(abig, right=False)
    lmbda = -np.sort(-np.real(egv))
    return lmbda[0]

def eigval_noslip(wnk, ranum, ncheb, **kwargs):
    """eigenvalue for given wavenumber and Rayleigh number for Noslip BCs"""

    # second order derivative.
    ddm = dm.chebdif(ncheb+2, 2)
    # Freeslip BCs for temperature
    # factor 2 because reference interval is [-1,1]
    dd2 = 4.*ddm[1, 1:ncheb+1, 1:ncheb+1]
    # Clamped BCs for W: W=0 and W'=0
    dd4 = 16.*dm.cheb4c(ncheb+2)
    # identity
    ieye = np.eye(dd2.shape[0])

    # square of the wave number
    wn2 = wnk**2.
    wn4 = wn2**2.

    aww = dd4-2.*wn2*dd2+wn4*ieye
    awt = ranum*wn2*ieye
    atw = -ieye
    att = dd2-wn2*ieye

    abig = np.concatenate((np.concatenate((-aww, -awt)),
                           np.concatenate((atw, att))), axis=1)

    egv = linalg.eig(abig, right=False)
    lmbda = -np.sort(-np.real(egv))
    return lmbda[0]

def search_ra(kwn, ray, ncheb, eigfun, **kwargs):
    """find rayleigh number ray which gives neutral stability"""
    ray0 = ray/math.sqrt(1.2)
    ray1 = ray*math.sqrt(1.2)
    la0 = np.real(eigfun(kwn, ray0, ncheb, **kwargs))
    la1 = np.real(eigfun(kwn, ray1, ncheb, **kwargs))
    while la0 > 0. or la1 < 0.:
        if la0 > 0.:
            ray1 = ray0
            ray0 = ray0/.2
        if la1 < 0.:
            ray0 = ray1
            ray1 = 2.*ray1
        la0 = np.real(eigfun(kwn, ray0, ncheb, **kwargs))
        la1 = np.real(eigfun(kwn, ray1, ncheb, **kwargs))
    while la1-la0 > 0.001:
        raym = (ray0+ray1)/2.
        lam = np.real(eigfun(kwn, raym, ncheb, **kwargs))
        if lam < 0.:
            la0 = lam
            ray0 = raym
        else:
            la1 = lam
            ray1 = raym
    return (ray0*la1-ray1*la0)/(la1-la0)

def ra_ks(rag, wng, ncheb, eigfun, **kwargs):
    """finds the minimum in the Ra-wn curve"""
    # find 3 values of Ra for 3 different wave numbers
    eps = [0.1, 0.01]
    wns = np.linspace(wng-eps[0], wng+2*eps[0], 3)
    ray = [search_ra(kkx, rag, ncheb, eigfun, **kwargs) for kkx in wns]

    # fit a degree 2 polynomial
    pol = np.polyfit(wns, ray, 2)

    # minimum value
    kmin = -0.5*pol[1]/pol[0]
    for i, err in enumerate([0.03, 0.001]):
        while np.abs(kmin-wns[1]) > err:
            wns = np.linspace(kmin-eps[i], kmin+eps[i], 3)
            ray = [search_ra(kkx, rag, ncheb, eigfun, **kwargs) for kkx in wns]
            pol = np.polyfit(wns, ray, 2)
            kmin = -0.5*pol[1]/pol[0]
            rag = ray[1]

    return rag, kmin

def findplot_rakx(ncheb, eigfun, title, **kwargs):
    """
    Finds the minimum and plots Ra(kx)

    Inputs
    ----------
    ncheb  = number of Chebyshev points in the calculation
    eigfun = name of the eigenvalue finding function
    title     = string variable to use in figure name
    """
    ramin, kxmin = ra_ks(600, 2, ncheb, eigfun, **kwargs)
    print(title+': Ra=', ramin, 'kx=', kxmin)

    # plot Ra as function of wavenumber
    wnum = np.linspace(0.4, 8, 100)
    rayl = [search_ra(kxwn, ramin, ncheb, eigfun, **kwargs) for kxwn in wnum]

    fig = plt.figure()
    plt.plot(wnum, rayl, linewidth=2)
    plt.plot(kxmin, ramin, 'o', label=r'$Ra_{min}=%.2f ; k_x=%.2f$' %(ramin, kxmin))
    plt.xlabel('Wavenumber', fontsize=FTSZ)
    plt.ylabel('Rayleigh number', fontsize=FTSZ)
    plt.xticks(fontsize=FTSZ)
    plt.yticks(fontsize=FTSZ)
    plt.legend(loc='upper right', fontsize=FTSZ)
    plt.savefig('Ra_kx_'+title+'.pdf', format='PDF')
    plt.close(fig)


if COMPUTE_FREESLIP:
    # find the minimum - Freeslip
    findplot_rakx(NCHEB, eigval_freeslip, 'Freeslip')

if COMPUTE_NOSLIP:
    # find the minimum - Noslip
    findplot_rakx(NCHEB, eigval_noslip, 'Noslip')

if COMPUTE_FREERIGID:
    # using the more general function
    findplot_rakx(NCHEB, eigval_general, 'FreeRigid', bcsu=np.array([[1, 0, 0],[0, 1, 0]], float))

