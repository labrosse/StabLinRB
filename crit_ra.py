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
from scipy import integrate
import dmsuite.dmsuite as dm
import matplotlib.pyplot as plt
import seaborn as sns

######## Options #######
# Free slip at both boudaries
COMPUTE_FREESLIP = False
# No slip at both boundaries
COMPUTE_NOSLIP = False
# Rigid bottom, free slip top
COMPUTE_FREERIGID = True
# whether to plot the stream function or use streamplot
COMPUTE_STREAMF = False
NCHEB = 10
FTSZ = 14
######################

def eigval_general(wnk, ranum, ncheb,
                   bcsu=np.array([[1, 0, 0], [1, 0, 0]], float),
                   bcsw=np.array([[1, 0, 0], [1, 0, 0]], float),
                   bcst=np.array([[1, 0, 0], [1, 0, 0]], float),
                   output_eigvec=False):
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
    bcsu=array of boundary conditions for u (default Dirichlet)
    bcsw=array of boundary conditions for w (default Dirichlet)
    bcst=array of boundary conditions for t (default Dirichlet)
    plot_eigvec=True|False (default) whether to return the (right) eigenvector

    OUTPUT
    eigv = eigenvalue with the largest real part (fastest growing if positive).
    eigvec = corresponding eigen vector
    """

    # setup indices for each field depending on BCs
    iu0 = 0
    iun = ncheb
    iw0 = 0
    iwn = ncheb
    it0 = 0
    itn = ncheb

    if bcsu[0, 1] == 0:
        # Dirichlet at z=-1/2
        iu0 = 1
    if bcsw[0, 1] == 0:
        # Dirichlet at z=-1/2
        iw0 = 1

    if bcsu[1, 1] == 0:
        # Dirichlet ar z=1/2
        iun = ncheb-1
    if bcsw[1, 1] == 0:
        # Dirichlet ar z=1/2
        iwn = ncheb-1

    if bcst[0, 1] == 0:
        # Dirichlet at z=-1/2
        it0 = 1
    if bcst[1, 1] == 0:
        # Dirichlet ar z=1/2
        itn = ncheb-1

    # For pressure. No BCs but side values needed or removed
    # depending on the BCs for W. number of lines need to be
    # the same as that of d2w and depends on bcsw.
    if output_eigvec:
            xxt, ddm = dm.chebdif(ncheb, 1, outputx=True)
    else:
            ddm = dm.chebdif(ncheb, 1)

    # compute differentiation matrices
    # For horizontal velocity
    d2u = dm.cheb2bc(ncheb, bcsu)[0]
    # For vertical velocity
    d2w, d1w = dm.cheb2bc(ncheb, bcsw)[0:2]
    # For temperature
    d2t = dm.cheb2bc(ncheb, bcst)[0]

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
    bbig = np.eye(abig.shape[0])
    # 0 for pressure
    bbig[range(d1p.shape[1]), range(d1p.shape[1])] = 1.e-10
    # Find the eigenvalues
    if output_eigvec:
        egv, rvec = linalg.eig(abig, bbig, right=True)
    else:
        egv = linalg.eig(abig, bbig, right=False)
    leig = np.argmax(np.real(egv))
    if output_eigvec:
        return egv[leig], rvec[:, leig], xxt
    else:
        return egv[leig]


def eigval_freeslip(wnk, ranum, ncheb, **kwargs):
    """
    Eigenvalue for given wavenumber and Rayleigh number with Freeslip BCs

    The same result can be obtained using the relevant BCs in eigval_general
    """

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

def stream_function(uvec, wvec, xcoo, zcoo, geometry='cartesian'):
    """
    Computes the stream function from vector field

    INPUT
    uvec : horizontal velocity, 2D array
    wvec : vertical velocity, 2D array
    xcoo : xcoordinate, 1D array
    zcoo : zcoordinate, 1D array
    **kwargs :
    geometry: 'cartesian' (default), 'spherical'

    OUTPUT
    psi : stream function
    """
        
    nnr, nph = uvec.shape
    psi = np.zeros(uvec.shape)
    # integrate first on phi or x
    psi[0, 0] = 0.
    psi[0, 1:nph] = - integrate.cumtrapz(wvec[0, :], xcoo)
    # multiply by rcmb in the spherical case
    if geometry == 'spherical':
        psi[0, 1:nph] = psi[0, 1:nph]*zcoo[0]
    # integrate on r or z
    for iph in range(0, nph):
        psi[1:nnr, iph] = psi[0, iph] + integrate.cumtrapz(uvec[:, iph], zcoo/2)
    psi = psi - np.mean(psi)
    return psi

def plot_mode(kwn, ranum, ncheb, eigfun, title,
                   bcsu=np.array([[1, 0, 0], [1, 0, 0]], float),
                   bcsw=np.array([[1, 0, 0], [1, 0, 0]], float),
                   bcst=np.array([[1, 0, 0], [1, 0, 0]], float),
                   output_eigvec=True, npoints=100):
    """
    Plots the fastest growing mode for wavenumber kwn and ranum

    INPUT
    kwn : wavenumber
    ranum : Rayleigh number
    ncheb : number of Chebyshev points
    eigfun : eigenvalue function
    title : string to use in the pdf file name
    **kwargs : passed on to eigfun
    """
    egv, eigvec, zzr = eigfun(kwn, ranum, ncheb, bcsu=bcsu,
                              bcsw=bcsw, bcst=bcst, output_eigvec=output_eigvec)
    print('Eigenvalue = ', egv)
    # setup indices for each field depending on BCs
    iu0 = ncheb
    iun = 2*ncheb
    if bcsu[0, 1] == 0:
        # Dirichlet at z=-1/2
        iun -= 1
    if bcsu[1, 1] == 0:
        # Dirichlet ar z=1/2
        iun -= 1

    iw0 = iun
    iwn = iun+ncheb
    if bcsw[0, 1] == 0:
        # Dirichlet at z=-1/2
        iwn -= 1
    if bcsw[1, 1] == 0:
        # Dirichlet ar z=1/2
        iwn -= 1

    it0 = iwn
    itn = iwn+ncheb
    if bcst[0, 1] == 0:
        # Dirichlet at z=-1/2
        itn -= 1
    if bcst[1, 1] == 0:
        # Dirichlet ar z=1/2
        itn -= 1
        
    # now split the eigenvector into the different fields
    pmod = eigvec[0:ncheb]/np.max(np.abs(eigvec[0:ncheb]))
    umod = eigvec[iu0:iun]/np.max(np.abs(eigvec[iu0:iun]))
    # add boundary values in Dirichlet case
    if bcsu[0, 1] == 0:
        # Dirichlet at z=-1/2
        umod = np.insert(umod, 0, bcsu[0, 2])
    if bcsu[1, 1] == 0:
        # Dirichlet ar z=1/2
        umod = np.append(umod, bcsu[1, 2])
    wmod = eigvec[iw0:iwn]/np.max(np.abs(eigvec[iw0:iwn]))
    if bcsw[0, 1] == 0:
        # Dirichlet at z=-1/2
        wmod = np.insert(wmod, [0], [bcsw[0, 2]])
    if bcsw[1, 1] == 0:
        # Dirichlet ar z=1/2
        wmod = np.append(wmod, bcsw[1, 2])
    tmod = eigvec[it0:itn]/np.max(np.abs(eigvec[it0:itn]))
    if bcst[0, 1] == 0:
        # Dirichlet at z=-1/2
        tmod = np.insert(tmod, 0, bcst[0, 2])
    if bcst[1, 1] == 0:
        # Dirichlet ar z=1/2
        tmod = np.append(tmod, bcst[1, 2])

    # define the z values on which to interpolate modes
    zpl = np.linspace(-1, 1, npoints)
    # interpolate
    upl = dm.chebint(umod, zpl)
    wpl = dm.chebint(wmod, zpl)
    tpl = dm.chebint(tmod, zpl)
    ppl = dm.chebint(pmod, zpl)

    # plot the norm of the mode profiles
    fig, axe = plt.subplots(1, 4, sharey=True)
    plt.setp(axe, xlim=[-0.1, 1.1], ylim=[-0.5, 0.5], xticks=[0.1, 0.5, 0.9])
    axe[0].plot(np.abs(upl), zpl/2)
    axe[0].plot(np.abs(umod), zzr/2, "o", label=r'$U$')
    axe[0].set_ylabel(r'$z$', fontsize=FTSZ)
    axe[0].set_xlabel(r'$|U|$', fontsize=FTSZ)
    axe[1].plot(np.abs(wpl), zpl/2)
    axe[1].plot(np.abs(wmod), zzr/2, "o", label=r'$W$')
    axe[1].set_xlabel(r'$|W|$', fontsize=FTSZ)
    axe[2].plot(np.abs(tpl), zpl/2)
    axe[2].plot(np.abs(tmod), zzr/2, "o", label=r'$\theta$')
    axe[2].set_xlabel(r'$|\theta|$', fontsize=FTSZ)
    axe[3].plot(np.abs(ppl), zpl/2)
    axe[3].plot(np.abs(pmod), zzr/2, "o", label=r'$P$')
    axe[3].set_xlabel(r'$|P|$', fontsize=FTSZ)
    plt.savefig("Mode_profiles"+title+".pdf", format='PDF')

    # now plot the modes in 2D
    xvar = np.linspace(0, 2*np.pi/kwn, npoints)
    xgr, zgr = np.meshgrid(xvar, zpl)
    zgr = 0.5*zgr
    # temperature
    modx = np.exp(1j*kwn*xvar)
    t2d1, t2d2 = np.meshgrid(modx, tpl)
    t2d = np.real(t2d1*t2d2)
    plt.rcParams['contour.negative_linestyle'] = 'solid'
    plt.figure()
    plt.pcolormesh(xgr, zgr, t2d, cmap='RdBu', linewidth=0,)
    plt.axis([xgr.min(), xgr.max(), zgr.min(), zgr.max()])
    # stream function
    u2d1, u2d2 = np.meshgrid(modx, upl)
    u2d = np.real(u2d1*u2d2)
    w2d1, w2d2 = np.meshgrid(modx, wpl)
    w2d = np.real(w2d1*w2d2)
    if COMPUTE_STREAMF:
        psi = stream_function(u2d, w2d, xvar, zpl/2)
        plt.contour(xgr, zgr, psi)
    else:
        speed = np.sqrt(u2d**2+w2d**2)
        lw = 2*speed/speed.max()
        plt.streamplot(xgr, zgr, u2d, w2d, linewidth=lw, density=0.7)
    # save image
    plt.savefig("mode"+title+".pdf", format='PDF')
    plt.close(fig)
    return

def findplot_rakx(ncheb, eigfun, title, **kwargs):
    """
    Finds the minimum and plots Ra(kx)

    Inputs
    ----------
    ncheb  = number of Chebyshev points in the calculation
    eigfun = name of the eigenvalue finding function
    title  = string variable to use in figure name
    **kwargs: most are just to be passed on to eigfun
    output_eigvec (True|False) controls whether to plot the eigenvector
    of the first ustable mode. Default is False
    """
    if kwargs != {}:
        for key, value in kwargs.items():
            if key == 'output_eigvec':
                plot_eigvec = value
                # alternate kwargs to not output eigenvector
                kwargs2 = kwargs.copy()
                kwargs2['output_eigvec'] = False

    ramin, kxmin = ra_ks(600, 2, ncheb, eigfun, **kwargs2)
    print(title+': Ra=', ramin, 'kx=', kxmin)

    # plot Ra as function of wavenumber
    wnum = np.linspace(0.4, 8, 100)
    rayl = [search_ra(kxwn, ramin, ncheb, eigfun, **kwargs2) for kxwn in wnum]

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

    if plot_eigvec:
        plot_mode(kxmin, ramin, ncheb, eigfun, title, **kwargs)

if COMPUTE_FREESLIP:
    # find the minimum - Freeslip
    findplot_rakx(NCHEB, eigval_freeslip, 'Freeslip')

if COMPUTE_NOSLIP:
    # find the minimum - Noslip
    findplot_rakx(NCHEB, eigval_noslip, 'Noslip')

if COMPUTE_FREERIGID:
    # using the more general function
    findplot_rakx(NCHEB, eigval_general, 'FreeFree',
                  bcsu=np.array([[0, 1, 0], [0, 1, 0]],
                                float), output_eigvec=True)

