#!/usr/bin/env python
"""
Finds critical Rayleigh number.

Plots the Critical Ra as function of wave-number and finds the minimum.
Based on the matlab code provided by Thierry Alboussiere.
Can do both the no-slip and free-slip BCs, applying to both boundaries.
"""
import math
import numpy as np
from scipy import linalg
import pyddx.sc.dmsuite as dm
import matplotlib.pyplot as plt
import seaborn as sns

######## Options #######
COMPUTE_FREESLIP = True
COMPUTE_NOSLIP = True
NCHEB = 10
FTSZ = 14

######################
def eigval_freeslip(wnk, ranum, ncheb):
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

def eigval_noslip(wnk, ranum, ncheb):
    """eigenvalue for given wavenumber and Rayleigh number for Noslip BCs"""

    # second order derivative.
    ddm = dm.chebdif(ncheb+2, 2)
    # Freeslip BCs for temperature
    # factor 2 because reference interval is [-1,1]
    dd2 = 4.*ddm[1, 1:ncheb+1, 1:ncheb+1]
    # Clamped BCs for W: W=0 and W'=0
    d4c = 16.*dm.cheb4c(ncheb+2)
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

def search_ra(kwn, ray, ncheb, eigfun):
    """find rayleigh number ray which gives neutral stability"""
    ray0 = ray/math.sqrt(1.2)
    ray1 = ray*math.sqrt(1.2)
    la0 = eigfun(kwn, ray0, ncheb)
    la1 = eigfun(kwn, ray1, ncheb)
    while la0 > 0. or la1 < 0.:
        if la0 > 0.:
            ray1 = ray0
            ray0 = ray0/.2
        if la1 < 0.:
            ray0 = ray1
            ray1 = 2.*ray1
        la0 = eigfun(kwn, ray0, ncheb)
        la1 = eigfun(kwn, ray1, ncheb)
    while la1-la0 > 0.001:
        raym = (ray0+ray1)/2.
        lam = eigfun(kwn, raym, ncheb)
        if lam < 0.:
            la0 = lam
            ray0 = raym
        else:
            la1 = lam
            ray1 = raym
    return (ray0*la1-ray1*la0)/(la1-la0)

def ra_ks(rag, wng, ncheb, eigfun):
    """finds the minimum in the Ra-wn curve"""
    # find 3 values of Ra for 3 different wave numbers
    eps = [0.1, 0.01]
    wns = np.linspace(wng-eps[0], wng+2*eps[0], 3)
    ray = [search_ra(kkx, rag, ncheb, eigfun) for kkx in wns]

    # fit a degree 2 polynomial
    pol = np.polyfit(wns, ray, 2)

    # minimum value
    kmin = -0.5*pol[1]/pol[0]
    for i, err in enumerate([0.03, 0.001]):
        while np.abs(kmin-wns[1]) > err:
            wns = np.linspace(kmin-eps[i], kmin+eps[i], 3)
            ray = [search_ra(kkx, rag, ncheb, eigfun) for kkx in wns]
            pol = np.polyfit(wns, ray, 2)
            kmin = -0.5*pol[1]/pol[0]
            rag = ray[1]

    return rag, kmin

if COMPUTE_FREESLIP:
    # find the minimum - Freeslip
    ramin, kxmin = ra_ks(600, 2, NCHEB, eigval_freeslip)
    print 'Free-slip: Ra=', ramin, 'kx=', kxmin

    # plot Ra as function of wavenumber
    wnum = np.linspace(0.3, 8, 100)
    rayl = [search_ra(kxwn, 600, NCHEB, eigval_freeslip) for kxwn in wnum]

    fig = plt.figure()
    plt.plot(wnum, rayl, linewidth=2)
    plt.plot(kxmin, ramin, 'o', label=r'$Ra_{min}=%.2f ; k_x=%.2f$' %(ramin, kxmin))
    plt.xlabel('Wavenumber', fontsize=FTSZ)
    plt.ylabel('Rayleigh number', fontsize=FTSZ)
    plt.xticks(fontsize=FTSZ)
    plt.yticks(fontsize=FTSZ)
    plt.legend(loc='upper right', fontsize=FTSZ)
    plt.savefig('Ra_kx_Freeslip.pdf', format='PDF')
    plt.close(fig)

if COMPUTE_NOSLIP:
    # find the minimum - Noslip
    ramin, kxmin = ra_ks(1700, 2, NCHEB, eigval_noslip)
    print 'No-slip: Ra=', ramin, 'kx=', kxmin

    # plot Ra as function of wavenumber
    wnum = np.linspace(0.3, 8, 100)
    rayl = [search_ra(kxwn, 600, NCHEB, eigval_noslip) for kxwn in wnum]

    fig = plt.figure()
    plt.plot(wnum, rayl, linewidth=2)
    plt.plot(kxmin, ramin, 'o', label=r'$Ra_{min}=%.2f ; k_x=%.2f$' %(ramin, kxmin))
    plt.xlabel('Wavenumber', fontsize=FTSZ)
    plt.ylabel('Rayleigh number', fontsize=FTSZ)
    plt.xticks(fontsize=FTSZ)
    plt.yticks(fontsize=FTSZ)
    plt.legend(loc='upper right', fontsize=FTSZ)
    plt.savefig('Ra_kx_Noslip.pdf', format='PDF')
