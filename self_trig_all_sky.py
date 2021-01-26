import numpy as np
import healpy as hp
import scipy as sp
import scipy.stats as st

def get_binomial_p(pvals, max_num=10000, return_scan=False, return_num=False):
    """Calculate minimum binomial p-value from our
    list of p-values, scanning over the list of results
        - max_num: maximum number of sources to consider 
            in the scan to save time
    """
    sorted_ps = np.sort(pvals)
    obs_p = 1.
    scan = []
    for i, p in enumerate(sorted_ps[:max_num]):
        tmp = st.binom_test(i+1, len(pvals), p, alternative='greater')
        if tmp < obs_p and tmp != 0.0:
            if tmp == 0.0:
                print("WHY DOES THE BINOMIAL VALUE EQUAL ZERO")
            obs_p = tmp
        if return_scan or return_num:
            scan.append(tmp)
    if return_scan:
        scan = np.asarray(scan)
        return obs_p, scan
    elif return_num:
        return obs_p, np.argmin(np.asarray(scan))
    else:
        return obs_p

def get_one_scan(nside=256, df=2):
    npix = hp.nside2npix(nside)
    pixels = np.arange(npix)
    theta, phi = hp.pix2ang(nside, pixels)
    ra = phi
    dec = np.pi/2. - theta
    northern_sky = dec > np.radians(-5.)

    chi2 = st.chi2(df=df)
    vals = chi2.rvs(size=npix)
    pvals = 1. - chi2.cdf(vals)
    pvals = np.where(northern_sky, pvals, 1.)
    return get_binomial_p(pvals[northern_sky], return_scan=True)

def run_full_binom_trials(num=1, nside=256, df=2, return_num=False):
    npix = hp.nside2npix(nside)
    pixels = np.arange(npix)
    theta, phi = hp.pix2ang(nside, pixels)
    ra = phi
    dec = np.pi/2. - theta
    northern_sky = dec > np.radians(-5.)

    chi2 = st.chi2(df=df)
    vals = chi2.rvs(size=(npix, num))
    pvals = 1. - chi2.cdf(vals)
    pvals = np.where(northern_sky[:, np.newaxis], pvals, 1.)
        
    all_binom_p = []
    if return_num:
        all_k = []
    for pval_list in pvals.T:
        if return_num:
            binom_p, k = get_binomial_p(pval_list[northern_sky], return_num=return_num)
            all_k.append(k)
        else:
            binom_p = get_binomial_p(pval_list[northern_sky], return_num=return_num)
        all_binom_p.append(binom_p)
    if return_num:
        return np.asarray(all_binom_p), np.asarray(all_k)
    return np.asarray(all_binom_p)

def get_one_skymap(nside=256, in_log=True, df=2):
    npix = hp.nside2npix(nside)
    pixels = np.arange(npix)
    theta, phi = hp.pix2ang(nside, pixels)
    ra = phi
    dec = np.pi/2. - theta
    northern_sky = dec > np.radians(-5.)
    chi2 = st.chi2(df=df)
    vals = chi2.rvs(size=npix)
    pvals = 1. - chi2.cdf(vals)
    if in_log:
        log10p = -np.log10(pvals)
        log10p = np.where(northern_sky, log10p, 0.)
        return log10p
    else:
        pvals = np.where(northern_sky, pvals, 1.0)
        return pvals
    