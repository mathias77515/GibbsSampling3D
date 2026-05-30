import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import scipy

from src.lib import readPlanckMaps
from src.lib import intensityTemplate
from src.data.getData import dataFolder
from src.lib.sampler import GibbsSampler

# -------------------------------------
# PARAMETERS
# -------------------------------------

nside = 16
FILENAME_FFP_SIMS = "/Users/mregnier/Desktop/toulouse/python/LocalBubble/LocalBubble/data/noise/nside64/100realizations_noise_ffp10_353_GHz.fits"

# -------------------------------------
# READ DATA
# -------------------------------------

# --- Read 3D data ---
intensity = hp.read_map(str(dataFolder) + f"/Intensity_Nside{nside}.fits", field=None)
labels = hp.read_map(str(dataFolder) + f"/Labels_Nside{nside}_alpha0.5000.fits", field=None)
fwhm = 2*np.degrees(hp.nside2resol(nside))

# --- Planck's maps ---
reader = readPlanckMaps.PlanckReader()
Iobs, Qobs, Uobs = reader.read_observed_maps(nside)

# --- Planck's variances from FFP10 simulations ---
varQQobs, varQUobs, varUUobs = readPlanckMaps.read_variance_from_systematics(FILENAME_FFP_SIMS, nside_target=nside)

# -------------------------------------
# MATRICES BUILDER
# -------------------------------------

Ibuilder = intensityTemplate.IntensityTemplateBuilder(labels=labels, intensity=intensity)
H = Ibuilder.build(Iobs)
H = scipy.sparse.csr_matrix(H)

# --- Inverse of dense block-diagonal matrix ---
det = varQQobs * varUUobs - varQUobs**2
wQQFull =  varUUobs / det
wUUFull =  varQQobs / det
wQUFull = -varQUobs / det

# --- Noise weights -> 1/sigma^2 ---
w = scipy.sparse.bmat([
    [scipy.sparse.diags(wQQFull), scipy.sparse.diags(wQUFull)],
    [scipy.sparse.diags(wQUFull), scipy.sparse.diags(wUUFull)]
], format="csr")  # shape (2*npix, 2*npix)

# --- Noise weights -> 1/sigma ---
wsqrt = scipy.sparse.bmat([
    [scipy.sparse.diags(wQQFull**0.5), scipy.sparse.diags(wQUFull*0)],
    [scipy.sparse.diags(wQUFull*0), scipy.sparse.diags(wUUFull**0.5)]
], format="csr")  # shape (2*npix, 2*npix)


H = scipy.sparse.bmat([[H, None], 
                       [None, H]])

d = np.concatenate((Qobs, Uobs))

gibbs   = GibbsSampler(H, w, d, wsqrt, maxiterCG=10)
results = gibbs.run(niter=500)
