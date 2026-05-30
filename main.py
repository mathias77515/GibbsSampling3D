import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import scipy
import pickle
import os
import time

from src.lib import readPlanckMaps
from src.lib import intensityTemplate
from src.data.getData import dataFolder
from src.lib.sampler import GibbsSampler
from src.utils import parser

# -------------------------------------
# PARAMETERS
# -------------------------------------

args       = parser.parse_args()
FILENAME_FFP_SIMS = "/Users/mregnier/Desktop/toulouse/python/LocalBubble/LocalBubble/data/noise/nside64/100realizations_noise_ffp10_353_GHz.fits"
FOLDERSAVE = ""
chainID    = int(os.environ.get("SLURM_ARRAY_TASK_ID", 1))

# -------------------------------------
# PARAMETERS
# -------------------------------------

t0_total = time.perf_counter()

print("=" * 60)
print("GIBBS SAMPLING RECONSTRUCTION")
print("=" * 60)
print(f"Nside    : {args.nside}")
print(f"Chain ID : {chainID}")
print("=" * 60)

# -------------------------------------
# READ DATA
# -------------------------------------

t0 = time.perf_counter()
print("\n[1/4] Reading data...")

# --- Read 3D data ---
intensity = hp.read_map(str(dataFolder) + f"/Intensity_Nside{args.nside}.fits", field=None)
labels = hp.read_map(str(dataFolder) + f"/Labels_Nside{args.nside}_alpha0.5000.fits", field=None)
fwhm = 2*np.degrees(hp.nside2resol(args.nside))

# --- Planck maps ---
reader           = readPlanckMaps.PlanckReader()
Iobs, Qobs, Uobs = reader.read_observed_maps(args.nside)
Iobs             = hp.smoothing(Iobs, fwhm=np.radians(fwhm))
Qobs             = hp.smoothing(Qobs, fwhm=np.radians(fwhm))
Uobs             = hp.smoothing(Uobs, fwhm=np.radians(fwhm))

# --- Planck variances ---
varQQobs, varQUobs, varUUobs = readPlanckMaps.read_variance_from_systematics(
    FILENAME_FFP_SIMS,
    nside_target=args.nside
)

print(f"✓ Data loaded in {time.perf_counter() - t0:.2f} s")

# -------------------------------------
# MATRICES BUILDER
# -------------------------------------

t0 = time.perf_counter()
print("\n[2/4] Building matrices...")

Ibuilder = intensityTemplate.IntensityTemplateBuilder(
    labels=labels,
    intensity=intensity
)

H = Ibuilder.build(Iobs)
H = scipy.sparse.csr_matrix(H)

det = varQQobs * varUUobs - varQUobs**2

wQQFull =  varUUobs / det
wUUFull =  varQQobs / det
wQUFull = -varQUobs / det

w = scipy.sparse.bmat([
    [scipy.sparse.diags(wQQFull), scipy.sparse.diags(wQUFull)],
    [scipy.sparse.diags(wQUFull), scipy.sparse.diags(wUUFull)]
], format="csr")

wsqrt = scipy.sparse.bmat([
    [scipy.sparse.diags(wQQFull**0.5), scipy.sparse.diags(wQUFull*0)],
    [scipy.sparse.diags(wQUFull*0), scipy.sparse.diags(wUUFull**0.5)]
], format="csr")

H = scipy.sparse.bmat([
    [H, None],
    [None, H]
])

d = np.concatenate((Qobs, Uobs))

print(f"✓ Matrices built in {time.perf_counter() - t0:.2f} s")
print(f"  H shape : {H.shape}")
print(f"  W shape : {w.shape}")

# -------------------------------------
# GIBBS SAMPLING
# -------------------------------------

t0 = time.perf_counter()
print("\n[3/4] Running Gibbs sampler...")

gibbs = GibbsSampler(
    H,
    w,
    d,
    wsqrt,
    maxiterCG=10,
    sigmaPrior=0.1
)

results = gibbs.run(niter=500)

elapsed_sampling = time.perf_counter() - t0

print(f"✓ Gibbs sampling completed in {elapsed_sampling:.2f} s")
print(f"  Average time per iteration: {elapsed_sampling / 500:.3f} s")

# -------------------------------------
# SAVE RESULTS
# -------------------------------------

t0 = time.perf_counter()
print("\n[4/4] Saving results...")

out_path = FOLDERSAVE + f"reconstructionGibbsNside{args.nside}_chain{chainID}.pkl"

with open(out_path, "wb") as fh:
    pickle.dump(results, fh, protocol=pickle.HIGHEST_PROTOCOL)

print(f"✓ Saved in {time.perf_counter() - t0:.2f} s")
print(f"  File: {out_path}")

# -------------------------------------
# FINAL SUMMARY
# -------------------------------------

print("\n" + "=" * 60)
print("JOB COMPLETED")
print("=" * 60)
print(f"Total runtime: {time.perf_counter() - t0_total:.2f} s")
print("=" * 60)