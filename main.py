import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import scipy
import pickle
import os
import time
import LocalBubble

from src.lib import readPlanckMaps
from src.lib import intensityTemplate
from src.data.getData import dataFolder
from src.lib.sampler import GibbsSampler
from src.utils import parser

# -------------------------------------
# PARAMETERS
# -------------------------------------
dataFolder = "/work/scratch/data/regniem/clouds"
args       = parser.parse_args()
FILENAME_FFP_SIMS = "/work/scratch/data/regniem/noise/nside64/100realizations_noise_ffp10_353_GHz.fits"
FOLDERSAVE = "/work/scratch/data/regniem/clouds"
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

maskGalPlane = LocalBubble.get_mask_galactic_plane(args.nside, field=3)#1)
print(f"\nFsky : {maskGalPlane.sum()/maskGalPlane.size:.3f}\n")
t0 = time.perf_counter()
print("\n[1/4] Reading data...")

# --- Read 3D data ---
intensity = hp.read_map(str(dataFolder) + f"/Intensity_Nside{args.nside}.fits", field=None)
#labels = hp.read_map(str(dataFolder) + f"/Labels_Nside{args.nside}_alpha0.5000.fits", field=None)
# Regularisation parameters used to produce each label map
alphas = [0.01, 0.03, 0.05, 0.1, 0.5]
labels = np.array([
    hp.read_map(dataFolder + f"/Labels_Nside{args.nside}_alpha{alpha:.4f}.fits", field=None)
    for alpha in alphas
])

# Merge multi-scale labels into a single consistent labelling
labels = intensityTemplate.assembleLabelsClouds(labels)
fwhm = 2*np.degrees(hp.nside2resol(args.nside))

# --- Planck maps ---
reader           = readPlanckMaps.PlanckReader(filename="/work/scratch/data/regniem/dust/HFI_SkyMap_353_2048_R3.01_full.fits")
Iobs, Qobs, Uobs = reader.read_observed_maps(args.nside)
Iobs             = hp.smoothing(Iobs, fwhm=np.radians(fwhm))#[maskGalPlane]
Qobs             = hp.smoothing(Qobs, fwhm=np.radians(fwhm))#[maskGalPlane]
Uobs             = hp.smoothing(Uobs, fwhm=np.radians(fwhm))#[maskGalPlane]
qobs = Qobs / Iobs
uobs = Uobs / Iobs
Qobs = Qobs[maskGalPlane]
Uobs = Uobs[maskGalPlane]
# --- Planck variances ---
varQQobs, varQUobs, varUUobs = readPlanckMaps.read_variance_from_systematics(
    FILENAME_FFP_SIMS,
    nside_target=args.nside
)
varQQobs = varQQobs[maskGalPlane]
varQUobs = varQUobs[maskGalPlane]
varUUobs = varUUobs[maskGalPlane]

print(f"✓ Data loaded in {time.perf_counter() - t0:.2f} s")

# -------------------------------------
# MATRICES BUILDER
# -------------------------------------

t0 = time.perf_counter()
print("\n[2/4] Building matrices...")

Ibuilder = intensityTemplate.IntensityTemplateBuilder(
    labels=labels,
    intensity=intensity,
    mask=maskGalPlane,
)

H = Ibuilder.build(Iobs)

H = scipy.sparse.csr_matrix(H)
uniqueLabels = np.unique(labels)[:-1]
nclouds = len(uniqueLabels)

Aq0 = np.zeros(nclouds)
Au0 = np.zeros(nclouds)

for k, icloud in enumerate(uniqueLabels):
    mask = np.where(labels == icloud)[1]
    Aq0[k] = qobs[mask].mean()
    Au0[k] = uobs[mask].mean()
Aq0 = Aq0[Ibuilder.valid_clouds]
Au0 = Au0[Ibuilder.valid_clouds]
muPrior = np.concatenate((Aq0, Au0))
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
    maxiterCG=20,
    sigmaPrior=0.1,
    muPrior=muPrior
)

niter = 2000
results = gibbs.run(niter=niter)
QSample = np.ones((niter, maskGalPlane.size)) * np.nan
QSample[:, maskGalPlane] = results["QSample"]
USample = np.ones((niter, maskGalPlane.size)) * np.nan
USample[:, maskGalPlane] = results["USample"]

results["QSample"] = QSample
results["USample"] = USample

stokesQobs = np.ones(maskGalPlane.size) * np.nan
stokesUobs = np.ones(maskGalPlane.size) * np.nan
stokesQvar = np.ones(maskGalPlane.size) * np.nan
stokesUvar = np.ones(maskGalPlane.size) * np.nan

stokesQobs[maskGalPlane] = Qobs
stokesUobs[maskGalPlane] = Uobs
stokesQvar[maskGalPlane] = varQQobs
stokesUvar[maskGalPlane] = varUUobs

results["stokesQobs"]   = stokesQobs
results["stokesUobs"]   = stokesUobs
results["stokesQvar"]   = stokesQvar
results["stokesUvar"]   = stokesUvar
results["maskGalPlane"] = maskGalPlane
elapsed_sampling = time.perf_counter() - t0

print(f"✓ Gibbs sampling completed in {elapsed_sampling:.2f} s")
print(f"  Average time per iteration: {elapsed_sampling / 500:.3f} s")

# -------------------------------------
# SAVE RESULTS
# -------------------------------------

t0 = time.perf_counter()
print("\n[4/4] Saving results...")

out_path = FOLDERSAVE + f"/reconstructionGibbsNside{args.nside}_chain{chainID}.pkl"

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