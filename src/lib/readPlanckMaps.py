from pathlib import Path
import healpy as hp
from astropy.io import fits
import numpy as np

from ..data.getData import dataFolder


# Default path to the Planck 353 GHz full-mission sky map
DEFAULT_PLANCK_FILENAME = Path(dataFolder) / "HFI_SkyMap_353_2048_R3.01_full.fits"

# Conversion factors
K_TO_MICRO_K = 1e6       # K_cmb → μK_cmb
VAR_TO_MICRO_K_SQ = 1e12  # K² → μK²

def read_variance_from_systematics(filename: Path | str, nside_target: int) -> tuple:
    """Estimate per-pixel QQ, QU, UU variance maps from systematic noise realizations.

    Loads an ensemble of noise realizations, downgradles them to the target
    resolution, then estimates the Q/U covariance matrix across realizations.
    The diagonal blocks of this covariance give per-pixel variance maps.

    Args:
        filename:     Path to the FITS file containing noise realizations
                      in extension 1, shaped (n_reals, n_pix, 3) for (I, Q, U).
        nside_target: Target HEALPix Nside resolution parameter.

    Returns:
        Tuple of (var_QQ, var_QU, var_UU) per-pixel variance maps in μK²,
        each of shape (n_pix_target,).
    """
    n_pix_target = hp.nside2npix(nside_target)

    # Load noise realizations from the first FITS extension
    with fits.open(filename) as hdul:
        noise_reals = hdul[1].data.copy()

    n_reals = noise_reals.shape[0]

    # Downgrade each realization to the target resolution (K → μK)
    noise_down = np.zeros((n_reals, n_pix_target, 3))
    for i in range(n_reals):
        # Transpose: HEALPix expects (3, n_pix), data is stored as (n_pix, 3)
        noise_down[i] = hp.ud_grade(noise_reals[i].T, nside_target).T * K_TO_MICRO_K

    # Estimate the joint Q/U covariance across realizations
    # cov_qu is (2 * n_pix_target, 2 * n_pix_target), laid out as [Q | U]
    cov_qu = np.cov(noise_down[..., 1], noise_down[..., 2], rowvar=False)

    # Extract diagonal blocks: per-pixel QQ, QU, UU variances
    var_qq = np.diag(cov_qu[:n_pix_target, :n_pix_target])
    var_qu = np.diag(cov_qu[n_pix_target:, :n_pix_target])
    var_uu = np.diag(cov_qu[n_pix_target:, n_pix_target:])

    return var_qq, var_qu, var_uu
    
class PlanckReader:
    """Reader for Planck HFI sky maps in FITS format.

    Handles loading and downgrading of intensity/polarization maps
    and their associated variance maps to a target HEALPix resolution.
    """

    def __init__(self, filename: Path | str = DEFAULT_PLANCK_FILENAME):
        """
        Args:
            filename: Path to the Planck FITS file.
                      Defaults to the standard 353 GHz full-mission map.
        """
        self.filename = Path(filename)

    def read_observed_maps(self, nside_target: int) -> tuple:
        """Read and downgrade the I, Q, U sky maps to the target resolution.

        Stokes parameters are converted from K_cmb to μK_cmb.

        Args:
            nside_target: Target HEALPix Nside resolution parameter.

        Returns:
            Tuple of (I, Q, U) downgraded maps in μK_cmb.
        """
        i_map, q_map, u_map = hp.read_map(self.filename, field=(0, 1, 2))

        i_map = hp.ud_grade(i_map, nside_target) * K_TO_MICRO_K
        q_map = hp.ud_grade(q_map, nside_target) * K_TO_MICRO_K
        u_map = hp.ud_grade(u_map, nside_target) * K_TO_MICRO_K

        return i_map, q_map, u_map

    def read_variance_maps(self, nside_target: int) -> tuple:
        """Read and downgrade the QQ, QU, UU variance maps to the target resolution.

        Variance maps are converted from K² to μK² and downgraded
        with power=2 to correctly propagate variance under resolution change.

        Args:
            nside_target: Target HEALPix Nside resolution parameter.

        Returns:
            Tuple of (var_QQ, var_QU, var_UU) downgraded variance maps in μK².
        """
        var_qq, var_qu, var_uu = hp.read_map(self.filename, field=(7, 8, 9))

        var_qq = hp.ud_grade(var_qq, nside_target, power=2) * VAR_TO_MICRO_K_SQ
        var_qu = hp.ud_grade(var_qu, nside_target, power=2) * VAR_TO_MICRO_K_SQ
        var_uu = hp.ud_grade(var_uu, nside_target, power=2) * VAR_TO_MICRO_K_SQ

        return var_qq, var_qu, var_uu