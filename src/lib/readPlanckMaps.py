from pathlib import Path
import healpy as hp

from ..data.getData import dataFolder


# Default path to the Planck 353 GHz full-mission sky map
DEFAULT_PLANCK_FILENAME = Path(dataFolder) / "HFI_SkyMap_353_2048_R3.01_full.fits"

# Conversion factors
K_TO_MICRO_K = 1e6       # K_cmb → μK_cmb
VAR_TO_MICRO_K_SQ = 1e12  # K² → μK²


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