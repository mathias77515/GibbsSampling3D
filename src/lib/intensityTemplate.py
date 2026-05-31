"""
intensityTemplate.py
"""

from pathlib import Path
import numpy as np


def assembleLabelsClouds(labels: np.ndarray) -> np.ndarray:
    
    jointLabels = labels[0].copy()

    k = np.unique(labels[0])[:-1].size + 1
    for label in labels[1:]:
        uniqueClouds = np.unique(label)[:-1]
        for cloud_label in uniqueClouds:
            maski = label == cloud_label
            jointLabels[maski] = k
            k += 1
    return jointLabels


class IntensityTemplateBuilder:
    """Constructs and normalises the intensity template matrix H.

    Args:
        labels:    Label array, shape (n_layers, n_pix).
        intensity: Layer intensity arrays, shape (n_layers, n_pix).
        mask:      Boolean array of shape (n_pix,). True = keep pixel.
                   If None, all pixels are used.
        min_pix:   Minimum number of unmasked pixels for a cloud to survive.
    """

    def __init__(
        self,
        labels: np.ndarray,
        intensity: np.ndarray,
        mask: np.ndarray = None,
        min_pix: int = 5,
    ):
        self.labels    = labels
        self.intensity = intensity
        self.n_layers, self.n_pix = intensity.shape
        self.min_pix   = min_pix

        # --- Build pixel mask ---
        if mask is None:
            self.mask = np.ones(self.n_pix, dtype=bool)
        else:
            assert mask.shape == (self.n_pix,), "mask must have shape (n_pix,)"
            self.mask = mask

        self.n_pix_masked = self.mask.sum()

        # --- Cloud labels on full sky ---
        self.unique_labels_full = np.unique(self.labels)[:-1]

        # --- Find valid clouds (enough pixels outside mask) ---
        self.valid_clouds, self.old_to_new, self.unique_labels = self._prune_clouds()
        self.n_clouds = len(self.unique_labels)

        print(f"Pixels  : {self.n_pix} total, {self.n_pix_masked} unmasked")
        print(f"Clouds  : {len(self.unique_labels_full)} total, {self.n_clouds} after masking")
        print(f"Removed : {(~self.valid_clouds).sum()} clouds")

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def build(self, Iobs: np.ndarray) -> np.ndarray:
        """Build the normalised template matrix on unmasked pixels.

        Args:
            Iobs: Observed intensity, shape (n_pix,) — full sky.

        Returns:
            H: shape (n_pix_masked, n_clouds_valid)
        """
        h_matrix = self._accumulate()                      # (n_pix_masked, n_clouds_valid)
        h_matrix = self._normalise(h_matrix, Iobs[self.mask])
        return h_matrix

    def restore_fullsky(self, x: np.ndarray, fill_value: float = 0.0) -> np.ndarray:
        """Project a masked-pixel map back to full sky.

        Args:
            x:          Array of shape (n_pix_masked,) or (n, n_pix_masked).
            fill_value: Value for masked pixels.

        Returns:
            Array of shape (n_pix,) or (n, n_pix).
        """
        if x.ndim == 1:
            out = np.full(self.n_pix, fill_value)
            out[self.mask] = x
        else:
            out = np.full((x.shape[0], self.n_pix), fill_value)
            out[:, self.mask] = x
        return out

    def restore_fullsky_clouds(self, x: np.ndarray, fill_value: float = np.nan) -> np.ndarray:
        """Map valid-cloud amplitudes back to full-cloud index space.

        Args:
            x:          Shape (n_clouds_valid,) or (n_iter, n_clouds_valid).
            fill_value: Value for removed clouds.

        Returns:
            Shape (n_clouds_full,) or (n_iter, n_clouds_full).
        """
        n_full = len(self.unique_labels_full)
        if x.ndim == 1:
            out = np.full(n_full, fill_value)
            out[self.valid_clouds] = x
        else:
            out = np.full((x.shape[0], n_full), fill_value)
            out[:, self.valid_clouds] = x
        return out

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _prune_clouds(self):
        """Identify clouds with enough unmasked pixels.

        Returns:
            valid_clouds: Boolean array, shape (n_clouds_full,).
            old_to_new:   Int array mapping old cloud index → new index (-1 if removed).
            unique_labels: Labels of surviving clouds.
        """
        n_clouds_full = len(self.unique_labels_full)
        pixel_counts  = np.zeros(n_clouds_full, dtype=int)

        for cloud_idx, label in enumerate(self.unique_labels_full):
            # Count unmasked pixels belonging to this cloud
            in_cloud            = (self.labels == label)          # (n_pix,)
            pixel_counts[cloud_idx] = (in_cloud & self.mask).sum()

        valid_clouds  = pixel_counts >= self.min_pix
        old_to_new    = np.full(n_clouds_full, -1, dtype=int)
        old_to_new[valid_clouds] = np.arange(valid_clouds.sum())

        return valid_clouds, old_to_new, self.unique_labels_full[valid_clouds]

    def _accumulate(self) -> np.ndarray:
        """Accumulate intensity into (n_pix_masked, n_clouds_valid)."""

        h_matrix = np.zeros((self.n_pix_masked, self.n_clouds))

        for cloud_idx, label in enumerate(self.unique_labels):
            for layer in range(self.n_layers):
                pixel_mask = (np.asarray(self.labels[layer], dtype=int) == label) & self.mask
                local_idx  = np.where(self.mask)[0]                        # masked pixel positions
                h_matrix[np.isin(local_idx, np.where(pixel_mask)[0]), cloud_idx] += \
                    self.intensity[layer, pixel_mask]

        return h_matrix

    def _normalise(self, h_matrix: np.ndarray, Iobs_masked: np.ndarray) -> np.ndarray:
        """Normalise on unmasked pixels only."""
        ratio    = np.nansum(h_matrix.T / Iobs_masked, axis=0)
        h_matrix = (h_matrix.T / np.where(ratio > 0, ratio, 1.0)).T
        h_matrix = np.nan_to_num(h_matrix, nan=0.0)
        return h_matrix