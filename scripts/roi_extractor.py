"""
ROI Extractor — Maps Schaefer atlas parcels to cognitive dimensions.

Uses the Yeo 7-network parcellation to group brain parcels into
functionally meaningful networks, then aggregates into 5 cognitive
dimensions for the neural scoring function.

The 5 dimensions:
  C: Comprehension depth     (Default A+B networks)
  M: Memory encoding         (Limbic A+B networks)
  A: Sustained attention     (Frontoparietal A + Dorsal Attention A)
  X: Confusion signal        (Ventral Attention A + Salience)
  D: DMN suppression         (negated Default C)
"""

import os
import json
import numpy as np
from dataclasses import dataclass, field


# Yeo 7-network labels as they appear in Schaefer atlas parcel names
# Each Schaefer parcel name encodes its network, e.g.:
#   "7Networks_LH_Default_PFC_1" → network = "Default"
#   "7Networks_RH_DorsAttn_Post_2" → network = "DorsAttn"
NETWORK_KEYWORDS = {
    "Default":    "default",
    "Cont":       "frontoparietal",   # Control/Frontoparietal network
    "DorsAttn":   "dorsal_attention",
    "SalVentAttn": "ventral_attention",  # Salience/Ventral Attention
    "Limbic":     "limbic",
    "SomMot":     "somatomotor",
    "Vis":        "visual",
}

# Sub-network labels (some Schaefer parcels have A/B/C suffixes)
SUBNETWORK_MAP = {
    # Default network sub-divisions
    "Default_PFC":     "default_C",     # Medial PFC → DMN core
    "Default_Temp":    "default_A",     # Lateral temporal → comprehension
    "Default_pCunPCC": "default_C",     # Precuneus/PCC → DMN core
    "Default_Par":     "default_B",     # Parietal → semantic integration
    # These are approximations — the actual sub-network labels vary.
    # Claude Code can refine these after inspecting real Schaefer labels.
}


@dataclass
class CognitiveDimensions:
    """Extracted cognitive dimensions from a brain activation map."""
    comprehension: float = 0.0      # C: semantic depth
    memory_encoding: float = 0.0    # M: hippocampal engagement
    sustained_attention: float = 0.0 # A: executive control
    confusion: float = 0.0          # X: error/conflict detection
    dmn_suppression: float = 0.0    # D: mind-wandering suppression

    def to_array(self) -> np.ndarray:
        return np.array([
            self.comprehension,
            self.memory_encoding,
            self.sustained_attention,
            self.confusion,
            self.dmn_suppression,
        ])

    def to_dict(self) -> dict:
        return {
            "comprehension": self.comprehension,
            "memory_encoding": self.memory_encoding,
            "sustained_attention": self.sustained_attention,
            "confusion": self.confusion,
            "dmn_suppression": self.dmn_suppression,
        }

    @property
    def labels(self) -> list[str]:
        return [
            "Comprehension",
            "Memory encoding",
            "Sustained attention",
            "Confusion",
            "DMN suppression",
        ]


class ROIExtractor:
    """Extracts cognitive dimensions from Schaefer atlas brain activations.

    Handles both parcel-level (400/1000) and vertex-level (20484) input.
    For vertex-level input, parcellates using Schaefer surface atlas first.
    """

    # fsaverage5 has 10242 vertices per hemisphere = 20484 total
    FSAVG5_VERTICES = 20484

    def __init__(self, n_rois: int = 1000, yeo_networks: int = 7):
        """
        Args:
            n_rois: Number of input values. If 20484 (vertex-level),
                    will parcellate to 1000 Schaefer parcels first.
                    If 400/1000, uses directly as parcels.
            yeo_networks: Number of Yeo networks (7 or 17)
        """
        self.input_size = n_rois
        self.yeo_networks = yeo_networks
        self.network_masks = {}
        self._vertex_to_parcel = None

        # If vertex-level, set up parcellation to 1000 parcels
        if n_rois == self.FSAVG5_VERTICES:
            self.n_rois = 1000
            self._build_vertex_parcellation()
        else:
            self.n_rois = n_rois

        self._build_masks()

    def _build_vertex_parcellation(self):
        """Build mapping from fsaverage5 vertices to Schaefer 1000 parcels."""
        try:
            from nilearn import datasets, surface
            import nibabel as nib

            atlas = datasets.fetch_atlas_schaefer_2018(
                n_rois=1000, yeo_networks=self.yeo_networks, resolution_mm=1
            )
            atlas_img = nib.load(atlas["maps"])

            # Project volume atlas to fsaverage5 surface
            fsavg = datasets.fetch_surf_fsaverage(mesh="fsaverage5")
            lh_labels = surface.vol_to_surf(
                atlas_img, fsavg["pial_left"], interpolation="nearest_most_frequent"
            ).astype(int)
            rh_labels = surface.vol_to_surf(
                atlas_img, fsavg["pial_right"], interpolation="nearest_most_frequent"
            ).astype(int)
            self._vertex_to_parcel = np.concatenate([lh_labels, rh_labels])
            n_assigned = (self._vertex_to_parcel > 0).sum()
            print(f"[ROIExtractor] Vertex parcellation: {n_assigned}/{self.FSAVG5_VERTICES} "
                  f"vertices assigned to {len(np.unique(self._vertex_to_parcel)) - 1} parcels")
        except Exception as e:
            print(f"[ROIExtractor] Surface parcellation failed ({e}), using chunked fallback")
            # Fallback: evenly divide vertices into 1000 chunks
            labels = np.zeros(self.FSAVG5_VERTICES, dtype=int)
            chunk_size = self.FSAVG5_VERTICES / 1000
            for i in range(self.FSAVG5_VERTICES):
                labels[i] = min(int(i / chunk_size) + 1, 1000)
            self._vertex_to_parcel = labels

    def _parcellate(self, vertex_activation: np.ndarray) -> np.ndarray:
        """Average vertex-level activation into parcel-level."""
        parcel_values = np.zeros(self.n_rois)
        for p in range(1, self.n_rois + 1):
            mask = self._vertex_to_parcel == p
            if mask.sum() > 0:
                parcel_values[p - 1] = vertex_activation[mask].mean()
        return parcel_values

    def _build_masks(self):
        """Build boolean masks mapping parcels to networks.

        Two strategies:
        1. If nilearn is available, use actual Schaefer atlas labels
        2. Fallback: approximate assignment based on parcel index ranges
        """
        try:
            self._build_masks_from_atlas()
        except Exception as e:
            print(f"[ROIExtractor] Atlas loading failed ({e}), using index-based fallback")
            self._build_masks_fallback()

    def _build_masks_from_atlas(self):
        """Build masks from actual Schaefer atlas labels."""
        from nilearn import datasets

        atlas = datasets.fetch_atlas_schaefer_2018(
            n_rois=self.n_rois,
            yeo_networks=self.yeo_networks,
        )

        raw_labels = [
            label.decode("utf-8") if isinstance(label, bytes) else str(label)
            for label in atlas["labels"]
        ]

        # Skip "Background" label if present (index 0)
        if raw_labels and "background" in raw_labels[0].lower():
            labels = raw_labels[1:]
        else:
            labels = raw_labels

        # Initialize masks
        networks = [
            "default_A", "default_B", "default_C",
            "frontoparietal", "dorsal_attention",
            "ventral_attention", "limbic",
            "somatomotor", "visual",
        ]
        self.network_masks = {net: np.zeros(self.n_rois, dtype=bool) for net in networks}

        # Parse each label to assign to network
        for i, label in enumerate(labels):
            label_upper = label.upper()

            if "DEFAULT" in label_upper:
                # Try sub-network assignment
                if "TEMP" in label_upper or "PAR" in label_upper:
                    # Temporal and parietal default → comprehension
                    self.network_masks["default_A"][i] = True
                    self.network_masks["default_B"][i] = True
                elif "PFC" in label_upper or "PCUNPCC" in label_upper:
                    # Medial PFC and precuneus → DMN core
                    self.network_masks["default_C"][i] = True
                else:
                    # Unspecified default → assign to A
                    self.network_masks["default_A"][i] = True

            elif "CONT" in label_upper:
                self.network_masks["frontoparietal"][i] = True

            elif "DORSATTN" in label_upper:
                self.network_masks["dorsal_attention"][i] = True

            elif "SALVENTATTN" in label_upper:
                self.network_masks["ventral_attention"][i] = True

            elif "LIMBIC" in label_upper:
                self.network_masks["limbic"][i] = True

            elif "SOMMOT" in label_upper:
                self.network_masks["somatomotor"][i] = True

            elif "VIS" in label_upper:
                self.network_masks["visual"][i] = True

        # Report
        total = sum(mask.sum() for mask in self.network_masks.values())
        print(f"[ROIExtractor] Assigned {total}/{self.n_rois} parcels to networks:")
        for name, mask in self.network_masks.items():
            if mask.sum() > 0:
                print(f"  {name}: {mask.sum()} parcels")

        self.labels = labels

    def _build_masks_fallback(self):
        """Approximate network masks based on index ranges.

        The Schaefer atlas is ordered: LH networks first, then RH.
        Within each hemisphere, networks are ordered roughly as:
        Visual → SomMot → DorsAttn → SalVentAttn → Limbic → Cont → Default

        This is a rough approximation. Let Claude Code refine it after
        inspecting the actual parcel labels.
        """
        n = self.n_rois
        half = n // 2  # LH vs RH split

        # Approximate proportions (from Schaefer 2018 paper)
        # These fractions are rough and should be validated
        fractions = {
            "visual":            0.15,
            "somatomotor":       0.15,
            "dorsal_attention":  0.12,
            "ventral_attention": 0.10,
            "limbic":            0.08,
            "frontoparietal":    0.18,
            "default_A":         0.08,
            "default_B":         0.07,
            "default_C":         0.07,
        }

        self.network_masks = {}
        idx = 0
        for name, frac in fractions.items():
            count = int(half * frac)
            mask = np.zeros(n, dtype=bool)
            # LH
            mask[idx:idx + count] = True
            # RH (mirror)
            mask[half + idx:half + idx + count] = True
            self.network_masks[name] = mask
            idx += count

        print(f"[ROIExtractor] Using fallback index-based masks (approximate)")
        self.labels = [f"parcel_{i}" for i in range(n)]

    def extract(self, activation: np.ndarray) -> CognitiveDimensions:
        """Extract cognitive dimensions from a brain activation map.

        Args:
            activation: np.ndarray of shape (n_vertices,) or (n_parcels,)

        Returns:
            CognitiveDimensions dataclass
        """
        # Parcellate vertex-level data if needed
        if len(activation) == self.FSAVG5_VERTICES and self._vertex_to_parcel is not None:
            activation = self._parcellate(activation)

        assert len(activation) == self.n_rois, (
            f"Expected {self.n_rois} parcels, got {len(activation)}"
        )

        def safe_mean(mask):
            if mask.sum() == 0:
                return 0.0
            return float(activation[mask].mean())

        # Comprehension: Default A + B (semantic integration areas)
        comprehension = np.mean([
            safe_mean(self.network_masks["default_A"]),
            safe_mean(self.network_masks["default_B"]),
        ])

        # Memory encoding: Limbic (hippocampal/parahippocampal)
        memory = safe_mean(self.network_masks["limbic"])

        # Sustained attention: Frontoparietal + Dorsal Attention
        attention = np.mean([
            safe_mean(self.network_masks["frontoparietal"]),
            safe_mean(self.network_masks["dorsal_attention"]),
        ])

        # Confusion: Ventral Attention / Salience network
        confusion = safe_mean(self.network_masks["ventral_attention"])

        # DMN suppression: negated Default C (lower = more engaged)
        dmn_suppression = -safe_mean(self.network_masks["default_C"])

        return CognitiveDimensions(
            comprehension=comprehension,
            memory_encoding=memory,
            sustained_attention=attention,
            confusion=confusion,
            dmn_suppression=dmn_suppression,
        )

    def extract_batch(self, activations: np.ndarray) -> list[CognitiveDimensions]:
        """Extract dimensions for multiple activation maps.

        Args:
            activations: np.ndarray of shape (n_samples, n_parcels)

        Returns:
            List of CognitiveDimensions.
        """
        return [self.extract(act) for act in activations]

    def get_network_activations(self, activation: np.ndarray) -> dict:
        """Get raw mean activation per network (for debugging/visualization)."""
        return {
            name: float(activation[mask].mean()) if mask.sum() > 0 else 0.0
            for name, mask in self.network_masks.items()
        }


if __name__ == "__main__":
    # Quick test with random data
    print("Testing ROI Extractor with random data...")
    extractor = ROIExtractor(n_rois=1000)

    fake_activation = np.random.randn(1000)
    dims = extractor.extract(fake_activation)

    print(f"\nExtracted dimensions (random input):")
    for label, value in zip(dims.labels, dims.to_array()):
        print(f"  {label}: {value:.4f}")

    print(f"\nAll values near 0 (expected for random input): ",
          all(abs(v) < 1.0 for v in dims.to_array()))
