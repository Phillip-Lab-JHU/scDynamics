'''
Alpha-SMA / DAPI mask-only morphometrics
'''
from __future__ import annotations

import os
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from skimage import io as skio
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects, skeletonize, convex_hull_image
from skimage.segmentation import find_boundaries, relabel_sequential
from scipy.ndimage import distance_transform_edt, binary_erosion, convolve
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree, dijkstra

RAD2DEG = 180.0 / math.pi


@dataclass
class AssignParams:
    overlap_min_frac: float = 0.1
    keep_centroid_inside_only: bool = False


class Actin_nucleus_feature_extractor:
    def __init__(self,
                 pixel_size_um: float = 1.0,
                 min_cell_area_px: int = 20,
                 min_nuc_area_px: int = 10,
                 assign_params: Optional[AssignParams] = None,
                 rim_threshold_um: float = 2.0,
                 obb_angle_samples: int = 45,
                 chord_bins: int = 25,
                 voronoi_max_pixels: int = 200000,
                 save_overlays: bool = False,
                 overlay_dir: Optional[str] = None,
                 overlay_format: str = "png"):
        self.pixel_size_um = float(pixel_size_um)
        self.min_cell_area_px = int(min_cell_area_px)
        self.min_nuc_area_px = int(min_nuc_area_px)
        self.assign_params = assign_params or AssignParams()
        self.rim_threshold_um = float(rim_threshold_um)
        self.obb_angle_samples = int(obb_angle_samples)
        self.chord_bins = int(chord_bins)
        self.voronoi_max_pixels = int(voronoi_max_pixels)
        self.save_overlays = bool(save_overlays)
        self.overlay_dir = overlay_dir  # default decided in extract_from_paths
        self.overlay_format = overlay_format.lower()

    # ---------------- I/O helpers ----------------
    @staticmethod
    def _labels_from_colored_mask(
            img,
            bg_colors=((0, 0, 0), (0, 0, 0, 0)),  # treat black or transparent as background
            connectivity=1,  # 1=4-connected (diagonals NOT connected); 2=8-connected
            tol=0  # RGB tolerance (in case of slight palette noise)
    ):
        """
        Convert an RGB/RGBA color-encoded instance mask into a labeled mask.
        Each unique color (excluding bg_colors) is split into connected components,
        so repeated colors yield distinct labels if they are not touching.
        """
        if img.ndim != 3 or img.shape[2] not in (3, 4):
            raise ValueError("Expected an RGB/RGBA image for color mask.")

        img = img.astype(np.int64)
        H, W, C = img.shape
        has_alpha = (C == 4)

        # Background boolean mask from alpha==0 (if present) or listed bg colors
        bg = np.zeros((H, W), dtype=bool)
        if has_alpha:
            bg |= (img[..., 3] == 0)

        def close_to(color, tol=0):
            color = np.array(color, dtype=np.int64)
            comps = []
            for k in range(min(C, len(color))):
                comps.append(np.abs(img[..., k] - color[k]) <= tol)
            return np.logical_and.reduce(comps)

        for c in bg_colors:
            if (has_alpha and len(c) == 4) or (not has_alpha and len(c) in (3, 4)):
                bg |= close_to(c, tol=tol)

        # Build a color ID map (ignore background pixels)
        # Use a packed integer to represent RGB(A) colors
        base = np.array([1, 256, 256 * 256, 256 * 256 * 256][:C], dtype=np.int64)
        packed = (img * base).sum(axis=2)
        packed[bg] = 0  # force background to 0

        labels = np.zeros((H, W), dtype=np.int32)
        cur = 1

        # Unique nonzero colors
        colors = np.unique(packed)
        colors = colors[colors != 0]

        for color in colors:
            mask = (packed == color)
            if not mask.any():
                continue
            # Split into instances for THIS color only
            comp = label(mask, connectivity=connectivity)
            if comp.max() == 0:
                continue
            # Assign unique ids into the global label image
            m = comp > 0
            labels[m] = comp[m] + cur - 1
            cur += comp.max()

        return labels

    @staticmethod
    def _ensure_binary_or_label(img: np.ndarray) -> np.ndarray:
        """
        Accepts:
          - 2D binary: labels it
          - 2D labeled: returns as-is
          - 3/4-channel color instance mask: converts to labeled using connected components per color
        """
        # Color instance mask -> labeled instances
        if img.ndim == 3 and img.shape[2] in (3, 4):
            return Actin_nucleus_feature_extractor._labels_from_colored_mask(img, bg_colors=((0, 0, 0), (0, 0, 0, 0)), connectivity=1, tol=0)

        # 2D case
        img2d = img
        uniq = np.unique(img2d)
        # Already labeled (many values)?
        if (img2d.dtype.kind in "ui" and uniq.size > 2):
            return img2d.astype(np.int32)
        # Binary -> label
        return label(img2d > 0).astype(np.int32)

    def _is_binary(arr: np.ndarray) -> bool:
        uniq = np.unique(arr)
        return arr.dtype == bool or (uniq.size <= 2 and set(uniq).issubset({0, 1}))

    def _prepare_labels(self, arr: np.ndarray, min_size: int, connectivity: int = 1) -> np.ndarray:
        """
        Returns an int32 labeled image with small objects removed,
        without merging distinct instances.
        """
        if Actin_nucleus_feature_extractor._is_binary(arr):
            # Binary → remove small → label
            binmask = remove_small_objects(arr.astype(bool), min_size=min_size, connectivity=connectivity)
            lab = label(binmask, connectivity=connectivity).astype(np.int32)

            Actin_nucleus_feature_extractor._remove_edge_touching_cells(lab)
            lab, _, _ = relabel_sequential(lab)  # Make labels compact (1..N)
            return lab.astype(np.int32)
        else:
            # Already labeled → remove small by label (sets small labels to 0), no merging
            lab = remove_small_objects(arr.astype(np.int32), min_size=min_size, connectivity=connectivity)
            lab = lab.astype(np.int32)
            Actin_nucleus_feature_extractor._remove_edge_touching_cells(lab)
            lab, _, _ = relabel_sequential(lab)  # Compact labels to 1..N (preserves boundaries)
            return lab.astype(np.int32)

    # ---------- Edge-touching filter & overlay ----------

    @staticmethod
    def _remove_edge_touching_cells(lab: np.ndarray) -> np.ndarray:
        H, W = lab.shape
        keep_labels = []
        for r in regionprops(lab):
            minr, minc, maxr, maxc = r.bbox
            touches = (minr == 0) or (minc == 0) or (maxr == H) or (maxc == W)
            if not touches:
                keep_labels.append(r.label)
        if len(keep_labels) == 0:
            return np.zeros_like(lab, dtype=np.int32)
        kept = np.isin(lab, keep_labels)
        return label(kept)

    @staticmethod
    def _save_overlay(overlay_dir: str,
                      label_id: int,
                      cell_crop: np.ndarray,
                      nuclei_union_crop: np.ndarray,
                      fmt: str = "png") -> None:
        os.makedirs(overlay_dir, exist_ok=True)
        h, w = cell_crop.shape
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        # actin in red
        rgb[..., 0] = np.where(cell_crop, 255, 0).astype(np.uint8)
        # nuclei in cyan (G+B)
        nu = np.where(nuclei_union_crop, 255, 0).astype(np.uint8)
        rgb[..., 1] = nu
        rgb[..., 2] = nu
        fname = os.path.join(overlay_dir, f"{label_id}.{fmt}")
        skio.imsave(fname, rgb)

    @staticmethod
    def load_mask(path: str) -> np.ndarray:
        img = skio.imread(path)
        return Actin_nucleus_feature_extractor._ensure_binary_or_label(img)

    # ------------- Basic geometry & features -------------

    def _regionprops_basic(self, lab: np.ndarray) -> Dict[int, dict]:
        props: Dict[int, dict] = {}
        ps = self.pixel_size_um
        for r in regionprops(lab):
            if r.area == 0:
                continue
            try:
                convex_area = r.convex_area
            except Exception:
                minr, minc, maxr, maxc = r.bbox
                crop = (lab[minr:maxr, minc:maxc] == r.label)
                ch = convex_hull_image(crop)
                convex_area = float(ch.sum())

            perim = getattr(r, "perimeter", None)
            if perim is None or perim == 0:
                perim = getattr(r, "perimeter_crofton", 0.0)

            feret_max = getattr(r, "feret_diameter_max", None)
            if feret_max is None or feret_max == 0:
                minr, minc, maxr, maxc = r.bbox
                feret_max = math.hypot(maxr - minr, maxc - minc)

            area_px = float(r.area)
            props[r.label] = dict(
                label=int(r.label),
                area_um2=area_px * (ps**2),
                perimeter_um=float(perim) * ps,
                equivalent_diameter_um=float(r.equivalent_diameter) * ps,
                major_axis_um=float(r.major_axis_length) * ps,
                minor_axis_um=float(r.minor_axis_length) * ps,
                aspect_ratio=(r.major_axis_length / r.minor_axis_length) if r.minor_axis_length > 0 else np.nan,
                eccentricity=float(r.eccentricity),
                orientation_rad=float(r.orientation),
                circularity=float((4.0 * math.pi * area_px) / (perim**2) if perim > 0 else np.nan),
                solidity=float(r.solidity),
                extent=float(r.extent),
                convex_area_um2=float(convex_area) * (ps**2),
                feret_diameter_max_um=float(feret_max) * ps,
                bbox_area_um2=float((r.bbox_area if hasattr(r, "bbox_area") else (r.bbox[2]-r.bbox[0])*(r.bbox[3]-r.bbox[1]))) * (ps**2),
                euler_number=float(r.euler_number),
                centroid_row=float(r.centroid[0]),
                centroid_col=float(r.centroid[1]),
            )
            try:
                minr, minc, maxr, maxc = r.bbox
                crop = (lab[minr:maxr, minc:maxc] == r.label)
                ch = convex_hull_image(crop)
                ch_perim = find_boundaries(ch, mode="inner").sum()
                props[r.label]["convexity"] = (ch_perim / perim) if perim > 0 else np.nan
            except Exception:
                props[r.label]["convexity"] = np.nan
        return props

    @staticmethod
    def _skeleton_features(mask: np.ndarray, pixel_size_um: float) -> Tuple[dict, np.ndarray]:
        skel = skeletonize(mask)
        skel_len_um = skel.sum() * pixel_size_um
        kernel = np.array([[1,1,1],[1,10,1],[1,1,1]], dtype=np.uint8)
        neigh = convolve(skel.astype(np.uint8), kernel, mode="constant", cval=0)
        neighbor_count = neigh - (skel.astype(np.uint8) * 10)
        endpoints = np.logical_and(skel, neighbor_count == 1).sum()
        junctions = np.logical_and(skel, neighbor_count >= 3).sum()
        return (dict(skeleton_length_um=float(skel_len_um),
                     skeleton_endpoints=int(endpoints),
                     skeleton_junctions=int(junctions)),
                skel)

    @staticmethod
    def _pairwise_dist(coords_um: np.ndarray) -> np.ndarray:
        if coords_um.shape[0] < 2:
            return np.zeros((coords_um.shape[0], coords_um.shape[0]))
        diff = coords_um[:, None, :] - coords_um[None, :, :]
        return np.sqrt((diff**2).sum(axis=2))

    @staticmethod
    def _mst_total_length(coords_um: np.ndarray) -> float:
        if coords_um.shape[0] < 2:
            return 0.0
        D = Actin_nucleus_feature_extractor._pairwise_dist(coords_um)
        mst = minimum_spanning_tree(D)
        return float(mst.toarray().sum())

    # ---------- Angle helpers ----------

    @staticmethod
    def _fold_axial_angles_rad(a: np.ndarray) -> np.ndarray:
        a = np.mod(a, np.pi)
        a = np.where(a > (np.pi/2), np.pi - a, a)
        return a

    @staticmethod
    def _angle_to_vertical_deg(theta_rad: float) -> float:
        a = Actin_nucleus_feature_extractor._fold_axial_angles_rad(np.array([theta_rad], dtype=float))[0]
        return float(a * RAD2DEG)

    @staticmethod
    def _nematic_order_relative(theta_list: np.ndarray, ref_theta: float) -> float:
        if theta_list.size == 0:
            return np.nan
        delta = theta_list - ref_theta
        return float(np.nanmean(np.cos(2.0 * delta)))

    # ---------- Geometry helpers ----------

    @staticmethod
    def _obb_and_feret_from_points(yx: np.ndarray, angle_samples: int) -> Tuple[float, float, float, float]:
        if yx.shape[0] == 0:
            return (np.nan, np.nan, np.nan, np.nan)
        angles = np.linspace(0.0, np.pi/2, num=max(3, angle_samples), endpoint=False, dtype=float)
        best_area = np.inf
        best = (np.nan, np.nan, np.nan, np.nan)
        y = yx[:,0].astype(float); x = yx[:,1].astype(float)
        for a in angles:
            u = np.array([np.cos(a), -np.sin(a)])
            v = np.array([np.sin(a),  np.cos(a)])
            tu = y*u[0] + x*u[1]
            tv = y*v[0] + x*v[1]
            L = float(tu.max() - tu.min())
            W = float(tv.max() - tv.min())
            area = L * W
            if area < best_area:
                best_area = area
                best = (W, L, area, a)
        return best

    @staticmethod
    def _gini(x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        x = x[x >= 0]
        if x.size == 0:
            return np.nan
        mean = np.mean(x)
        if mean == 0:
            return 0.0
        diffsum = np.abs(x[:, None] - x[None, :]).mean()
        return float(diffsum / (2.0 * mean))

    @staticmethod
    def _geodesic_diameter_um(skel: np.ndarray, pixel_size_um: float) -> float:
        coords = np.argwhere(skel)
        if coords.shape[0] < 2:
            return 0.0
        idx_map = -np.ones(skel.shape, dtype=int)
        for i, (r,c) in enumerate(coords):
            idx_map[r,c] = i
        rows = []; cols = []; data = []
        for i, (r,c) in enumerate(coords):
            for dr in (-1,0,1):
                for dc in (-1,0,1):
                    if dr==0 and dc==0:
                        continue
                    rr = r+dr; cc = c+dc
                    if 0 <= rr < skel.shape[0] and 0 <= cc < skel.shape[1] and skel[rr,cc]:
                        j = idx_map[rr,cc]
                        if j >= 0:
                            w = math.hypot(dr, dc)
                            rows.append(i); cols.append(j); data.append(w)
        if len(rows) == 0:
            return 0.0
        G = csr_matrix((data, (rows, cols)), shape=(coords.shape[0], coords.shape[0]))
        dist0, _ = dijkstra(G, indices=[0], return_predecessors=True)
        s = int(np.nanargmax(dist0[0]))
        dists, _ = dijkstra(G, indices=[s], return_predecessors=True)
        t = int(np.nanargmax(dists[0]))
        geod_px = float(dists[0, t])
        return geod_px * pixel_size_um

    # ------------- Nucleus-to-cell assignment -------------

    def assign_nuclei_to_cells(self, cell_labels: np.ndarray, nuclei_labels: np.ndarray) -> Dict[int, List[int]]:
        p = self.assign_params
        mapping: Dict[int, List[int]] = {lab: [] for lab in np.unique(cell_labels) if lab != 0}
        cell_centroids = {r.label: np.array([r.centroid[0], r.centroid[1]], dtype=float) for r in regionprops(cell_labels)}

        for nr in regionprops(nuclei_labels):
            nlab = nr.label
            ncent = np.array([nr.centroid[0], nr.centroid[1]], dtype=float)

            cell_at_centroid = int(cell_labels[int(round(ncent[0])), int(round(ncent[1]))]) if (
                0 <= int(round(ncent[0])) < cell_labels.shape[0] and 0 <= int(round(ncent[1])) < cell_labels.shape[1]
            ) else 0

            if p.keep_centroid_inside_only and cell_at_centroid == 0:
                continue

            if cell_at_centroid != 0:
                closest_cell = cell_at_centroid
            else:
                if not cell_centroids:
                    continue
                dists = {clab: np.linalg.norm(ncent - ccent) for clab, ccent in cell_centroids.items()}
                closest_cell = min(dists, key=dists.get)

            minr, minc, maxr, maxc = nr.bbox
            nuc_crop = (nuclei_labels[minr:maxr, minc:maxc] == nlab)
            cell_crop = (cell_labels[minr:maxr, minc:maxc] == closest_cell)
            overlap = (nuc_crop & cell_crop).sum()
            frac = overlap / float(nuc_crop.sum() + 1e-9)

            if frac >= p.overlap_min_frac or cell_at_centroid == closest_cell:
                mapping.setdefault(closest_cell, []).append(nlab)

        return mapping

    # ------------- Feature extraction -------------

    def pipeline(self, cell_labels: np.ndarray, nuclei_labels: np.ndarray, overlay_save_dir: Optional[str] = None) -> pd.DataFrame:
        # Cleanup (remove small, edge touching object) & relabel
        cell_labels = self._prepare_labels(cell_labels, self.min_cell_area_px, connectivity=1)
        nuclei_labels = self._prepare_labels(nuclei_labels, self.min_nuc_area_px, connectivity=1)

        # Basic shape features
        cell_props = self._regionprops_basic(cell_labels)
        nuc_props = self._regionprops_basic(nuclei_labels)

        # Precompute nucleus info
        nuclei_centroids_px: Dict[int, np.ndarray] = {}
        nuclei_boundary_idx: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        nuclei_masks: Dict[int, Tuple[slice, slice, np.ndarray]] = {}
        nuc_orientations_rad: Dict[int, float] = {}
        for nr in regionprops(nuclei_labels):
            nlab = nr.label
            nuclei_centroids_px[nlab] = np.array([nr.centroid[0], nr.centroid[1]], dtype=float)
            minr, minc, maxr, maxc = nr.bbox
            crop = (nuclei_labels[minr:maxr, minc:maxc] == nlab)
            er = binary_erosion(crop, structure=np.ones((3,3), dtype=bool))
            boundary = crop & (~er)
            by, bx = np.where(boundary)
            nuclei_boundary_idx[nlab] = (by + minr, bx + minc)
            nuclei_masks[nlab] = (slice(minr, maxr), slice(minc, maxc), crop.copy())
            nuc_orientations_rad[nlab] = float(nr.orientation)

        # Assign nuclei to (filtered) cells
        assigned = self.assign_nuclei_to_cells(cell_labels, nuclei_labels)

        rows: List[dict] = []

        # Derive overlay target directory (if requested)
        overlay_dir = overlay_save_dir if overlay_save_dir is not None else self.overlay_dir
        if self.save_overlays and overlay_dir is None:
            overlay_dir = os.path.join(os.getcwd(), "cell_overlays")

        # Per-cell loop
        for r in regionprops(cell_labels):
            clab = r.label
            cfeat = cell_props.get(clab, {}).copy()
            if not cfeat:
                continue

            # Skip cells with no assigned nuclei
            assigned_nucs = assigned.get(clab, [])
            if len(assigned_nucs) == 0:
                continue
            cfeat["nuclear_count"] = int(len(assigned_nucs))

            minr, minc, maxr, maxc = r.bbox
            cell_crop = (cell_labels[minr:maxr, minc:maxc] == clab)

            # Cell boundary points for OBB/Feret
            er_cell = binary_erosion(cell_crop, structure=np.ones((3,3), dtype=bool))
            cell_boundary = cell_crop & (~er_cell)
            by, bx = np.where(cell_boundary)
            by_full = by + minr; bx_full = bx + minc
            boundary_points = np.column_stack([by_full, bx_full])

            # Skeleton
            try:
                skf, skel = self._skeleton_features(cell_crop, self.pixel_size_um)
            except Exception:
                skf, skel = (dict(skeleton_length_um=np.nan, skeleton_endpoints=np.nan, skeleton_junctions=np.nan),
                             np.zeros_like(cell_crop, dtype=bool))
            cfeat.update(skf)

            # Thickness map
            Dcell_px = distance_transform_edt(cell_crop)
            Dcell_um = Dcell_px * self.pixel_size_um
            thick_diam_um = 2.0 * Dcell_um[cell_crop]
            if thick_diam_um.size > 0:
                cfeat["thickness_diam_median_um"] = float(np.nanmedian(thick_diam_um))
                cfeat["thickness_diam_p10_um"]    = float(np.nanpercentile(thick_diam_um, 10))
                cfeat["thickness_diam_p90_um"]    = float(np.nanpercentile(thick_diam_um, 90))
                cfeat["thickness_diam_mean_um"]   = float(np.nanmean(thick_diam_um))
                cfeat["thickness_diam_std_um"]    = float(np.nanstd(thick_diam_um))
                mean = cfeat["thickness_diam_mean_um"]
                cfeat["thickness_diam_cv"]        = float(cfeat["thickness_diam_std_um"]/mean) if mean and mean>0 else np.nan
            else:
                cfeat["thickness_diam_median_um"] = np.nan
                cfeat["thickness_diam_p10_um"]    = np.nan
                cfeat["thickness_diam_p90_um"]    = np.nan
                cfeat["thickness_diam_mean_um"]   = np.nan
                cfeat["thickness_diam_std_um"]    = np.nan
                cfeat["thickness_diam_cv"]        = np.nan

            # Skeleton radius stats
            skel_idx = np.argwhere(skel)
            if skel_idx.shape[0] > 0:
                rad_um_vals = Dcell_um[skel]
                if rad_um_vals.size > 0:
                    cfeat["skel_radius_mean_um"]   = float(np.nanmean(rad_um_vals))
                    cfeat["skel_radius_median_um"] = float(np.nanmedian(rad_um_vals))
                    cfeat["skel_radius_max_um"]    = float(np.nanmax(rad_um_vals))
                    cfeat["skel_diameter_mean_um"] = 2.0 * cfeat["skel_radius_mean_um"]
                    cfeat["skel_diameter_median_um"] = 2.0 * cfeat["skel_radius_median_um"]
                    cfeat["skel_diameter_max_um"]  = 2.0 * cfeat["skel_radius_max_um"]
                else:
                    cfeat["skel_radius_mean_um"] = cfeat["skel_radius_median_um"] = cfeat["skel_radius_max_um"] = np.nan
                    cfeat["skel_diameter_mean_um"] = cfeat["skel_diameter_median_um"] = cfeat["skel_diameter_max_um"] = np.nan
            else:
                cfeat["skel_radius_mean_um"] = cfeat["skel_radius_median_um"] = cfeat["skel_radius_max_um"] = np.nan
                cfeat["skel_diameter_mean_um"] = cfeat["skel_diameter_median_um"] = cfeat["skel_diameter_max_um"] = np.nan

            # Geodesic diameter
            try:
                cfeat["geodesic_diameter_um"] = self._geodesic_diameter_um(skel, self.pixel_size_um)
            except Exception:
                cfeat["geodesic_diameter_um"] = np.nan

            # OBB / Feret
            try:
                minW_px, maxL_px, obb_area_px2, best_angle = self._obb_and_feret_from_points(boundary_points, self.obb_angle_samples)
                cfeat["min_feret_diameter_um"] = float(minW_px) * self.pixel_size_um
                if np.isfinite(cfeat.get("feret_diameter_max_um", np.nan)) and np.isfinite(minW_px) and minW_px>0:
                    cfeat["feret_ratio"] = float(cfeat["feret_diameter_max_um"] / (minW_px * self.pixel_size_um))
                else:
                    cfeat["feret_ratio"] = np.nan
                cfeat["obb_area_um2"] = float(obb_area_px2) * (self.pixel_size_um**2)
                cfeat["obb_width_um"] = float(minW_px) * self.pixel_size_um
                cfeat["obb_length_um"] = float(maxL_px) * self.pixel_size_um
                if cfeat.get("area_um2", 0) > 0 and np.isfinite(obb_area_px2) and obb_area_px2>0:
                    cfeat["rectangularity"] = float(cfeat["area_um2"] / (obb_area_px2 * (self.pixel_size_um**2)))
                else:
                    cfeat["rectangularity"] = np.nan
            except Exception:
                cfeat["min_feret_diameter_um"] = np.nan
                cfeat["feret_ratio"] = np.nan
                cfeat["obb_area_um2"] = np.nan
                cfeat["obb_width_um"] = np.nan
                cfeat["obb_length_um"] = np.nan
                cfeat["rectangularity"] = np.nan
            # (assigned_nucs already set above)

            union_nuc_crop = np.zeros_like(cell_crop, dtype=bool)
            nuc_areas_um2 = []; nuc_circularities = []; nuc_eccs = []; nuc_solids = []
            nuc_centroids_px_list = []; nuc_orients = []

            for nlab in assigned_nucs:
                (rs, cs, crop) = nuclei_masks[nlab]
                rr0 = max(0, rs.start - minr); rr1 = min(cell_crop.shape[0], rs.stop - minr)
                cc0 = max(0, cs.start - minc); cc1 = min(cell_crop.shape[1], cs.stop - minc)
                rr0n = max(0, minr - rs.start); rr1n = rr0n + (rr1 - rr0)
                cc0n = max(0, minc - cs.start); cc1n = cc0n + (cc1 - cc0)
                if rr1 > rr0 and cc1 > cc0:
                    union_nuc_crop[rr0:rr1, cc0:cc1] |= crop[rr0n:rr1n, cc0n:cc1n]

                npd = nuc_props.get(nlab, None)
                if npd is not None:
                    nuc_areas_um2.append(npd["area_um2"])
                    nuc_circularities.append(npd["circularity"])
                    nuc_eccs.append(npd["eccentricity"])
                    nuc_solids.append(npd["solidity"])
                    nuc_orients.append(nuc_orientations_rad[nlab])

                nuc_centroids_px_list.append(nuclei_centroids_px[nlab])

            def _agg(vals: List[float], prefix: str) -> dict:
                if len(vals) == 0:
                    return {f"{prefix}_mean": np.nan, f"{prefix}_std": np.nan, f"{prefix}_cv": np.nan,
                            f"{prefix}_min": np.nan, f"{prefix}_max": np.nan}
                arr = np.asarray(vals, dtype=float)
                mean = float(np.nanmean(arr)); std = float(np.nanstd(arr))
                cv = float(std/mean) if mean != 0 and np.isfinite(mean) else np.nan
                return {f"{prefix}_mean": mean, f"{prefix}_std": std, f"{prefix}_cv": cv,
                        f"{prefix}_min": float(np.nanmin(arr)), f"{prefix}_max": float(np.nanmax(arr))}

            cfeat.update(_agg(nuc_areas_um2, "nuc_area_um2"))
            cfeat.update(_agg(nuc_circularities, "nuc_circularity"))
            cfeat.update(_agg(nuc_eccs, "nuc_eccentricity"))
            cfeat.update(_agg(nuc_solids, "nuc_solidity"))
            cfeat["nuclear_total_area_um2"] = float(np.nansum(nuc_areas_um2)) if len(nuc_areas_um2)>0 else np.nan
            cfeat["nuclear_to_cell_area_ratio"] = (cfeat["nuclear_total_area_um2"] / cfeat["area_um2"]) if (
                np.isfinite(cfeat.get("area_um2", np.nan)) and cfeat.get("area_um2", 0) > 0 and np.isfinite(cfeat.get("nuclear_total_area_um2", np.nan))
            ) else np.nan

            # Arrangement vs. cell axis
            theta_cell = cfeat.get("orientation_rad", 0.0)
            u_major = np.array([math.cos(theta_cell), -math.sin(theta_cell)])
            u_minor = np.array([math.sin(theta_cell),  math.cos(theta_cell)])
            cell_centroid_px = np.array([cfeat["centroid_row"], cfeat["centroid_col"]])
            maj_len_um = cfeat.get("major_axis_um", np.nan)
            half_major_um = 0.5 * maj_len_um if np.isfinite(maj_len_um) else np.nan

            nuc_centroids_um = None
            if len(nuc_centroids_px_list) > 0:
                nuc_centroids_px_arr = np.vstack(nuc_centroids_px_list)
                nuc_centroids_um = nuc_centroids_px_arr * self.pixel_size_um
                cell_centroid_um = cell_centroid_px * self.pixel_size_um

            if nuc_centroids_um is not None:
                mean_nuc_centroid_um = nuc_centroids_um.mean(axis=0)
                offset_um = float(np.linalg.norm(mean_nuc_centroid_um - cell_centroid_um))
                cfeat["nuc_centroid_offset_um"] = offset_um
                eq_radius_um = math.sqrt(max(cfeat["area_um2"], 0.0) / math.pi)
                cfeat["nuc_centroid_offset_norm"] = (offset_um / eq_radius_um) if eq_radius_um > 0 else np.nan
            else:
                cfeat["nuc_centroid_offset_um"] = np.nan
                cfeat["nuc_centroid_offset_norm"] = np.nan

            if nuc_centroids_um is not None and np.isfinite(half_major_um) and half_major_um > 0:
                rel_um = nuc_centroids_um - cell_centroid_um
                s_along = rel_um @ (u_major * self.pixel_size_um)
                s_across = rel_um @ (u_minor * self.pixel_size_um)
                s_norm = s_along / half_major_um
                cfeat["nuc_axial_mean"] = float(np.nanmean(s_norm))
                cfeat["nuc_axial_std"] = float(np.nanstd(s_norm))
                left = np.sum(s_across < 0); right = np.sum(s_across >= 0); k = len(s_across)
                cfeat["nuc_left_right_asym"] = float(abs(right - left) / k) if k > 0 else np.nan
                norms = np.linalg.norm(rel_um, axis=1)
                unit_vecs = np.where(norms[:, None] > 0, rel_um / norms[:, None], 0.0)
                cfeat["nuc_polarity_index"] = float(np.linalg.norm(unit_vecs.mean(axis=0))) if unit_vecs.size > 0 else np.nan
            else:
                cfeat["nuc_axial_mean"] = np.nan
                cfeat["nuc_axial_std"] = np.nan
                cfeat["nuc_left_right_asym"] = np.nan
                cfeat["nuc_polarity_index"] = np.nan

            # Spacing metrics
            if nuc_centroids_um is not None and nuc_centroids_um.shape[0] >= 2:
                D = self._pairwise_dist(nuc_centroids_um)
                np.fill_diagonal(D, np.nan)
                nnd = np.nanmin(D, axis=1)
                cfeat["nuc_nnd_mean_um"] = float(np.nanmean(nnd))
                cfeat["nuc_nnd_std_um"] = float(np.nanstd(nnd))
                mean = cfeat["nuc_nnd_mean_um"]
                cfeat["nuc_nnd_cv"] = float(cfeat["nuc_nnd_std_um"]/mean) if mean and mean>0 else np.nan
                cfeat["nuc_mst_length_um"] = self._mst_total_length(nuc_centroids_um)
                cfeat["nuc_max_pair_dist_um"] = float(np.nanmax(D))
                cfeat["nuc_spread_over_major"] = (cfeat["nuc_max_pair_dist_um"]/maj_len_um) if (maj_len_um and maj_len_um>0) else np.nan
            else:
                cfeat["nuc_nnd_mean_um"] = np.nan
                cfeat["nuc_nnd_std_um"] = np.nan
                cfeat["nuc_nnd_cv"] = np.nan
                cfeat["nuc_mst_length_um"] = np.nan
                cfeat["nuc_max_pair_dist_um"] = np.nan
                cfeat["nuc_spread_over_major"] = np.nan

            # Cytoplasm thickness metrics around nuclear boundaries
            if len(assigned_nucs) > 0:
                all_bdists = []
                for nlab in assigned_nucs:
                    byn, bxn = nuclei_boundary_idx[nlab]
                    mask_in_bbox = (byn >= minr) & (byn < maxr) & (bxn >= minc) & (bxn < maxc)
                    byb = byn[mask_in_bbox] - minr
                    bxb = bxn[mask_in_bbox] - minc
                    if byb.size > 0:
                        all_bdists.extend(Dcell_um[byb, bxb].tolist())
                if len(all_bdists) > 0:
                    all_bdists = np.array(all_bdists, dtype=float)
                    cfeat["cyto_thickness_median_um"] = float(np.nanmedian(all_bdists))
                    cfeat["cyto_thickness_min_um"] = float(np.nanmin(all_bdists))
                    cfeat["cyto_boundary_frac_thin"] = float(np.mean(all_bdists < self.rim_threshold_um))
                else:
                    cfeat["cyto_thickness_median_um"] = np.nan
                    cfeat["cyto_thickness_min_um"] = np.nan
                    cfeat["cyto_boundary_frac_thin"] = np.nan
            else:
                cfeat["cyto_thickness_median_um"] = np.nan
                cfeat["cyto_thickness_min_um"] = np.nan
                cfeat["cyto_boundary_frac_thin"] = np.nan

            # Nuclear packing fraction
            try:
                if union_nuc_crop.any():
                    ch = convex_hull_image(union_nuc_crop)
                    hull_area_um2 = float(ch.sum()) * (self.pixel_size_um**2)
                    cfeat["nuc_packing_fraction"] = hull_area_um2 / cfeat["area_um2"] if cfeat["area_um2"]>0 else np.nan
                else:
                    cfeat["nuc_packing_fraction"] = np.nan
            except Exception:
                cfeat["nuc_packing_fraction"] = np.nan

            # Alignment/direction
            cfeat["cell_angle_to_vertical_deg"] = self._angle_to_vertical_deg(theta_cell)
            cfeat["cell_nematic_vertical_S"] = float(np.cos(2.0 * theta_cell))

            nuc_orients_arr = np.array(nuc_orients, dtype=float) if len(nuc_orients)>0 else np.array([])
            if nuc_orients_arr.size > 0:
                angles_vert_deg = self._fold_axial_angles_rad(nuc_orients_arr) * RAD2DEG
                cfeat["nuc_angle_to_vertical_deg_mean"] = float(np.nanmean(angles_vert_deg))
                cfeat["nuc_angle_to_vertical_deg_std"]  = float(np.nanstd(angles_vert_deg))
                cfeat["nuc_angle_to_vertical_deg_min"]  = float(np.nanmin(angles_vert_deg))
                cfeat["nuc_angle_to_vertical_deg_max"]  = float(np.nanmax(angles_vert_deg))
                mis = self._fold_axial_angles_rad(np.abs(nuc_orients_arr - theta_cell))
                mis_deg = mis * RAD2DEG
                cfeat["nuc_misalignment_to_cell_deg_mean"] = float(np.nanmean(mis_deg))
                cfeat["nuc_misalignment_to_cell_deg_std"]  = float(np.nanstd(mis_deg))
                cfeat["nuc_misalignment_to_cell_deg_min"]  = float(np.nanmin(mis_deg))
                cfeat["nuc_misalignment_to_cell_deg_max"]  = float(np.nanmax(mis_deg))
                cfeat["nuc_alignment_S_to_cell"] = self._nematic_order_relative(nuc_orients_arr, theta_cell)
                cfeat["nuc_nematic_vertical_S"] = float(np.nanmean(np.cos(2.0 * nuc_orients_arr)))
                R = np.abs(np.nanmean(np.exp(1j*2.0*mis)))
                cfeat["nuc_alignment_circvar_rel_cell"] = float(1.0 - R)
                cfeat["nuc_alignment_frac_within_15deg"] = float(np.mean(mis_deg <= 15.0))
            else:
                cfeat["nuc_angle_to_vertical_deg_mean"] = np.nan
                cfeat["nuc_angle_to_vertical_deg_std"]  = np.nan
                cfeat["nuc_angle_to_vertical_deg_min"]  = np.nan
                cfeat["nuc_angle_to_vertical_deg_max"]  = np.nan
                cfeat["nuc_misalignment_to_cell_deg_mean"] = np.nan
                cfeat["nuc_misalignment_to_cell_deg_std"]  = np.nan
                cfeat["nuc_misalignment_to_cell_deg_min"]  = np.nan
                cfeat["nuc_misalignment_to_cell_deg_max"]  = np.nan
                cfeat["nuc_alignment_S_to_cell"] = np.nan
                cfeat["nuc_nematic_vertical_S"] = np.nan
                cfeat["nuc_alignment_circvar_rel_cell"] = np.nan
                cfeat["nuc_alignment_frac_within_15deg"] = np.nan

            # Chord-width profile along major axis
            yy, xx = np.where(cell_crop)
            if yy.size > 0 and np.isfinite(maj_len_um) and maj_len_um>0:
                pts = np.column_stack([yy + minr, xx + minc]).astype(float)
                c0 = np.array([cfeat["centroid_row"], cfeat["centroid_col"]], dtype=float)
                rel = pts - c0
                t = rel @ u_major
                s = rel @ u_minor
                tb = np.linspace(t.min(), t.max(), num=max(5,self.chord_bins)+1)
                widths = []
                for i in range(len(tb)-1):
                    mask = (t >= tb[i]) & (t < tb[i+1])
                    if np.any(mask):
                        s_slice = s[mask]
                        widths.append(float(s_slice.max() - s_slice.min()))
                widths = np.asarray(widths, dtype=float)
                if widths.size > 0:
                    widths_um = widths * self.pixel_size_um
                    cfeat["chord_bins_n"] = int(widths.size)
                    cfeat["chord_width_mean_um"] = float(np.nanmean(widths_um))
                    cfeat["chord_width_std_um"]  = float(np.nanstd(widths_um))
                    q = max(1, widths.size // 4)
                    prox_mean = float(np.nanmean(widths_um[:q]))
                    dist_mean = float(np.nanmean(widths_um[-q:]))
                    cfeat["chord_width_prox25_um"] = prox_mean
                    cfeat["chord_width_dist25_um"] = dist_mean
                    cfeat["chord_tapering_index"]  = float(dist_mean / prox_mean) if prox_mean>0 else np.nan
                else:
                    cfeat["chord_bins_n"] = 0
                    cfeat["chord_width_mean_um"] = np.nan
                    cfeat["chord_width_std_um"]  = np.nan
                    cfeat["chord_width_prox25_um"] = np.nan
                    cfeat["chord_width_dist25_um"] = np.nan
                    cfeat["chord_tapering_index"]  = np.nan
            else:
                cfeat["chord_bins_n"] = 0
                cfeat["chord_width_mean_um"] = np.nan
                cfeat["chord_width_std_um"]  = np.nan
                cfeat["chord_width_prox25_um"] = np.nan
                cfeat["chord_width_dist25_um"] = np.nan
                cfeat["chord_tapering_index"]  = np.nan

            # Voronoi-in-cell & radial nuclear distribution
            if len(assigned_nucs) >= 1:
                if nuc_centroids_um is not None:
                    r_um = np.linalg.norm(nuc_centroids_um - cell_centroid_um, axis=1)
                    eq_radius_um = math.sqrt(max(cfeat["area_um2"], 0.0) / math.pi)
                    if eq_radius_um > 0:
                        r_norm = r_um / eq_radius_um
                        cfeat["nuc_radial_rnorm_median"] = float(np.nanmedian(r_norm))
                        cfeat["nuc_radial_rnorm_p10"]    = float(np.nanpercentile(r_norm, 10))
                        cfeat["nuc_radial_rnorm_p90"]    = float(np.nanpercentile(r_norm, 90))
                        cfeat["nuc_centrality_frac_leq_0_5"]   = float(np.mean(r_norm <= 0.5))
                        cfeat["nuc_peripherality_frac_geq_0_8"] = float(np.mean(r_norm >= 0.8))
                    else:
                        cfeat["nuc_radial_rnorm_median"] = np.nan
                        cfeat["nuc_radial_rnorm_p10"] = np.nan
                        cfeat["nuc_radial_rnorm_p90"] = np.nan
                        cfeat["nuc_centrality_frac_leq_0_5"] = np.nan
                        cfeat["nuc_peripherality_frac_geq_0_8"] = np.nan
                else:
                    cfeat["nuc_radial_rnorm_median"] = np.nan
                    cfeat["nuc_radial_rnorm_p10"] = np.nan
                    cfeat["nuc_radial_rnorm_p90"] = np.nan
                    cfeat["nuc_centrality_frac_leq_0_5"] = np.nan
                    cfeat["nuc_peripherality_frac_geq_0_8"] = np.nan

                P = int(cell_crop.sum())
                if P > 0 and P <= self.voronoi_max_pixels and len(assigned_nucs) >= 1:
                    yyx, xxy = np.where(cell_crop)
                    py = yyx + minr; px = xxy + minc
                    pts = np.column_stack([py, px]).astype(float)
                    centers = np.vstack([nuclei_centroids_px[nlab] for nlab in assigned_nucs]).astype(float)
                    dif = pts[:, None, :] - centers[None, :, :]
                    dist2 = np.sum(dif**2, axis=2)
                    labels = np.argmin(dist2, axis=1)
                    counts = np.bincount(labels, minlength=len(assigned_nucs)).astype(float)
                    areas_um2 = counts * (self.pixel_size_um**2)
                    cfeat["voronoi_area_mean_um2"] = float(np.nanmean(areas_um2)) if areas_um2.size>0 else np.nan
                    cfeat["voronoi_area_std_um2"]  = float(np.nanstd(areas_um2)) if areas_um2.size>0 else np.nan
                    mean = cfeat["voronoi_area_mean_um2"]
                    cfeat["voronoi_area_cv"]       = float(cfeat["voronoi_area_std_um2"]/mean) if mean and mean>0 else np.nan
                    cfeat["voronoi_gini"]          = self._gini(areas_um2)
                else:
                    cfeat["voronoi_area_mean_um2"] = np.nan
                    cfeat["voronoi_area_std_um2"]  = np.nan
                    cfeat["voronoi_area_cv"]       = np.nan
                    cfeat["voronoi_gini"]          = np.nan
            else:
                cfeat["nuc_radial_rnorm_median"] = np.nan
                cfeat["nuc_radial_rnorm_p10"] = np.nan
                cfeat["nuc_radial_rnorm_p90"] = np.nan
                cfeat["nuc_centrality_frac_leq_0_5"] = np.nan
                cfeat["nuc_peripherality_frac_geq_0_8"] = np.nan
                cfeat["voronoi_area_mean_um2"] = np.nan
                cfeat["voronoi_area_std_um2"]  = np.nan
                cfeat["voronoi_area_cv"]       = np.nan
                cfeat["voronoi_gini"]          = np.nan

            # Save overlay if enabled
            if self.save_overlays and overlay_dir is not None:
                try:
                    self._save_overlay(overlay_dir, int(clab), cell_crop, union_nuc_crop, fmt=self.overlay_format)
                except Exception:
                    pass

            rows.append(cfeat)

        return pd.DataFrame(rows)

    # ------------- Convenience wrapper -------------

    def extract_from_paths(self, cell_mask_path: str, nuclei_mask_path: str) -> pd.DataFrame:
        cell = self.load_mask(cell_mask_path)
        nuc  = self.load_mask(nuclei_mask_path)

        # Decide default overlay directory when saving overlays
        overlay_dir = self.overlay_dir
        if self.save_overlays and overlay_dir is None:
            base = os.path.splitext(os.path.basename(cell_mask_path))[0]
            parent = os.path.dirname(cell_mask_path) or os.getcwd()
            overlay_dir = os.path.join(parent, f"{base}_cell_overlays")

        df = self.pipeline(cell, nuc, overlay_save_dir=overlay_dir)
        df.insert(0, "cell_mask_path", os.path.basename(cell_mask_path))
        df.insert(1, "nuc_mask_path",  os.path.basename(nuclei_mask_path))
        return df


# ---------------- Minimal CLI ----------------
def _main():
    import argparse
    p = argparse.ArgumentParser(description="Alpha-SMA / DAPI mask-only morphometrics (advanced + overlays, edge-filter)")
    p.add_argument("--cell", required=True)
    p.add_argument("--nuclei", required=True)
    p.add_argument("--pixel-size-um", type=float, default=1.0)
    p.add_argument("--min-cell-area-px", type=int, default=20)
    p.add_argument("--min-nuc-area-px", type=int, default=10)
    p.add_argument("--overlap-min-frac", type=float, default=0.1)
    p.add_argument("--centroid-inside-only", action="store_true")
    p.add_argument("--rim-threshold-um", type=float, default=2.0)
    p.add_argument("--obb-angle-samples", type=int, default=45)
    p.add_argument("--chord-bins", type=int, default=25)
    p.add_argument("--voronoi-max-pixels", type=int, default=200000)
    p.add_argument("--save-overlays", action="store_true")
    p.add_argument("--overlay-dir", type=str, default=None)
    p.add_argument("--overlay-format", type=str, default="png")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    extractor = Actin_nucleus_feature_extractor(
        pixel_size_um=args.pixel_size_um,
        min_cell_area_px=args.min_cell_area_px,
        min_nuc_area_px=args.min_nuc_area_px,
        assign_params=AssignParams(overlap_min_frac=args.overlap_min_frac,
                                   keep_centroid_inside_only=args.centroid_inside_only),
        rim_threshold_um=args.rim_threshold_um,
        obb_angle_samples=args.obb_angle_samples,
        chord_bins=args.chord_bins,
        voronoi_max_pixels=args.voronoi_max_pixels,
        save_overlays=args.save_overlays,
        overlay_dir=args.overlay_dir,
        overlay_format=args.overlay_format
    )
    df = extractor.extract_from_paths(args.cell, args.nuclei)
    df.to_csv(args.out, index=False)
    print(f"Wrote {args.out} with {len(df)} cells.")


if __name__ == "__main__":
    _main()
