# Chanhong Min <cmin11@jhmi.edu>

# Copyright 2023 The Phillip tiME Lab at the Johns Hopkins University
# All rights reserved.
#
# Licensed under a modified Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.github.com/Phillip-Lab-JHU/
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import os
import json
import math
import re
import numpy as np
from skimage import io
import napari
import skimage
import matplotlib.pyplot as plt
from tqdm import tqdm
import tifffile
from utils.img_utils import *
import scipy
import seaborn as sns
import pyclesperanto_prototype as cle
from utils.traj_utils import *
############################### Read FDC masks ###############################

import h5py


def imaris_index(key):
    match = re.search(r"\d+$", key)
    if match is None:
        return (1, key)
    return (0, int(match.group()))


def sort_imaris_keys(keys):
    return sorted(keys, key=imaris_index)


def h5_value_to_python(value):
    """Convert h5py/numpy attribute values into JSON-friendly Python values."""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.bytes_):
        return bytes(value).decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        values = [h5_value_to_python(v) for v in value.tolist()]
        if values and all(isinstance(v, str) and len(v) == 1 for v in values):
            return "".join(values)
        return values
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return value


def h5_attrs_to_dict(obj):
    return {key: h5_value_to_python(value) for key, value in obj.attrs.items()}


def collect_h5_metadata(obj):
    """
    Recursively collect HDF5 group attributes and dataset descriptors.

    For image data, this stores shape/dtype/chunks/compression only; it does not
    read the pixel arrays.
    """
    if isinstance(obj, h5py.Dataset):
        return {
            "attrs": h5_attrs_to_dict(obj),
            "shape": obj.shape,
            "dtype": str(obj.dtype),
            "chunks": obj.chunks,
            "compression": obj.compression,
        }

    metadata = {"attrs": h5_attrs_to_dict(obj)}
    for key, child in obj.items():
        metadata[key] = collect_h5_metadata(child)
    return metadata


def find_h5_paths(h5_file, patterns):
    matches = []
    lowered_patterns = [pattern.lower() for pattern in patterns]

    def visit(name, obj):
        lowered_name = name.lower()
        if any(pattern in lowered_name for pattern in lowered_patterns):
            item = {
                "path": name,
                "type": type(obj).__name__,
                "attrs": h5_attrs_to_dict(obj),
            }
            if isinstance(obj, h5py.Dataset):
                item.update(
                    {
                        "shape": obj.shape,
                        "dtype": str(obj.dtype),
                        "chunks": obj.chunks,
                        "compression": obj.compression,
                    }
                )
            matches.append(item)

    h5_file.visititems(visit)
    return matches


def find_segmentation_candidates(h5_file):
    """
    Look for embedded Imaris segmentation/object data.

    Raw image channels are normally stored under DataSet/ResolutionLevel */...
    Imaris segmentations may appear as additional label-like datasets or as
    Surpass/Scene object data such as Spots, Surfaces, Cells, Labels, or Masks.
    """
    return find_h5_paths(
        h5_file,
        [
            "mask",
            "label",
            "segmentation",
            "surface",
            "spots",
            "cells",
            "filaments",
            "surpass",
            "scene",
        ],
    )


def get_channel_info(h5_file):
    channel_info = []
    data_set_info = h5_file.get("DataSetInfo")
    if data_set_info is None:
        return channel_info

    channel_keys = [
        key for key in data_set_info.keys()
        if re.match(r"Channel \d+$", key)
    ]

    for key in sort_imaris_keys(channel_keys):
        attrs = h5_attrs_to_dict(data_set_info[key])
        channel_info.append(
            {
                "index": imaris_index(key)[1],
                "group": key,
                "name": attrs.get("Name"),
                "description": attrs.get("Description"),
                "color": {
                    "red": attrs.get("ColorRed"),
                    "green": attrs.get("ColorGreen"),
                    "blue": attrs.get("ColorBlue"),
                    "alpha": attrs.get("ColorAlpha"),
                },
                "attrs": attrs,
            }
        )

    return channel_info


def get_image_channel_keys(h5_file):
    dataset = h5_file.get("DataSet/ResolutionLevel 0")
    if dataset is None:
        return []

    time_keys = sort_imaris_keys(dataset.keys())
    if not time_keys:
        return []

    return sort_imaris_keys(dataset[time_keys[0]].keys())


def print_channel_info(channel_info):
    if not channel_info:
        print("No DataSetInfo/Channel N metadata found.")
        return

    print("Channel metadata:")
    for channel in channel_info:
        name = channel["name"] or "(no name)"
        color = channel["color"]
        print(
            f"  Channel {channel['index']}: {name}; "
            f"RGB=({color['red']}, {color['green']}, {color['blue']})"
        )


def print_image_channel_order(h5_file):
    channel_keys = get_image_channel_keys(h5_file)
    if not channel_keys:
        print("No image channels found under DataSet/ResolutionLevel 0.")
        return
    print(f"Image channel order in img[:, channel, z, y, x]: {channel_keys}")


def summarize_h5_object(path, obj):
    summary = {
        "path": path,
        "type": type(obj).__name__,
        "attrs": h5_attrs_to_dict(obj),
    }
    if isinstance(obj, h5py.Dataset):
        summary.update(
            {
                "shape": obj.shape,
                "dtype": str(obj.dtype),
                "chunks": obj.chunks,
                "compression": obj.compression,
            }
        )
    return summary


def print_segmentation_candidate_details(h5_file, segmentation_candidates, max_children=25):
    if not segmentation_candidates:
        return

    print("Segmentation/object candidate details:")
    for candidate in segmentation_candidates:
        path = candidate["path"]
        obj = h5_file[path]
        print(json.dumps(summarize_h5_object(path, obj), indent=2))

        if isinstance(obj, h5py.Group):
            child_summaries = []

            def visit(name, child):
                if len(child_summaries) < max_children:
                    child_summaries.append(summarize_h5_object(f"{path}/{name}", child))

            obj.visititems(visit)
            if child_summaries:
                print("  Children:")
                print(json.dumps(child_summaries, indent=2))


def dataset_is_mask_like(dataset, image_shape):
    shape = tuple(dataset.shape)
    if not shape:
        return False

    image_tzyx = tuple(image_shape[0:1] + image_shape[2:])
    image_zyx = tuple(image_shape[2:])
    image_yx = tuple(image_shape[-2:])

    shape_matches_image = (
        shape == image_tzyx
        or shape == image_zyx
        or shape == image_yx
        or shape[-3:] == image_zyx
        or shape[-2:] == image_yx
    )
    if not shape_matches_image:
        return False

    dtype = np.dtype(dataset.dtype)
    return dtype.kind in ("b", "u", "i")


def load_embedded_mask_layers(h5_file, segmentation_candidates, image_shape):
    mask_layers = []
    visited_paths = set()

    def consider_dataset(path, dataset):
        if path in visited_paths:
            return
        visited_paths.add(path)

        if not dataset_is_mask_like(dataset, image_shape):
            print(f"Skipping candidate {path}: shape={dataset.shape}, dtype={dataset.dtype}")
            return

        mask = dataset[:]
        mask = np.squeeze(mask)
        if mask.dtype == bool:
            mask = mask.astype(np.uint8)

        mask_layers.append(
            {
                "name": path.replace("/", " | "),
                "data": mask,
                "path": path,
                "shape": mask.shape,
                "dtype": str(mask.dtype),
            }
        )
        print(f"Loaded mask candidate {path}: shape={mask.shape}, dtype={mask.dtype}")

    for candidate in segmentation_candidates:
        path = candidate["path"]
        obj = h5_file[path]
        if isinstance(obj, h5py.Dataset):
            consider_dataset(path, obj)
        else:
            def visit(name, child):
                child_path = f"{path}/{name}"
                if isinstance(child, h5py.Dataset):
                    consider_dataset(child_path, child)

            obj.visititems(visit)

    return mask_layers


def scale_for_layer(data, voxel_size_um):
    if data.ndim == 4 and is_valid_voxel_size(voxel_size_um):
        return (1, *voxel_size_um)
    if data.ndim == 3 and is_valid_voxel_size(voxel_size_um):
        return voxel_size_um
    if data.ndim == 2 and is_valid_voxel_size(voxel_size_um):
        return voxel_size_um[-2:]
    return None


def get_first_data_shape(h5_file):
    dataset = h5_file.get("DataSet/ResolutionLevel 0")
    if dataset is None:
        return None

    time_keys = sort_imaris_keys(dataset.keys())
    if not time_keys:
        return None

    channel_keys = sort_imaris_keys(dataset[time_keys[0]].keys())
    if not channel_keys:
        return None

    data = dataset[time_keys[0]][channel_keys[0]].get("Data")
    if data is None:
        return None
    return data.shape


def get_float_attr(attrs, key):
    value = attrs.get(key)
    if value is None:
        return None
    value = h5_value_to_python(value)
    if isinstance(value, list):
        value = value[0]
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def positive_float_or_none(value):
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value) or value <= 0:
        return None
    return value


def get_first_available_float(attrs, keys):
    for key in keys:
        value = positive_float_or_none(get_float_attr(attrs, key))
        if value is not None:
            return value
    return None


def extract_voxel_size_um(h5_file):
    """
    Return voxel size as (z_um, y_um, x_um) for napari image scale.

    Imaris usually stores physical extents in DataSetInfo/Image as:
    ExtMin0/ExtMax0 = x, ExtMin1/ExtMax1 = y, ExtMin2/ExtMax2 = z.
    Pixel counts can come from Image attrs X/Y/Z; otherwise the first Data
    array shape is used, which is usually (z, y, x).
    """
    image_info = h5_file.get("DataSetInfo/Image")
    if image_info is None:
        return None

    attrs = image_info.attrs
    data_shape = get_first_data_shape(h5_file)

    n_x = get_float_attr(attrs, "X") or (data_shape[-1] if data_shape else None)
    n_y = get_float_attr(attrs, "Y") or (data_shape[-2] if data_shape else None)
    n_z = get_float_attr(attrs, "Z") or (data_shape[-3] if data_shape else None)

    ext_min_x = get_float_attr(attrs, "ExtMin0")
    ext_max_x = get_float_attr(attrs, "ExtMax0")
    ext_min_y = get_float_attr(attrs, "ExtMin1")
    ext_max_y = get_float_attr(attrs, "ExtMax1")
    ext_min_z = get_float_attr(attrs, "ExtMin2")
    ext_max_z = get_float_attr(attrs, "ExtMax2")

    if None not in (n_x, n_y, n_z, ext_min_x, ext_max_x, ext_min_y, ext_max_y, ext_min_z, ext_max_z):
        x_um = positive_float_or_none(abs(ext_max_x - ext_min_x) / n_x)
        y_um = positive_float_or_none(abs(ext_max_y - ext_min_y) / n_y)
        z_um = positive_float_or_none(abs(ext_max_z - ext_min_z) / n_z)
        if None not in (z_um, y_um, x_um):
            return z_um, y_um, x_um

    # Some .ims files have zero physical extents but still store pixel sizes
    # under microscope/objective-specific metadata names.
    x_um = get_first_available_float(attrs, ["VoxelSizeX", "PixelSizeX", "MicronsPerPixelX", "UmPerPixelX"])
    y_um = get_first_available_float(attrs, ["VoxelSizeY", "PixelSizeY", "MicronsPerPixelY", "UmPerPixelY"])
    z_um = get_first_available_float(attrs, ["VoxelSizeZ", "PixelSizeZ", "MicronsPerPixelZ", "UmPerPixelZ"])
    if None not in (z_um, y_um, x_um):
        return z_um, y_um, x_um

    return None


def is_valid_voxel_size(voxel_size_um):
    if voxel_size_um is None:
        return False
    return all(positive_float_or_none(value) is not None for value in voxel_size_um)


def napari_scale_from_voxel_size(voxel_size_um):
    if not is_valid_voxel_size(voxel_size_um):
        return None
    return (1, *voxel_size_um)


def extract_ims_metadata(h5_file):
    metadata = {
        "root": {"attrs": h5_attrs_to_dict(h5_file)},
        "voxel_size_um_zyx": extract_voxel_size_um(h5_file),
        "channels": get_channel_info(h5_file),
        "segmentation_candidates": find_segmentation_candidates(h5_file),
    }

    for group_name in ["DataSetInfo", "DataSetTimes", "DataSet"]:
        if group_name in h5_file:
            metadata[group_name] = collect_h5_metadata(h5_file[group_name])

    return metadata


path = r'\\philliplab-server.wse.jhu.edu\data\Chanhong\Xi macrophage\data\24h after NP injection\\'
#path = r'\\philliplab-server.wse.jhu.edu\data\Chanhong\Cornell LN Spleen\FDC feature projection\Exp3-7-Good-D11-B2-ZT2-30-117-FOV230-256px\\'
#path = r'C:\Users\ChanhongMin\Downloads\Xi\\'
#path = r'\\philliplab-server.wse.jhu.edu\data\Chanhong\Nia Lab\Kathryn\data\XC Double Recipient\KR571\Young Recipient\ROI1\Young Recipient Cells\\'
files = [file for file in next(os.walk(path))[2] if file.lower().endswith(".ims")]
files.sort()

if not files:
    raise FileNotFoundError(f"No .ims files found in {path}")

ims_file = os.path.join(path, files[0])

with h5py.File(ims_file, "r") as f:
    metadata = extract_ims_metadata(f)
    voxel_size_um = metadata["voxel_size_um_zyx"]
    print(json.dumps(metadata, indent=2))
    print_channel_info(metadata["channels"])
    print_image_channel_order(f)
    if metadata["segmentation_candidates"]:
        print(f"Found {len(metadata['segmentation_candidates'])} segmentation/object candidate paths.")
        print_segmentation_candidate_details(f, metadata["segmentation_candidates"])
    else:
        print("No obvious embedded segmentation mask/object paths found in this .ims file.")
    if not is_valid_voxel_size(voxel_size_um):
        print("No positive voxel size found in metadata. Napari will use default pixel scale.")

    dataset = f["DataSet/ResolutionLevel 0"]

    time_keys = sort_imaris_keys(dataset.keys())  # TimePoint 0, TimePoint 1, ...
    volumes = []
    #print(f"Time order: {time_keys}")

    for t_key in time_keys:
        tp = dataset[t_key]
        channel_keys = sort_imaris_keys(tp.keys())  # Channel 0, Channel 1, ...
        channels = []

        for c_key in channel_keys:
            arr = tp[c_key]["Data"][:]    # usually (z, y, x)
            channels.append(arr)

        channels = np.stack(channels, axis=0)   # (c, z, y, x)
        volumes.append(channels)

    img = np.stack(volumes, axis=0)             # (t, c, z, y, x)
    mask_layers = load_embedded_mask_layers(f, metadata["segmentation_candidates"], img.shape)

print(img.shape)


um_per_pixel = 221.91 / 512
um_per_zslice = 62.96 / 24
scale = np.array([um_per_zslice, um_per_pixel, um_per_pixel])
voxel_size_um = tuple(scale)

viewer = napari.Viewer(ndisplay=3)
viewer.add_image(img, channel_axis=1,
                 colormap=["cyan", "green",'red','blue', 'yellow', 'white'],
                 #scale=napari_scale_from_voxel_size(voxel_size_um),
                 scale=(1, *voxel_size_um),
                 opacity=1)


for mask_layer in mask_layers:
    viewer.add_labels(
        mask_layer["data"],
        name=mask_layer["name"],
        scale=scale_for_layer(mask_layer["data"], voxel_size_um),
        opacity=0.5,
    )
# viewer = napari.Viewer(ndisplay=2)
# viewer.add_image(img[:, 0, 0, :, :], opacity=1)
