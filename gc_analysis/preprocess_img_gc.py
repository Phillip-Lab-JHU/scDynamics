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
"""Histogram matching and rescale intensity through time for GC dynamics"""

import os
import numpy as np
from skimage import io
import napari
import skimage
import matplotlib.pyplot as plt
from tqdm import tqdm
import tifffile


# path = r'\\philliplab-server.wse.jhu.edu\data\Chanhong\Cornell LN Spleen\Raw images\Exp3\\'
# files = next(os.walk(path))[2]
# bool_list = ['tif' in ele for ele in files]
# files = np.array(files)[np.array(bool_list)]
# tif = tifffile.TiffFile(path + files[6])  # Image(np.array) + Metadata
# ref_img = tif.asarray()  # Image(np.array)


#um_per_pixel = 230.9/320
um_per_pixel = 230.9/256
um_per_zsice = 3
resolution = (1./um_per_pixel, 1./um_per_pixel)

############################### Visualize raw image ###############################
path = r'\\philliplab-server.wse.jhu.edu\data\Chanhong\Cornell LN Spleen\Raw images\IgG\norm_img\\'
files = next(os.walk(path))[2]
bool_list = ['tif' in ele for ele in files]
files = np.array(files)[np.array(bool_list)]

tif = tifffile.TiffFile(path+files[4])
img = tif.asarray()  # Image(np.array)

viewer = napari.Viewer(ndisplay=3)
viewer.add_image(img,  scale=np.array([um_per_zsice, um_per_pixel, um_per_pixel]), channel_axis=2,
                 colormap=["blue", "green",'red','gray'], opacity=1)


############################### Normalize raw image ###############################
path = r'\\philliplab-server.wse.jhu.edu\data\Chanhong\Cornell LN Spleen\Raw images\IgG\\'
files = next(os.walk(path))[2]
bool_list = ['tif' in ele for ele in files]
files = np.array(files)[np.array(bool_list)]

for file in files[-3:]:

    tif = tifffile.TiffFile(path+file)  # Image(np.array) + Metadata
    img = tif.asarray()  # Image(np.array)

    metadata = tif.imagej_metadata   # Dict that has {'channels':, 'slices':, 'frames':}
    metadata['unit'] = 'um'
    metadata['spacing'] = um_per_zsice
    metadata['axes'] = tif.series[0].axes  # 'TZCYX'


    # metadata = {}
    # for tag in tif.pages[0].tags:
    #     tag_name, tag_value = tag.name, tag.value
    #     metadata[tag_name] = tag_value
    #     print(tag_name, tag_value)


    print("shape: {}".format(img.shape)) # (t, z, channel, y, x)
    print("dtype: {}".format(img.dtype))
    print("range: ({}, {})".format(np.min(img), np.max(img)))


    norm_img = np.empty(shape=img.shape, dtype=img.dtype)



    for t in tqdm(range(img.shape[0])):
        for c in range(img.shape[2]):
            zstack = img[t, :, c, :, :]  # (t, z, channel, y, x)

            match_zstack = skimage.exposure.match_histograms(image=zstack, reference=img[0,:,c,:,:],
                                                            channel_axis=None)
            # match_zstack = skimage.exposure.match_histograms(image=zstack, reference=ref_img[0, :, c, :, :],
            #                                                 channel_axis=None)

            # norm_zstack = (match_zstack - np.min(match_zstack)) / np.max(match_zstack) * 65535
            # norm_zstack = norm_zstack.astype(np.float64)
            # norm_zstack = norm_zstack / 65535.0

            vmin, vmax = np.quantile(img[0, :, c, :, :], q=(0.001, 0.999))
            norm_zstack = skimage.exposure.rescale_intensity(match_zstack, in_range=(vmin, vmax), out_range=np.uint16)

            #equalize_zstack = skimage.exposure.equalize_adapthist(norm_zstack)  # Too noisy background
            #equalize_zstack = skimage.exposure.equalize_hist(norm_zstack)  # Too bright on highly concentrated region

            norm_img[t, :, c, :, :] = norm_zstack
            # if c == 3:
            #     #print(np.max(match_zstack))
            #     # fig, ax = plt.subplots()
            #     # ax.hist(zstack.ravel(), bins=512)
            #     # plt.xlim(0, np.max(img[0,:,:,:,3]))
            #     # plt.savefig(path + 'hists/%s.png' % t)
            #
            #     fig, ax = plt.subplots()
            #     ax.hist(norm_zstack.ravel(), bins=512)
            #     plt.xlim(0, 1)
            #     plt.savefig(path + 'norm_hists/%s.png' % t)
            #     fig, ax = plt.subplots()
            #     ax.hist(norm_zstack.ravel(), bins=512)
            #     plt.xlim(0, 65535)
            #     plt.savefig(path + 'match_norm/%s.png' % t)
            #     plt.close()
            #     plt.clf()

    if not os.path.isdir(path + 'norm_img/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
        os.makedirs(path + 'norm_img/')
    #io.imsave(path+'norm_img/'+file, norm_img)
    tifffile.imwrite(path+ 'norm_img/' + file, norm_img, resolution=resolution, imagej=True, metadata=metadata)

# path = r'\\philliplab-server.wse.jhu.edu\data\Chanhong\Cornell LN Spleen\Raw images\Exp1\\'
# files = next(os.walk(path))[2]
#


# tif = tifffile.TiffFile(path+'norm_img/20240106-Good-D10-B1L-ZT1-10-97-fov230-256px-CD40L.tif')  # Image(np.array) + Metadata
# img1 = tif.asarray()
#
# tif = tifffile.TiffFile(path+'norm_img/aa20240106-Good-D10-B1L-ZT1-10-97-fov230-256px-CD40L.tif')  # Image(np.array) + Metadata
# img2 = tif.asarray()
#
# viewer = napari.view_image(img1, name=["WT_GCB", "MT_GCB", 'Tfh', 'FDC'], channel_axis=2,
#                            colormap=["blue", "green", 'red', 'gray'], scale=np.array([3.000, 0.898, 0.898]))
# napari.run()