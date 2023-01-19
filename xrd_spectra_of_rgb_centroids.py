#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 16:13:52 2023

@author: armi
"""

# Plot normalized XRD spectra (raw data, not resampled) of the RGB centroid
# compositions.

# First, run clustering_xrd.py.

# Then, run this file.

rgb_idx = [23, 29, 72]

plt.figure()
for i in range(len(rgb_idx)):
    intmax = np.max(r_data_intensities.iloc[rgb_idx[i],:])
    
    plot_spectra(r_data_thetas.iloc[rgb_idx[i],:],
             r_data_intensities.iloc[[rgb_idx[i]],:] /intmax,
             [0], ['RGB centroid: ' + create_csmafapbi_compos_str(
                 compositions, rgb_idx[i])],
             [cluster_colors[i]], new_figure = False, data_type = 'XRD',
             show = False)

plt.savefig('xrd_spectra_of_rgb_centroids.png')
plt.savefig('xrd_spectra_of_rgb_centroids.pdf')

plt.show()