#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 16:13:52 2023

@author: armi
"""

# Plot RGB degradation curves of the XRD centroid # compositions.

# First, run clustering_rgb.py.

# Then, run this file.

xrd_idx = [24, 108, 82]

rgb_timeseries_plot_for_paper(cleaned_data, k, time, mean, std, xrd_idx,
                                  cluster_colors,
                                  save_fig = True,
                                  filename = 'rgb_curves_of_xrd_centroids')
