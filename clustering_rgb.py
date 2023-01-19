#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 11:12:47 2022

@author: armi
"""

from scipy.cluster.hierarchy import linkage, fcluster
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from clustering_functions import load_camera_data, plot_specs, plot_hierarchical_clustering_dendrogram, drop_data, score_hier_numbers_of_clusters, score_kmeans_numbers_of_clusters, plot_clusters_into_triangles, compute_cluster_mean_std, plot_cluster_mean_spectra, extract_most_typical_sample,highlight_samples_in_cluster_triangle,plot_spectra, compute_cluster_compositional_mean, extract_compos_centroid_sample, create_csmafapbi_compos_str

def rgb_timeseries_plot_for_paper(cleaned_data, k, time, mean, std, cluster_rep,
                                  cluster_colors, save_fig = True,
                                  filename = 'rgb_timeseries'):
    
    series_length = int(cleaned_data.shape[1]/3)
    
    # Data sliced into color channels.
    rgb_sl = []
    mean_sl = []
    std_sl = []

    for i in range(3): # R, G, B
        
        # Color data for this channel.
        rgb_ch = cleaned_data.iloc[:, i*series_length : (i+1)*series_length]
        rgb_sl.append(rgb_ch)
        
        mean_sl.append([])
        std_sl.append([])
        
        for j in range(k): # Cluster
            
            # Mean and st.dev data for this channel and cluster.
            mean_ch = mean[j][i*series_length : (i+1)*series_length]
            std_ch = std[j][i*series_length : (i+1)*series_length]
            
            mean_sl[-1].append(mean_ch)
            std_sl[-1].append(std_ch)
            
    for channel in range(3):
        
        plt.figure()
        legend_content = []
        
        for i in range(k):
            
            plt.plot(time, mean_sl[channel][i], '-', c = cluster_colors[i])
            legend_content.extend(['Cluster ' + str(i) + ' mean'])

        for i in range(k):
            plt.fill_between(time, mean_sl[channel][i] - std_sl[channel][i],
                             mean_sl[channel][i] + std_sl[channel][i],
                             color = cluster_colors[i], alpha=0.2, edgecolor = None)
            # Printing only every 10th point makes the plot look nicer.
            plt.plot(time[::10],
                     rgb_sl[channel].iloc[cluster_rep[i],::10],
                            linestyle=(0,(3,5)), c = cluster_colors[i], linewidth=1)
            if i == 0:
                legend_content.extend(['St. dev.', 'RGB centroid'])

        plt.ylim((30,150))
        plt.xlabel('Aging time (hours)')
        plt.ylabel('Color in RGB (px)')
        if channel == 0:
            color = 'R'
        elif channel == 1:
            color = 'G'
        elif channel == 2:
            color = 'B'
        plt.text(48, 140, color)

        fig = plt.gcf()
        fig.set_size_inches(1.7, 2.9)
        plt.tight_layout()

        if save_fig == True:
            plt.savefig(filename + color + '.png')
            plt.savefig(filename + color + '.svg')
            plt.savefig(filename + color + '.pdf')

        plt.show()

###############################################################################

if __name__ == "__main__":
    
    # LOAD DATA 
    ##############################
    
    # Load RGB data with a data loader specified for the dataset under
    # investigation.
    data, compositions, time, sample_df, idx_dropped = load_camera_data(
        n_timepoints = 1200)

    # Set cluster colors for the plots and calculate the xy coordinates of each
    # composition in the tetragonal plots. Works only for tetragonal materials
    # spaces.
    xy, cluster_colors = plot_specs(compositions)
    # Let's swap colors 0 and 2 to be compatible with the colors in XRD data.
    cluster_colors[0], cluster_colors[2] = cluster_colors[2], cluster_colors[0]
     
    # HIERARCHICAL CLUSTERING ANALYSIS (BASIC)
    ##############################
    
    # Cluster the samples with RGB curves only (no composition information).
    
    # Create linkages for the hierarchical clustering dendrogram. Cosine metric
    # is used because the location of the signal (time point) is a better base
    # of clustering than the signal amplitude.
    Z = linkage(data, 'average', metric = 'cosine')
    
    plot_hierarchical_clustering_dendrogram(Z, save_fig = False, 
                                            sample_labels = True)
    
    # Set the number of clusters for to four as the dendrogram suggests there
    # would be four clusters.
    k = 4
    
    # Divide the data into k clusters.
    L = fcluster(Z, k, criterion='maxclust')
    L = L - 1 # Re-index because all the plot functions implemented here assume cluster numbering starts from 0.
    
    
    # HIERARCHICAL CLUSTERING ANALYSIS (OUTLIERS)
    ##############################
    
    # Are there individual samples that would be clustered into their own
    # cluster? Or very small clusters? These may be outliers and should be
    # dropped if they look like that after detailed analysis.
    
    print(np.where(L==3))
    print(sample_df.iloc[np.where(L==3)[0], :])

    # The above sample (145) always clusters to its own and turns out it is
    # one of the reference samples containing Br, unlike the other samples.
    # Br is also not in the target compositional region of this analysis.
    # Let's treat this composition as an outlier.
    
    to_be_dropped = np.where(sample_df == 'With Br')[0]
    cleaned_data, xy_cleaned, sample_df_cleaned, compositions_cleaned = drop_data(
        data, xy, sample_df, compositions, to_be_dropped)
    
    # Let's repeat the hierarchical clustering.
    Z = linkage(cleaned_data, 'average', metric = 'cosine')
    
    plot_hierarchical_clustering_dendrogram(Z, save_fig = False, 
                                            sample_labels = True)
    
    # CLUSTERING ANALYSIS (NUMBER OF CLUSTERS)
    ##############################
    
    # Dendrogram plot now suggests there is 3 clusters in the data.
    
    # Let's test a range of number of clusters to confirm.
    # Score metrics:
    # - Average silhouette score (with cosine metric): value close to 1 means very well defined clusters, -1 failed clustering, 0 overlapping clusters
    # - Average Davies - Bouldin score (with Euclidean metric): value 0 is the best possible separation between the clusters, smaller value is better
    # - Sample-by-sample silhouette score (with cosine metric): negative score of an individual sample means it has likely been clustered into the wrong cluster, generally, all the clusters should have most of the samples with at least as high silhouette score as the average value (dashed line)
    score_hier_numbers_of_clusters(cleaned_data, xy_cleaned, cluster_colors,
                                       score_metric = 'cosine', 
                                       max_n_clusters = 10, 
                                       save_fig = True,
                                       filename = 'rgb_score_hier')
    
    # The graphs suggest three is indeed the right number of clusters for this
    # dataset. Note that DB score uses Euclidean metric it is not as good
    # score metric for this data as the silhouette score, so where DB and
    # silhouette score suggest different number of clusters, silhouette is more
    # reliable.
    
    # For robustness, let's repeat the analysis also for k-means clustering
    # algorithm. Note that the input data for k-means is in this implementation
    # transformed with cosine kernel principal component analysis algorithm.
    # This is because k-means uses Euclidean metric by default and we want to
    # give weight on cosine metric, instead.
    score_kmeans_numbers_of_clusters(cleaned_data, xy_cleaned, cluster_colors,
                                       score_metric = 'cosine', 
                                       max_n_clusters = 10,
                                       save_fig = True,
                                       filename = 'rgb_score_kmeans')
    
    # The same result!
    
    # Set the number of clusters for the rest of the analysis to three.
    k = 3
    
    # HIERARCHICAL CLUSTERING ANALYSIS (FINAL CLUSTERING)
    ##############################
    
    Z = linkage(cleaned_data, 'average', metric = 'cosine')
    plot_hierarchical_clustering_dendrogram(Z, save_fig = True, 
                                                filename = 'rgb_hierarchical_clustering_dendrogram',
                                                sample_labels = False)
    
    # Divide the data into k clusters.
    L = fcluster(Z, k, criterion='maxclust')
    L = L - 1 # Re-index because all the plot functions implemented here assume cluster numbering starts from 0.
    
    plot_clusters_into_triangles(xy_cleaned, n_clusters = k, L=L,
                                 cluster_colors=cluster_colors,
                                 triangle_side_labels = ['Cs (%)', 'FA (%)', 'MA (%)'],
                                 to_single_plot = True)
    
    
    # INVESTIGATE MOST TYPICAL RGB SPECTRA WITHIN EACH CLUSTER
    ##############################
    
    # Plot arithmetic mean (cluster center) and st.dev of the spectra in each
    # cluster.
    
    mean, std = compute_cluster_mean_std(cleaned_data, L, k)
    
    plot_cluster_mean_spectra(time, mean, std, k, cluster_colors,
                             data_type = 'RGB', save_fig = True,
                             filename = 'xrd_cluster_mean')
    
    # The clusters indeed have differing signature spectra!
    
    # Extract the real samples that are the most typical to each cluster.
    cluster_rep, compos_str_cluster_rep, d = extract_most_typical_sample(
        cleaned_data, mean, k, compositions_cleaned, metric = 'euclidean')
    
    print('Aging test information for the most typical samples:\n',
          sample_df_cleaned.iloc[cluster_rep,:])
    
    # Plot them in the triangle.
    plt.figure()
    plot_clusters_into_triangles(xy_cleaned, k, L, cluster_colors,
                                    show = False)
    highlight_samples_in_cluster_triangle(xy_cleaned, k, cluster_colors,
                                              cluster_rep, cluster_rep_marker = '*',
                                              cluster_rep_label = 'Centroid C',
                                              savefig = True, filename = 'rgb_triangle')
    plt.show()
    
    # Plot their degradation curves.
    plot_spectra(time, cleaned_data, cluster_rep, 
                 ['C : ' + i for i in compos_str_cluster_rep],
                 cluster_colors, data_type = 'RGB')
    
    # INVESTIGATE COMPOSITIONAL CENTROIDS OF EACH CLUSTER
    ##############################
    
    mean_compos = compute_cluster_compositional_mean(cleaned_data, L, k,
                                           compositions_cleaned)
    
    # Extract the real sample nearest to the center.
    cluster_rep_compos, compos_str_cluster_rep_compos, d_compos = extract_compos_centroid_sample(
        cleaned_data, mean_compos, k, compositions_cleaned, metric = 'euclidean')
    
    # Plot them in the triangle.
    plt.figure()
    plot_clusters_into_triangles(xy_cleaned, k, L, cluster_colors,
                                    show = False)
    highlight_samples_in_cluster_triangle(xy_cleaned, k, cluster_colors,
                                          cluster_rep_compos,
                                          cluster_rep_marker = 'x',
                                          cluster_rep_label = 'Compos. centroid C')
    plt.show()
    
    # Plot their spectra.
    plot_spectra(time, cleaned_data, cluster_rep_compos, 
                 ['C :' + i for i in compos_str_cluster_rep_compos],
                 cluster_colors, data_type = 'RGB')
    
    # FINAL PLOT
    ##############################
    
    rgb_timeseries_plot_for_paper(cleaned_data, k, time, mean, std, cluster_rep,
                                      cluster_colors)
    
    # OPTIONAL: PLOT SUPPLEMENTARY FIGURES
    
    # Plot RGB degradation curves of the XRD centroid compositions.
    
    '''
    xrd_idx = [24, 108, 82]
    
    rgb_timeseries_plot_for_paper(cleaned_data, k, time, mean, std, xrd_idx,
                                  cluster_colors,
                                  save_fig = True,
                                  filename = 'rgb_curves_of_xrd_centroids')
    '''
