#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 18:38:24 2023

@author: armi
"""


import pandas as pd
import numpy as np
import numpy.matlib as nm
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import davies_bouldin_score, silhouette_score, silhouette_samples
from sklearn.cluster import KMeans
from sklearn.decomposition import KernelPCA

from cycler import cycler

from scipy.cluster.hierarchy import dendrogram, linkage, fcluster, set_link_color_palette

def load_xrd_data(data_filename = './Data/XRD/xrd_data_scaled.csv',
                  comps_filename = './Data/XRD/xrd_data_compositions.csv',
                  idx_to_be_dropped = None):

    # Read data.
    df_raw_data = pd.read_csv(data_filename)
    sample_df = pd.read_csv(comps_filename, index_col = 0)
    
    compositions = sample_df.copy() / 100
    materials = list(compositions.columns)
    data = df_raw_data.iloc[:, 1::].copy()
    
    print('Number of samples: ', sample_df.shape)
    print('Samples consist of:\n', sample_df.loc[:, materials])    
    
    # Drop samples that were discarded from the analysis (e.g. was not able to
    # read the composition from the file).
    if idx_to_be_dropped is not None:
        print('These indices contain discarded samples and are dropped from the analysis: ',
          list(idx_to_be_dropped))
        compositions.drop(index = idx_to_be_dropped, inplace = True)
        data.drop(index = idx_to_be_dropped, inplace = True)
        print('Shape of data afterward: ', data.shape)
        
    theta = data.columns.values.astype(float)
    
    print('Final shape: (n_samples, n_theta) ', data.shape)
    
    return data, compositions, theta, sample_df, df_raw_data


def plot_xrd_spectrum_expl(ax, texts=True, per=True, pbi2=True, dcspbi3=True,
    dfapbi3=True, fontsize="small", ymax=1.1):

    yp = 0 # points y
    yt = -0.09 * ymax # base y for text
    ys = -0.08 * ymax # step y between the lines of text

    if per == True:

        x = 14.1
        y = yp

        if texts == True:
            ax.text(
                x,
                yt,
                r"perovskite",
                fontsize=fontsize,
                horizontalalignment="center",
                verticalalignment="baseline",
            )

        ax.scatter([x], [y], marker="d", c="k", s=10)
        ax.plot([x, x], [y, ymax], "--k", linewidth=0.8)

    if pbi2 == True:

        x = 12.6
        y = yp

        if texts == True:
            ax.text(
                x,
                yt,
                r" PbI$_2$",
                c="darkgoldenrod",
                fontsize=fontsize,
                horizontalalignment="center",
                verticalalignment="baseline",
            )

        ax.scatter([x], [y], marker="h", c="darkgoldenrod", s=10)

    if dcspbi3 == True:

        x = [9.8, 13]
        y = [yp, yp]

        if texts == True:

            for i in range(len(x)):

                ax.text(
                    x[i],
                    yt + ys,
                    r" ${\rm \delta}$-CsPbI$_3$ (102)",
                    c="b",
                    fontsize=fontsize,
                    horizontalalignment="center",
                    verticalalignment="baseline",
                )

        ax.scatter(x, y, marker="*", c="b", s=10)

    
    if dfapbi3 == True:

        x = 11.8
        y = yp

        if texts == True:
            ax.text(
                x,
                yt + 2 * ys,
                r" ${\rm \delta}$-FAPbI$_3$",
                c="tab:brown",
                fontsize=fontsize,
                horizontalalignment="center",
                verticalalignment="baseline",
            )

        ax.scatter([x], [y], marker="h", c="tab:brown", s=10)

def load_camera_data(data_folder = './Data/Images/',
                  aging_tests = ['20190606', '20190614', '20190622', '20190711', '20190723', '20190809'],
                  n_timepoints = 1200, data_type_to_fetch = '-R1-JT/BMP/RGB/Calibrated/',
                  file_begin = 'sample_', file_end = '_cal.csv', colors = ['r', 'g', 'b'],
                  delta_t = 5, materials = ['CsPbI', 'FAPbI', 'MAPbI'], sample_filename = 'Samples.csv'):
    
    # First n timepoints will be analyzed. For this dataset, this is the first appr. 6000 min of the aging test.

    df_raw_data = [[] for j in range(len(aging_tests))]
    df_stacked_colors = []
    sample_df = []

    for i in range(len(aging_tests)):

        for j in range(len(colors)):

            file = data_folder + aging_tests[i] + data_type_to_fetch + file_begin + colors[j] + file_end
            df_raw_data[i].append(pd.read_csv(file, header = None))
            df_raw_data[i][j] = df_raw_data[i][j].iloc[:,0:n_timepoints]

        # Stack all colors one after each other for each sample.
        df_stacked_colors.append(np.hstack(df_raw_data[i]))

        sample_file = data_folder + aging_tests[i] + data_type_to_fetch + sample_filename
        sample_df.append(pd.read_csv(sample_file))
        print('Number of samples and aging tests: ', sample_df[i].shape)
        print('Samples consist of: ', sample_df[i].loc[:, materials])

        # Drop samples that were discarded from the analysis (e.g. due to bad quality film).
        idx_dropped = sample_df[i][(sample_df[i] == 'Discarded').values].index
        print('These indices contain discarded samples and are dropped from the analysis: ', list(idx_dropped))
        sample_df[i].drop(index = idx_dropped, inplace = True)
        df_stacked_colors[i] = np.delete(df_stacked_colors[i], obj = idx_dropped, axis = 0)

        print('Shape after colors stacked: ', df_stacked_colors[i].shape)

        if i == 0:
            compositions = sample_df[i].loc[:, materials] # Contains only the compositions, for clustering.
            sample_df_all = sample_df[i] # Contains all the sample data, for investigating samples.
        if i > 0:
            compositions = compositions.append(sample_df[i].loc[:, materials], ignore_index = True)
            sample_df_all = sample_df_all.append(sample_df[i], ignore_index = True)

    # Stack all samples.
    data = pd.DataFrame(np.vstack(df_stacked_colors))
    print('Final shape: (n_samples, n_timepoints_and_colors) ', data.shape)
    
    series_length = int(data.shape[1]/3)
    time = np.array(range(series_length))*delta_t/60 # Time in hours
        
    return data, compositions, time, sample_df_all, idx_dropped

def plot_rgb_timeseries_expl(ax, time, texts=True, fontsize="small"):
    
    delta = time.shape[0]/2
    color_texts = ['R', 'G', 'B']
    
    x = delta
    y = 25
    for i in range(3):
        
        ax.text(
        x,
        y,
        color_texts[i],
        fontsize=fontsize,
        horizontalalignment="center",
        verticalalignment="baseline")
        
        x = x + 2*delta

def compute_tetr_coord(compositions):
    
    # Tetragonal coordinates for the plots.
    xy = np.zeros((compositions.shape[0],2))
    xy[:,1] = compositions.iloc[:,2]*0.8660
    xy[:,0] = np.add(compositions.iloc[:,0], compositions.iloc[:,2]*0.5)
    
    return xy

def plot_specs(compositions):
    
    # Tetragonal coordinates for the plots.
    xy = compute_tetr_coord(compositions)
    
    # Color cycle for the plots.
    cluster_colors = ['darkturquoise', 'darkorchid', 'coral', 'magenta',
                      'r', 'g', 'b', 'cyan']
    matplotlib.rcParams['axes.prop_cycle'] = cycler(color= ['k'] + cluster_colors)
    
    set_link_color_palette(list(cluster_colors))
    
    return xy, cluster_colors

def plot_cluster_triangle(sample_xy, n_clusters, L, cluster_colors,
                          triangle_side_labels = ['Cs (%)', 'FA (%)', 'MA (%)'],
                          data_type = 'XRD'):
    
    plt.plot([0, 1, 0.5, 0], [0, 0, np.sqrt(0.75), 0], "k")

    plt.text(0.45, -0.05, triangle_side_labels[0])
    plt.text(0.14, 0.35, triangle_side_labels[1], rotation=55)
    plt.text(0.74, 0.34, triangle_side_labels[2], rotation=-55)
    plt.text(-0.0, -0.05, "0")
    plt.text(0.95, -0.05, "100")
    plt.text(-0.06, 0.01, "100", rotation=55)
    plt.text(0.42, 0.84, "0", rotation=55)
    plt.text(0.97, 0.03, "0", rotation=-55)
    plt.text(0.5, 0.8, "100", rotation=-55)
    
    for i in range(n_clusters):
        plt.plot(sample_xy[L==i,0],sample_xy[L==i,1], '.',
                 c = cluster_colors[i], label = data_type + ' cluster ' + str(i))
    plt.legend()
    plt.gcf().set_size_inches(3.5, 2.9)
    plt.tight_layout()
    plt.axis('off')

def plot_clusters_into_triangles(sample_xy, n_clusters, L, cluster_colors,
                                triangle_side_labels = ['Cs (%)', 'FA (%)', 'MA (%)'],
                                to_single_plot = True, show = True,
                                data_type = 'XRD'):
    
    if to_single_plot == True:
        
        if show == True:
            
            plt.figure()
        
        plot_cluster_triangle(sample_xy, n_clusters, L, cluster_colors,
                                  triangle_side_labels = triangle_side_labels,
                                  data_type = data_type)
        if show == True:
            
            plt.show()
        
    else:
        
        for i in range(n_clusters):
            
            L_temp = L[L==i]
            
            if len(L_temp) > 0:
                
                xy = sample_xy[L == i]
                
                if show == True:
                    
                    plt.figure()
                
                plot_cluster_triangle(xy, n_clusters, L_temp, cluster_colors,
                                          triangle_side_labels = triangle_side_labels,
                                          data_type = data_type)
                if show == True:
                    
                    plt.show()
                    
def highlight_samples_in_cluster_triangle(xy_cleaned, k, cluster_colors,
                                          cluster_rep, cluster_rep_marker = '*',
                                          cluster_rep_label = 'Centroid C',
                                          savefig = False, filename = 'triangle'):
    # Plot the most typical sample, cluster centroid.
    
    for i in range(k):
        plt.plot(xy_cleaned[cluster_rep[i],0], xy_cleaned[cluster_rep[i],1],
                 cluster_rep_marker, c = cluster_colors[i],
                 label = cluster_rep_label + str(i), markersize=8)
    
    plt.legend(bbox_to_anchor=(0, -0.3/2 - 0.1, 1, 0.3), 
               loc='lower left', ncol=2, borderaxespad=0,
               labelspacing=0, handletextpad=0.3, mode='expand')
    
    if savefig == True:
        plt.savefig(filename + '.png')
        plt.savefig(filename + '.svg')
        plt.savefig(filename + '.pdf')

def plot_hierarchical_clustering_dendrogram(Z, save_fig = False, 
                                            filename = 'hierarchical_clustering_dendrogram',
                                            sample_labels = True):
    
    if sample_labels == True:
        no_labels = False
    else:
        no_labels = True
    
    plt.figure()
    plt.clf()
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Sample')
    plt.ylabel('Cosine distance')
    dendrogram(Z, leaf_rotation=90., leaf_font_size=6., no_labels = no_labels)
    # Use this if we you want to see the samples in detail.
    #plt.gcf().set_size_inches(25,25)
    plt.tight_layout()
    if save_fig == True:
        plt.savefig(filename + '.png')
        plt.savefig(filename + '.pdf')
    plt.show()
    
    
def drop_data(data, xy, sample_df_all, compositions, to_be_dropped):
    
    cleaned_data = data.copy().drop(index = to_be_dropped).reset_index(drop = True)#np.delete(data, to_be_dropped, axis=0)
    xy_cleaned = np.delete(xy, to_be_dropped, axis=0)
    sample_df_cleaned = sample_df_all.copy().drop(index=to_be_dropped).reset_index(drop = True)
    compositions_cleaned = compositions.copy().drop(index=to_be_dropped).reset_index(drop = True)
    
    return cleaned_data, xy_cleaned, sample_df_cleaned, compositions_cleaned
    
def compute_cluster_mean_std(cleaned_data, L, k):
    
    mean = []
    std = []
    
    for i in range(k):
        
        mean.extend([np.average(cleaned_data[L==i], axis=0)])
        std.extend([np.std(cleaned_data[L==i], axis=0)])
        #n_points = mean[i].shape[0]
        
    return mean, std

def compute_cluster_compositional_mean(cleaned_data, L, k,
                                       compositions_cleaned):
    
    mean_compos = []
    
    for i in range(k):
        
        mean_compos.extend([np.average(compositions_cleaned.iloc[L==i,:], axis=0)])
        
    return mean_compos
    
def plot_cluster_mean_spectra(x_points, mean, std, k, cluster_colors,
                         data_type = 'XRD', save_fig = False,
                         filename = 'cluster_spectra',
                         continue_plotting = False):
    
    if data_type == 'RGB':
        
        x = range(x_points.shape[0]*3) # R, G, B, channels stacked so these are joint time-color points
    
    else:
        
        x = x_points
        
    plt.figure()
    
    for i in range(k):
        
        plt.plot(x, mean[i], '-', c = cluster_colors[i],
                 label = data_type + ' cluster ' + str(i) + ' mean')
        plt.fill_between(x, mean[i] - std[i], mean[i] + std[i],
                         color = cluster_colors[i], alpha=0.1, label = 'St. dev.')
        
    plt.legend()
    
    if data_type == 'XRD':
        plot_xrd_spectrum_expl(plt.gca(), texts=True, per=True, pbi2=True, dcspbi3=True,
            dfapbi3=True, fontsize="small", ymax=1.1)
        plt.xlim((8,15))
        plt.ylim((-0.2, 1.1))
        plt.xlabel('2$\Theta$ ($\degree$)')
        plt.ylabel('Intensity (a.u.)')
        
    elif data_type == 'RGB':
        plot_rgb_timeseries_expl(plt.gca(), x_points, texts=True, fontsize="small")
        plt.ylim((0,255))
        plt.xlabel('Timepoints and RGB colors stacked')
        plt.ylabel('RGB pixel value')
        
    if save_fig == True:
        plt.savefig(filename + '.png')
        plt.savefig(filename + '.pdf')
        
    if continue_plotting == False:
        plt.show()
        
def plot_spectra(x_points, cleaned_data, idx_to_plot, id_strs, colors,
                 new_figure = True, data_type = 'XRD', show = True,
                 show_xrd_expl = True):

    if data_type == 'RGB':
        
        x = range(x_points.shape[0]*3) # R, G, B, channels stacked so these are joint time-color points
    
    else:
        
        x = x_points
        
    if new_figure == True:
        plt.figure()
    
    for i in range(len(idx_to_plot)):
        
        # TO DO then remove for both compos and most typical
        #label = 'C' + str(i) + ': ' + id_strs[i]
                  
        plt.plot(x, cleaned_data.iloc[idx_to_plot[i],:].values, '-',
                 c = colors[i], linewidth=1, label = id_strs[i])
    
    plt.legend(bbox_to_anchor=(-0.15, -0.55, 1.15, 1), 
               loc='lower left', ncol=1, borderaxespad=0,
               labelspacing=0, handletextpad=0.3, mode='expand')
    
    if (data_type == 'XRD') or (data_type == 'XRD raw'):
        
        plt.xlim((8,15))
        
        if data_type == 'XRD':
            ymax = 1.1
            plt.ylabel('Intensity (a.u.)')
            
        else:
            ymax = np.max(np.max(cleaned_data, axis = 0), axis = 0)
            plt.ylabel('Intensity (counts)')
            
        plt.ylim((-0.2, ymax))
        plt.xlabel('2$\Theta$ ($\degree$)')
        if show_xrd_expl is True:
            plot_xrd_spectrum_expl(plt.gca(), texts=True, per=True, pbi2=True, dcspbi3=True,
            dfapbi3=False, fontsize="small", ymax=ymax)
        
        
    elif data_type == 'RGB':
        
        plt.ylim((0,255))
        plt.xlabel('Timepoints and RGB colors stacked')
        plt.ylabel('RGB pixel value')
        
    plt.gcf().set_size_inches(3.5,4)
    plt.tight_layout()
    
    if show == True:
        
        plt.show()

def create_csmafapbi_compos_str(compositions_cleaned, loc_idx):
    
    cs = f"{compositions_cleaned.loc[loc_idx,'CsPbI']:.2f}"
    fa = f"{compositions_cleaned.loc[loc_idx,'FAPbI']:.2f}"
    ma = f"{compositions_cleaned.loc[loc_idx,'MAPbI']:.2f}"
    compos_str = 'Cs$_{' + cs + '}$FA$_{' + fa + '}$MA$_{' + ma + '}$PbI$_3$'
    
    return compos_str    

def extract_most_typical_sample(cleaned_data, mean, k, compositions_cleaned,
                                metric = 'cosine'):
    
    # Pick the nearest actual spectrum to the cluster center - "the most
    # typical sample".
    
    cluster_rep = []
    compos_str_cluster_rep = []
    
    for i in range(k):
        
        d = pairwise_distances(np.concatenate((
            np.reshape(mean[i], (1,mean[i].shape[0])), cleaned_data), axis=0), 
            metric = metric)
    
        # Distance matrix is symmetric and the diagonal is the distance to
        # sample itself (zero). Now, the first column is the cluster mean and
        # we need the distance to cluster mean so the first row only without
        # the first element is considered.
        cluster_rep.append(np.argmin(d[0,1:], axis = 0))
        
        compos_str_cluster_rep.append(
            create_csmafapbi_compos_str(compositions_cleaned, cluster_rep[i]))
                
        print('Distance to the cluster mean value (XRD spectrum or degradation curve) i.e. finding the most typical sample in the cluster:')
        print('Minimum distance in cluster ', i, ': ', np.min(d[0,1:]),
              ', maximum distance: ', np.max(d))
        print('Index of the minimum distance sample: ', cluster_rep[i])
        print('Details of the min. distance sample:\n',
              compositions_cleaned.iloc[cluster_rep[i],:], '\n',
              compos_str_cluster_rep[i], '\n')
        
    return cluster_rep, compos_str_cluster_rep, d

def extract_compos_centroid_sample(cleaned_data, mean_compos, k, compositions_cleaned,
                                metric = 'euclidean'):
    
    cluster_rep_compos = []
    compos_str_cluster_rep_compos = []
    
    for i in range(k):
        
        d_compos = pairwise_distances(np.concatenate((np.reshape(
            mean_compos[i], (1,mean_compos[i].shape[0])),
            compositions_cleaned), axis=0), metric = metric)
    
        # Distance matrix is symmetric and the diagonal is the distance to
        # sample itself (zero). Now, the first column is the cluster mean and
        # we need the distance to cluster mean so the first row only without
        # the first element is considered.
        cluster_rep_compos.append(np.argmin(d_compos[0,1:],axis = 0))
        
        compos_str_cluster_rep_compos.append(
            create_csmafapbi_compos_str(compositions_cleaned, cluster_rep_compos[i]))
                
        print('Distance to the cluster compositional center i.e. finding the compositional centroid sample of each cluster:')
        print('Minimum distance in cluster ', i, ': ', np.min(d_compos[0,1:]),
              ', maximum distance: ', np.max(d_compos))
        print('index of minimum distance sample: ', cluster_rep_compos[i])
        print('Details of the min. distance sample:\n',
              compositions_cleaned.iloc[cluster_rep_compos[i],:], '\n',
              compos_str_cluster_rep_compos[i], '\n')
        
    return cluster_rep_compos, compos_str_cluster_rep_compos, d_compos

def score_hier_numbers_of_clusters(cleaned_data, xy_cleaned, cluster_colors,
                                   score_metric = 'cosine', 
                                   max_n_clusters = 10, save_fig = True, 
                                   filename = 'Scores'):
    
    # Test robustness of the number of clusters within hierarchical clustering.
    
    # Scores with different number of clusters.
    Z_t = []
    L_t = []
    
    cluster_range = range(2,max_n_clusters)
    for m in cluster_range:
        
        Z_temp = linkage(cleaned_data, 'average', metric = 'cosine')
        L_temp = fcluster(Z_temp, m, criterion='maxclust')
        
        if (np.max(L_temp) == m) and (np.min(L_temp) == 1):
            
            # All the plotting codes here assume cluster ids start from zero.
            L_temp = L_temp - 1
            
        Z_t.append(Z_temp)
        L_t.append(L_temp)
        
    
    score_numbers_of_clusters(cleaned_data, xy_cleaned, L_t, max_n_clusters,
                              cluster_colors, score_metric = score_metric,
                              title = 'hierarchical', save_fig = save_fig, 
                              filename = filename)
    
def score_kmeans_numbers_of_clusters(cleaned_data, xy_cleaned, cluster_colors,
                                   score_metric = 'cosine', 
                                   max_n_clusters = 10, save_fig = True, filename = 'Scores'):
    
    # Test robustness of the number of clusters within hierarchical clustering.
    
    # Scores with different number of clusters.
    km_t = []
    L_t = []
    
    cluster_range = range(2,max_n_clusters)
    for m in cluster_range:
        
        # PCA with cosine kernel is done because the euclidean distance
        # with k-means algorithm does not work well with XRD or RGB data (for
        # which the angle or point in time, respectively, is more important
        # basis for clustering than the actual amplitude.)
        pca = KernelPCA(n_components = cleaned_data.shape[0], kernel='cosine'
                        ).fit_transform(cleaned_data)
        km_temp = KMeans(n_clusters=m).fit(pca)
        L_temp = km_temp.labels_
        
        if (np.max(L_temp) == m) and (np.min(L_temp) == 1):
            
            # All the plotting codes here assume cluster ids start from zero.
            L_temp = L_temp - 1
        
        km_t.append(km_temp)
        L_t.append(L_temp)
        
    score_numbers_of_clusters(cleaned_data, xy_cleaned, L_t, max_n_clusters,
                              cluster_colors, score_metric = score_metric,
                              title = 'k-means with cosine kernel PCA',
                              save_fig = save_fig, filename = filename)

def score_numbers_of_clusters(cleaned_data, xy_cleaned, L_t, max_n_clusters,
                              cluster_colors, score_metric = 'cosine',
                              title = '', save_fig = True, filename = 'Scores'):
    
    
    # Davies-Bouldin scores
    dbs = []
    # Average silhouette scores
    ss = []
    # Individual silhouette score plots implemented as scikit-learn tutorial
    # suggests.
    
    # Scores with different number of clusters.
    for m in range(2,max_n_clusters):
        
        dbs.append(davies_bouldin_score(xy_cleaned, L_t[m-2]))
        ss.append(silhouette_score(cleaned_data, L_t[m-2], 
                                   metric=score_metric))
        
        if m < 5:
            fig, (ax1, ax2) = plt.subplots(1, 2)
            
            plt.axes(ax1)
            plot_clusters_into_triangles(xy_cleaned, n_clusters=m, 
                                         L=L_t[m-2],
                                         cluster_colors = cluster_colors,
                                         data_type = 'XRD',
                                         show = False)
            
            # Compute the silhouette scores for each sample
            sss = silhouette_samples(cleaned_data, L_t[m-2], metric=score_metric)
            
            plt.axes(ax2)
            y_lower = 10
            
            for i in range(m):
                
                ith_cluster_silhouette_values = sss[L_t[m-2] == i]
                ith_cluster_silhouette_values.sort()
                size_cluster_i = ith_cluster_silhouette_values.shape[0]
                
                y_upper = y_lower + size_cluster_i
        
                color = cluster_colors[i]
                plt.fill_betweenx(
                    np.arange(y_lower, y_upper),
                    0,
                    ith_cluster_silhouette_values,
                    facecolor=color,
                    edgecolor=color,
                    alpha=0.7,
                )
        
                # Label the silhouette plots with their cluster numbers at the middle
                ax2.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        
                # Compute the new y_lower for next plot
                y_lower = y_upper + 10  # 10 for the 0 samples
    
            plt.title('Silhouette plot for ' + str(m) + ' ' + title + 
                          ' clusters')
            ax2.set_xlabel("The silhouette coefficient values")
            ax2.set_ylabel("Cluster label")
            
            # The vertical line for average silhouette score of all the values
            ax2.axvline(x=ss[-1], color="red", linestyle="--")
        
            ax2.set_yticks([])  # Clear the yaxis labels / ticks
            ax2.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
            
            plt.gcf().set_size_inches(6.5, 2.9)
            plt.tight_layout()
            
            if save_fig == True:
                plt.savefig(filename + 'silhouette_' + str(m) + 'clusters.png')
                plt.savefig(filename + 'silhouette_' + str(m) + 'clusters.pdf')
            plt.show()
    
    plt.figure()
    plt.plot(range(2,max_n_clusters), dbs, '.-')
    plt.ylabel('Davies-Bouldin score (lower is better)')
    plt.xlabel('Number of ' + title + ' clusters')
    plt.gcf().set_size_inches(3.5, 2.9)
    plt.tight_layout()
    if save_fig == True:
        plt.savefig(filename + '_db.png')
        plt.savefig(filename + '_db.pdf')
    
    plt.show()
    
    plt.figure()
    plt.plot(range(2,max_n_clusters), ss, '.-')
    plt.ylabel('Silhouette score (higher is better)')
    plt.xlabel('Number of ' + title + ' clusters')
    plt.gcf().set_size_inches(3.5, 2.9)
    plt.tight_layout()
    if save_fig == True:
        plt.savefig(filename + 'ave_silhouette.png')
        plt.savefig(filename + 'ave_silhouette.pdf')
    plt.show()
