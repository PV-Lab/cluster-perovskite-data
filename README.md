cluster-perovskite-data
===========

## Description

Clustering of perovskite degradation data (sample colors vs. time), identification of the cluster centroids, and plotting the XRD graphs for the cluster centroids. This repository is a part of our article:

[1] [article details come here]

## Description of Data

The data included in this repository is from our previous article: 

[2] Shijing Sun, Armi Tiihonen, Felipe Oviedo, Zhe Liu, Janak Thapa, Yicheng Zhao, Noor Titan P. Hartono, Anuj Goyal, Thomas Heumueller, Clio Batali, Alex Encinas, Jason J. Yoo, Ruipeng Li, Zekun Ren, I. Marius Peters, Christoph J. Brabec, Moungi G. Bawendi, Vladan Stevanovic, John Fisher, Tonio Buonassisi, "A data fusion approach to optimize compositional stability of halide perovskites", Matter, 2021, https://doi.org/10.1016/j.matt.2021.01.008.

The data shared inside the "Data" folder is sufficient to reproduce the methods described in article [1]. The data were obtained observing perovskite ([MA-FA-Cs]PbI3) films at MIT in 2019 and presented in article [2]. Film compositions and preparation are described in article [2]. Each film was halved and one half was exposed to X-ray diffraction (XRD) measurement and another half to an aging test. Data shared inside "Data\XRD" are XRD data of the as-grown films, which were also presented in the Matter paper referenced above, and are formatted in 2 columns (.xy files) with theta (degrees) versus intensity (a.u.). Image data ("Data\Images") were collected during an aging test, using an RGB camera within an environmental chamber (Generation 1, as described in the manuscript). Raw film color data were converted into color-calibrated data ("Data\Images\Calibrated") using 3-dimensional thin plate spline method and a reference color chart (also called "color card" or "color calibration tile"), as described in the Methods section of the manuscript.

Running the code represented in this repository as it is with the data provided reproduces Figure 3 in article [1].

## Installation

$ git clone https://github.com/PV-Lab/cluster-perovskite-data.git
$ cd cluster-perovskite-data

TO DO add env file

## Description of the Code

The codes in this repo combine to reproduce Figure 3 of article [1] from data contained within the "Data" folder.

- "Clustering_of_Camera_Image_Data.ipynb": This code applies a hierarchical clustering algorithm to camera image data, and identifies the centroids. In this dataset, three clusters are identified, with six centroids in total (one compositional centroid and one "the most typical degradation curve" centroid for each cluster); the precise number of clusters varies depending on the dataset. For each of the six centroids, the nearest film composition was identified. For each of these compositions, the XRD spectrum of the as-grown film was placed into the "Data\XRD" folder manually. NB: One film that repeatedly clusteered to its own single-film cluster was dropped from the clustering analysis as an outlier. NB: Each "camera image datum" is the entire camera time series for a given composition — capturing the film degradation in the environmental chamber as a function of time. (Films with different decay dynamics, as well as different starting and ending colors, are expected to cluster differently.)
- "Plotting_of_XRD_Patterns.ipynb": This code plots the representative (the film nearest to each cluster centroid) XRD spectra mentioned above, contained in "Data\XRD," and reproduces the plot shown in Figure 3. Note that XRD intensity (y-axis) is in arbitrary units (e.g., depends on the X-ray flux generated by the filament, and/or distance between sample and detector), and therefore an intensity normalization is performed prior to plotting ("# Scale the data" section within the code).

## Versions

- 1.0 / Jun, 2022: Latest version

## Authors
||                    |
| ------------- | ------------------------------ |
| **AUTHORS**      | Armi Tiihonen, Shijing Sun | 
| **VERSION**      | 1.0 / June, 2022 | 
| **EMAILS**      | armi.tiihonen@gmail.com | 
||                    |

## Attribution

Please, acknowledge use of this work with the appropriate citation to the repository and research article.

## Citation

    @Misc{cluster-perovskite-data2022,
      author =   {The cluster-perovskite-data authors},
      title =    {{cluster-perovskite-data}: Clustering perovskite degradation data and plotting XRD for cluster centroids},
      howpublished = {\url{https://github.com/PV-Lab/cluster-perovskite-data}},
      year = {2022}
    }
    
    [Insert details of the open HW paper]
