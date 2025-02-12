This repo contains codes for our paper on Optimal-Transport based conformal prediction, available at https://arxiv.org/abs/2501.18991

# Requirements

Experiments in classification require the package arc for the ARS method, available at: https://github.com/msesia/arc 
File : arc 

Experiments in regression involve the local ellipsoid method, available at:  https://github.com/M-Soundouss/EllipsoidalConformalMTR/tree/main
File : ellipsoidal_conformal_utilities.py

functions.py contains our main functions

# Classification

Classif_SimuData.ipynb contains experiments for Classification for simulated data in the main body of the paper.

Classif_OTCP.ipynb contains experiments for Classification for MNIST and Fashion-MNIST for K=10 labels 

Classif_OTCP2.ipynb contains experiments for Classification in the supplementary material for K=5 labels. 

# Regression 

The file 'data' contains the real datasets that were downloaded from https://github.com/tsoumakas/mulan/blob/master/data/multi-target/README.md 

Regression.ipynb contains experiments for Regression, on simulated data and on real data.

If one wants to compute the WSC coverage metric (which may take time), the file functions.py must be replaced by functions_with_wscCoverage.py

# Others 

Examples.ipynb contains codes to produce figures that exemplify transport-based quantiles 

