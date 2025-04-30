This repo contains codes for our paper on Optimal-Transport based conformal prediction, available at https://arxiv.org/abs/2501.18991

functions.py contains our main functions

# Classification

Classification.ipynb contains experiments for Classification in the main body of the paper.

Classification2.ipynb contains experiments for Classification in the supplementary material for K=5 labels. 

# Regression 

The file 'data' contains the real datasets that were downloaded from https://github.com/tsoumakas/mulan/blob/master/data/multi-target/README.md 

Regression.ipynb contains experiments for Regression, on simulated data and on real data.

# Files taken from previous works

Experiments in classification require the package arc for the ARS method, available at: https://github.com/msesia/arc 
Folder : arc 

Experiments in regression involve the local ellipsoid method, available at:  https://github.com/M-Soundouss/EllipsoidalConformalMTR/tree/main
File : ellipsoidal_conformal_utilities.py
