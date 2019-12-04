#!/usr/bin/julia
## **********************************************************************
## Main script to train a procedure for the clustering example in Gaussian Mixture Model (GMM).
## **********************************************************************
## * Requires running on GPU *
## **********************************************************************
## NOTE: Make sure this script is executable to run in batch mode.
## **********************************************************************

include("prediction_cluster_GMM.jl")

main = prediction_cluster_GMM.main

maxRisk, riskmat = main(
"--seed 54321
--nepochs 100000000 --maxRiskInit 0 --gpu 1 --maxRiskEvery 100 --nbatch 1000
--udim 2 --xdim 10 --nT 25 --hiddenT 40 20 40 --hiddenPi 25
--optimPi Adam(lr=0.001,beta1=0.5) --optimT Adam(lr=0.001,beta1=0.5)
--outdir ./saved_models_n10/ --saveEvery 100");
