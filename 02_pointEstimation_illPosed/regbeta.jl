#!/usr/bin/julia

# cd ./learning-to-learn-from-data/02_pointEstimation_illPosed
# julia

# Load the functions
include("maximinNN1.jl")
main = maximinNN1.main

##########
# Estimating the beta in front of the disp variable for a continuous outcome generated
# according to a linear regression using the design matrix of (disp,cyl) from the mtcars
# data set. More iterations used than in the first run (at the top of this page)

main("--nepochs 10000000 --seed 54321 --truepsi Ψσ --name mtcars_design --xdim 32 --udim 3 --niter 1 --hiddenT 100 --hiddenPi 30 --optimPi Adam(lr=0.0001,beta1=0.0) --optimT Adam(lr=0.001,beta1=0.5) --numPars 3 --nT 10 --maxRiskEvery 25000 --SGAnruns 0 --SGAnstarts 2000 --SGAnbatch 1000 --maxRiskInit 1 --gpu 1 --verbose 0 --Rgrid 0 --Rgridsize 1.0 --outdir ./mtcars2/ --saveEvery 5000 --l2constraint 10")

##########
# Estimating the beta in front of the x1 variable for a continuous outcome generated
# according to a linear regression using the design matrix of (x1,x2) from the Mandel
# data set

main("--nepochs 500000 --seed 54321 --truepsi Ψσ --name mandel_design --xdim 8 --udim 3 --niter 1 --hiddenT 100 --hiddenPi 30 --optimPi Adam(lr=0.0001,beta1=0.0) --optimT Adam(lr=0.0001,beta1=0.5)  --numPars 3 --nT 10 --maxRiskEvery 25000 --SGAnruns 0 --SGAnstarts 2000 --SGAnbatch 1000 --maxRiskInit 1 --gpu 1 --verbose 0 --Rgrid 0 --Rgridsize 1.0 --outdir ./mandel2/ --saveEvery 5000 --l2constraint 10")
