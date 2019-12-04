#!/usr/bin/julia

# cd ./01_pointEstimation_basic
# Start Julia in paralell mode:
# julia -p 4

# Load the functions
@everywhere include("maximinNN1.jl")
main = maximinNN1.main

##########
# Estimating the parameter of the distribution from
# https://radfordneal.wordpress.com/2008/08/09/inconsistent-maximum-likelihood-estimation-an-ordinary-example/
# at different sample sizes

for n in [10,20,40,80,160]
	println("n: ", n)
	# Similar grid search settings as for estimating mean at n=1 (but, Rgrid=0.001 instead of 0.0001)
	main("--nepochs 500000 --seed 54321 --truepsi Ψμ --name neal --xdim "*string(n)*" --udim 2 --niter 1 --hiddenT "*string(2*n)*" "*string(n)*" "*string(n)*" --hiddenPi 10 10 10 --optimPi Adam(lr=0.001,beta1=0.5) --optimT Adam(lr=0.001,beta1=0.5) --parsrange [[0.0, 2.0], [0.0, 0.0]] --maxRiskEvery 500000 --maxRiskInit 0 --gpu 1 --verbose 0  --Rgrid 1 --Rgridsize 0.001 --ntest 50000 --outdir ./globalNeal_n"*string(n)*"/ --saveEvery 50000")
end
