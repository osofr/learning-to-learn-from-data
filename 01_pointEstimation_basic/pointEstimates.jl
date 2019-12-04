#!/usr/bin/julia

# Load the functions
@everywhere include("maximinNN1.jl")
main = maximinNN1.main

##########
# Estimating the mean at n=50

main("--nepochs 100000 --seed 54321 --truepsi Ψμ --name norm --xdim 50 --udim 2 --niter 1 --hiddenT 30 30 30 --hiddenPi 10 10 10 --optimPi Adam(lr=0.001,beta1=0.5) --optimT Adam(lr=0.001,beta1=0.5) --parsrange [[-5.0, 5.0], [1.0, 4.0]] --maxRiskEvery 25 --SGAnruns 1000 --SGAnstarts 50 --maxRiskInit 1 --gpu 1 --verbose 0 --Rgrid 1 --Rgridsize 0.125 --outdir ./globalNormalMean_n50/ --saveEvery 25")

##########
# Estimating the standard deviation at n=50

main("--nepochs 200000 --seed 54321 --truepsi Ψσ --name norm --xdim 50 --udim 2 --niter 1 --hiddenT 50 50 10 10 10 --hiddenPi 10 10 10 --optimPi Adam(lr=0.001,beta1=0.5) --optimT Adam(lr=0.001,beta1=0.5) --parsrange [[-5.0, 5.0], [1.0, 4.0]] --maxRiskEvery 400 --SGAnruns 1000 --SGAnstarts 50 --maxRiskInit 1 --gpu 1 --verbose 0 --Rgrid 1 --Rgridsize 0.125 --outdir ./globalNormalStdDev_n50/ --saveEvery 400")


##########
# Estimating the mean at n=1

# Run the maximal risk in the setting of Table 2 of Casella & Strawderman 1981 (minimax for m=1.4,1.5,1.6)
for m in [1.4:0.1:1.6;1.05;0.1:0.1:1.0]
	println("m: ", m)
	# Fairly good convergence by 5k epochs
	main("--nepochs 100000 --seed 54321 --truepsi Ψμ --name norm --xdim 1 --udim 2 --niter 1 --hiddenT 15 15 15 15 --hiddenPi 10 10 10 10 --optimPi Adam(lr=0.001,beta1=0.5) --optimT Adam(lr=0.001,beta1=0.5) --parsrange [[-"*string(m)*", "*string(m)*"], [1.0, 1.0]] --maxRiskEvery 10000 --maxRiskInit 0 --gpu 1 --verbose 0 --Rgrid 1 --Rgridsize 0.0001 --ntest 50000 --outdir ./globalNormalMean_n1_m"*string(m)*"/ --saveEvery 10000")
end



# This R code was used to evaluate the true minimax risk for the normal mean estimation example with n=1.
# R code for evaluating true minimax risk for m<=1.05
# m.max = 1.05
# mat = rbind(c(NA,NA))
# for(m in seq(0,m.max,by=0.01)){
# 	err = c(m,integrate(function(x){(m*tanh(m*x)-m)^2*dnorm(x,mean=m)},lower=-7,upper=7)$value)
# 	mat = rbind(mat,err)
# }
# mat = mat[-1,]
# print(mat)
