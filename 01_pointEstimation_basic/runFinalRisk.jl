#!/usr/bin/julia

# cd ./01_pointEstimation_basic
# Start Julia in paralell mode:
# julia -p 16
# NOTE: These jobs were launched interactively, rather than by submitting a bash script, because wasn't running in parallel when submitted as a bash script

# Load the functions
@everywhere include("finalRisk.jl")
main = finalRisk.finalRiskMain

##########
# Estimating the mean at n=50

# Takes ~60k epochs to converge
main("--gpu 0 --ntest 50000 --xdim 50 --seed 54321 --parsrange [[-5.0, 5.0], [1.0, 4.0]] --truepsi Ψμ --name norm --loaddir ./globalNormalMean_n50/ --gridparsrange [[-5.0, 5.0], [3.8, 4.0]] --Rgridsize 0.01")

##########
# Estimating the standard deviation at n=50

main("--gpu 0 --ntest 50000 --xdim 50 --seed 54321 --parsrange [[-5.0, 5.0], [1.0, 4.0]] --truepsi Ψσ --name norm --loaddir ./globalNormalStdDev_n50/ --gridparsrange [[-5.0, 5.0], [3.8, 4.0]] --Rgridsize 0.01")

##########
# Estimating the mean at n=1

# Run the maximal risk in the setting of Table 2 of Casella & Strawderman 1981 (minimax for m=1.4,1.5,1.6)
for m in [1.4:0.1:1.6;1.05;0.1:0.1:1.0]
	println("m: ", m)
	main("--gpu 0 --ntest 50000 --xdim 1 --seed 54321 --parsrange [[-"*string(m)*", "*string(m)*"], [1.0, 1.0]] --truepsi Ψμ --name norm --loaddir ./globalNormalMean_n1_m"*string(m)*"/ --gridparsrange [[-"*string(m)*", "*string(m)*"], [1.0, 1.0]] --Rgridsize 0.0001")
end
