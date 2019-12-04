#!/usr/bin/julia

# cd ./learning-to-learn-from-data/02_pointEstimation_illPosed
# julia

# Load the functions
include("maximinNN1.jl")
findPᵏ = maximinNN1.findPᵏ
loadNN = maximinNN1.loadNN
ParamBox = maximinNN1.ParamBox
ParamModelPᵏ = maximinNN1.ParamModelPᵏ
predictTᵏ = maximinNN1.predictTᵏ

using JLD2
using Knet
using FileIO

l2constraint = 10f0
atype = KnetArray{Float32}
numPars = 3

for setting in ["mandel2","mtcars2"]
	@load joinpath(setting,"risks","maxRisk.jld2") maxRisk riskmat riskEpoch

	# Dimension of data
	if (setting=="mandel") | (setting=="mandel2")
		xdim = 8
	elseif (setting=="mtcars") | (setting=="mtcars2")
		xdim = 32
	end

	# Make the model
	parsrange = [[-l2constraint,l2constraint] for i in 1:numPars]

	## define the model parameter space
	parbox = ParamBox([convert(atype, par) for par in parsrange])
	if (setting=="mandel") | (setting=="mandel2")
		modelP = ParamModelPᵏ(parbox, Ψ=Symbol("Ψσ"); name = parse("mandel_design"), xdim=xdim, l2constraint=l2constraint)
	elseif (setting=="mtcars") | (setting=="mtcars2")
		modelP = ParamModelPᵏ(parbox, Ψ=Symbol("Ψσ"); name = parse("mtcars_design"), xdim=xdim, l2constraint=l2constraint)
	end

	# Identify the iteration to interrogate
	if setting=="mandel"	#choose the epoch with minimal estimated max risk
		epoch = riskEpoch[indmin(maxRisk)]
	elseif setting=="mtcars"	#choose the epoch with minimal estimated max risk occurring after iteration 4 million
		maxRisk = maxRisk[riskEpoch.>4e6]
		riskEpoch = riskEpoch[riskEpoch.>4e6]
		epoch = riskEpoch[indmin(maxRisk)]
	elseif setting=="mandel2"
		epoch = 75000
	elseif setting=="mtcars2"
		epoch = 3225000
	end

	# Load the data
	fn = @sprintf("%04d.jld2",epoch)
	mPiᵏ, mPiᵣ, mTᵏ = loadNN(atype,joinpath(setting,"models",fn))
	modelTᵏ(z1n) = predictTᵏ(mTᵏ,z1n)

	# Evaluate max risk based on random search
	# draws from l2constraint-radius sphere of numPars dimensions, and then ignores one of the dimensions
	# (the direction used to turn the l2 equality constraint into an inequality constraint), thereby
	# biasing towards large radii
	finalMaxRisk_outer = findPᵏ(modelTᵏ,modelP;atype=atype, opt = Adam, nbatch=2000, nruns=0, nstarts=50000, ntest=20000, uniformSphere = false, avgMaxOut = true)
	println((:finalMaxRisk_outer,finalMaxRisk_outer[1])); flush(STDOUT)

	# Evaluate max risk based on random search
	# Draws from a unit sphere of dimension (numPars-1), and then draws a radius uniformly
	# Samples more smaller-radii observations
	finalMaxRisk_uniform = findPᵏ(modelTᵏ,modelP;atype=atype, opt = Adam, nbatch=2000, nruns=0, nstarts=50000, ntest=20000, uniformSphere = true, avgMaxOut = true)
	println((:finalMaxRisk_uniform,finalMaxRisk_uniform[1])); flush(STDOUT)

	if finalMaxRisk_outer[1]>finalMaxRisk_uniform[1]
		finalMaxRisk = finalMaxRisk_outer
	else
		finalMaxRisk = finalMaxRisk_uniform
	end

	save(joinpath(setting,"risks","finalMaxRisk.jld2"),
        "finalMaxRisk", finalMaxRisk,
        "finalMaxRisk_outer", finalMaxRisk_outer,
        "finalMaxRisk_uniform", finalMaxRisk_uniform,
        "epoch", epoch)
end
