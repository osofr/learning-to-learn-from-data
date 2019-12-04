#!/usr/bin/julia

# cd ./01_pointEstimation_basic
# julia

#################################

# This script evaluates the risk of the estimators in the Neal example
# at a grid of points. This is the same grid of points that was used
# in the call in neal.jl that generated the maxRisk.jld2 file.
# Overall the risk values generated in this file are very similar
# to those in the interrogations in the maxRisk.jld2 file, though
# using the known bounds on gamma does slightly improve stability.

#################################

# number of Monte Carlo draws
ntest = 100000
# vector of parameter values at which to interrogate Tk
γvals = collect(0.0:0.001:2.0)
# epoch to be interrogated
finalEpoch = 500000
# sample sizes used in the models
nvals = 10.*2.^(collect(0:4))

#################################

using JLD2
using Knet
using FileIO

include("maximinNN1.jl")
main = maximinNN1.main
loadNN = maximinNN1.loadNN
predictTᵏ = maximinNN1.predictTᵏ
allocX1n = maximinNN1.allocX1n
ParamModelPᵏ = maximinNN1.ParamModelPᵏ
ParamBox = maximinNN1.ParamBox
sim! = maximinNN1.sim!

atype = KnetArray{Float32}

parbox = ParamBox([convert(atype, par) for par in [[0.0, 2.0], [0.0, 0.0]]])

for n in nvals
	println((:n,n))

	mPiᵏ, mPiᵣ, mTᵏ = loadNN(atype,"./globalNeal_n"*string(n)*"/models/"*string(finalEpoch)*".jld2")

	modelP = ParamModelPᵏ(parbox, Ψ=Symbol("Ψμ"); name = parse("neal"), xdim=n)

	riskmat = zeros(length(γvals),1,1)

	for i in 1:length(γvals)
		γ = γvals[i]

		γarray = [convert(atype,[γ for i=1:ntest]),convert(atype,[0.0 for i=1:ntest])]

		x = allocX1n(modelP, ntest; atype=atype)

		x = sim!(modelP,γarray,x)

		se = abs2.(min.(max.(0.0,convert(Array{Float32},predictTᵏ(mTᵏ,x)[1,1:ntest])),2.0).-γ)

		riskmat[i,1,1] = mean(se)
	end

	filepath = "./globalNeal_n"*string(n)*"/risks/maxRisk_estBounded.jld2"
    save(filepath,
        "maxRisk", maximum(riskmat),
        "riskmat", riskmat,
        "riskEpoch", [finalEpoch])

end