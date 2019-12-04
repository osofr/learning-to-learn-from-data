#!/usr/bin/julia

## **********************************************************************
## 4. Deep interrogatation of the Confidence Region estimator (run on GPU) (same as 3).
##    This script evaluates the risks of the final estimator for Confidence Region (CR) example.
##    Evaluation is performed over an evenly spaced grid of the parameter space.
##    Interrogate for low σ values in parameter space σ ∈ [0.5,1.0],
##    where we know our estimator is under-performing.
## **********************************************************************
## NOTE: Make sure this script is executable to be able to run in batch mode.
## **********************************************************************

## where selected best models are currently residing (along with max risk vector)
destdir = "./saved_models_offsetsLU/models/"
ngridμ = 500 ## number of random γ for random grid search
ngridσ = 50 ## number of random γ for random grid search
nbatch = 5000 ## batch size for evaluation of risk at each γ
ηvals = [0.05f0, 0.02f0, 0.0f0] ## values of η which are used during interrogation of CIs
selEpochs = 142100 ## the specific model update to interrogate
modeln = string.(selEpochs);
destpath = (destdir .* modeln .* ".bson");

## -----------------------------------------------------------------------------
srand(12345);
o = Dict(:kcent        => 2.0f0,
         :losstypecent => :L2,
         :epoch        => 1000,
         :Pirange      => [[-10.0f0, 10.0f0], [0.5f0, 1.0f0]],
         :Trange       => [[-10.0f0, 10.0f0], [0.5f0, 10.0f0]],
         :truepsi      => "Ψ2dim_μ_σ",
         :name         => "norm",
         :xdim         => 50,
         :mTmodel      => :LSTM2x,
         :Piδoffsets   => [0.5f0, 0.099f0],
         :Tδoffsets    => [0.5f0, 0.2f0],
         :nparamsT     => 2,
         :alpha        => 0.05f0,
         :Tlossη1      => 0.10f0,
         :Tlossη2      => 0.05f0,
         :Tlossη_α     => 1.5f0,
         :Tlossη_β     => 4.0f0,
         :lambdaCI     => 2.0f0,
         :lambdaMC     => 100.0f0,
         :Ttransform   => "linear",
         :nbatch       => nbatch,
         :ngridμ       => ngridμ,
         :ngridσ       => ngridσ)

include("maximinNN1.jl")
main = maximinNN1.main
using JLD, JLD2, FileIO, BSON, Flux
using maximinNN1: findMaxRisksPᵏ2, MaxRisksPᵏInterrogate, ParamBox, ParamModelPᵏ, ParamCI, corner, mid, rand, gpu

parboxPiᵏ = ParamBox([par for par in o[:Pirange]])
modelP = ParamModelPᵏ(parboxPiᵏ,Ψ=Symbol(o[:truepsi]);name=parse(o[:name]),
                      xdim=o[:xdim],mTmodel=o[:mTmodel],δ=o[:Piδoffsets])

parboxTᵏ = ParamBox([par for par in o[:Trange]])
paramCI = ParamCI(parboxTᵏ,o[:nparamsT];α=o[:alpha],
                  η1=o[:Tlossη1],η2=o[:Tlossη2],η_α=o[:Tlossη_α],η_β=o[:Tlossη_β],
                  λ=Float32(o[:lambdaCI]),λMC=o[:lambdaMC],
                  δ=o[:Tδoffsets],trtype=Symbol(o[:Ttransform]))

lb = maximinNN1.lb
ub = maximinNN1.ub
gridμ = linspace(lb(modelP)[1], ub(modelP)[1], o[:ngridμ])
gridσ = linspace(lb(modelP)[2], ub(modelP)[2], o[:ngridσ])
# riskmat = zeros(Float32, (length(grid[1]), length(grid[2]), div(o[:nepochs],o[:maxRiskEvery])+o[:maxRiskInit]))

xgridγ = Any[]
for xμ in gridμ, xσ in gridσ
  push!(xgridγ, [xμ, xσ])
end
xgridγ = hcat(xgridγ...)

function evalMeanRiskGrid(modeln, destdir, xinit, ηvals = [0.05f0])
  destpath = joinpath(destdir, modeln * ".bson");
  BSON.@load String(destpath) mTᵏ
  mTᵏ = mTᵏ |> gpu
  mTᵏ = Flux.mapleaves(Flux.Tracker.data, mTᵏ)

  MaxRisksPᵏ(mTᵏ,modelP,paramCI,o,SGAtypes,xinit,ηval) = MaxRisksPᵏInterrogate(
                                                          mTᵏ,modelP,paramCI,o,SGAtypes,xinit,ηval;
                                                          nruns=0,nbatch=o[:nbatch])

  SGAtypes = [:typeIerr,:len,:lenScaled,:cent]
  ηval = ηvals[1]
  # RisksALL = randγ = SGAtypesRef = maxRγD = max_tIerr = Risks_typeIerr = Risks_len = Risks_lenScaled = Risks_cent = 0.0f0
  # for ηval in ηvals
  maxR, maxRγ, maxRγD, SGAtypesRef, RisksALL, randγ = MaxRisksPᵏ(mTᵏ,modelP,paramCI,o,SGAtypes,xinit,ηval)
  for j in 1:5
    println((:η, ηval, :γ, randγ[:,j], :typeIerr, round(RisksALL[SGAtypesRef[:typeIerr],j],3),
                             :len, round(RisksALL[SGAtypesRef[:len],j],3),
                             :lenScaled, round(RisksALL[SGAtypesRef[:lenScaled],j],3),
                             :cent, round(RisksALL[SGAtypesRef[:cent],j],3)
                             ))
  end

  Risks_typeIerr = RisksALL[SGAtypesRef[:typeIerr],:]
  Risks_len = RisksALL[SGAtypesRef[:len],:]
  Risks_lenScaled = RisksALL[SGAtypesRef[:lenScaled],:]
  Risks_cent = RisksALL[SGAtypesRef[:cent],:]
  γ_max_tIerr = maxRγD[:typeIerr][2]
  max_tIerr = maxRγD[:typeIerr][1]
  mean_tIerr = mean(Risks_typeIerr)
  L2mean_tIerr = sqrt(mean(Risks_typeIerr .^ 2))

  println("----------------------------------------")
  println((:η,    ηval, :tIerr,
           :γ_max_tIerr,  γ_max_tIerr,
           :max_tIerr,    round(max_tIerr,3),
           :mean_tIerr,   round(mean_tIerr,3),
           :L2mean_tIerr, round(L2mean_tIerr,3)))
  println((:η,      ηval, :len,
           :γ_max_len,    maxRγD[:len][2],
           :max_len,      round(maxRγD[:len][1],3)))
  println((:η,    ηval, :lenσ²,
           :γ_max_lenσ²,  maxRγD[:lenScaled][2],
           :max_lenσ²,    round(maxRγD[:lenScaled][1],3),
           :mean_lenσ²,   round(mean(Risks_lenScaled),3)))
  println((:η,     ηval, :cent,
           :γ_max_cent,   maxRγD[:cent][2],
           :max_cent,     round(maxRγD[:cent][1],3)))
  println("----------------------------------------")
  # end

  println("----------------------------------------")
  println((:modeln, modeln, :max_tIerr, max_tIerr))
  println("----------------------------------------")
  return max_tIerr, [maxRγD, randγ, Risks_typeIerr, Risks_len, Risks_lenScaled, Risks_cent]
end

## -----------------------------------------------------------------------------
## Deeper interrogate of best models (max and mean risks based on new batch size and new # of random γ)
## -----------------------------------------------------------------------------
max_tIerr = ones(Float64, length(ηvals));
maxRγD = Any[]
Risks_typeIerr = Any[]
Risks_len = Any[]
Risks_lenScaled = Any[]
Risks_cent = Any[]

for i in 1:length(ηvals)
  max_tIerr[i], risks  = evalMeanRiskGrid(modeln, destdir, xgridγ, ηvals[i])
  push!(maxRγD, risks[1])
  push!(Risks_typeIerr, risks[3])
  push!(Risks_len, risks[4])
  push!(Risks_lenScaled, risks[5])
  push!(Risks_cent, risks[6])
end

println("----------------------------------------")
println("FINISHED GID INTERROGATION")
println("----------------------------------------")
println("max_tIerr: "); println(max_tIerr)
println("----------------------------------------")
println("maxRγD[1][:typeIerr]: "); println(maxRγD[1][:typeIerr])
println("maxRγD[2][:typeIerr]: "); println(maxRγD[2][:typeIerr])
println("maxRγD[3][:typeIerr]: "); println(maxRγD[3][:typeIerr])
println("maxRγD[1][:len]: "); println(maxRγD[1][:len])
println("maxRγD[2][:len]: "); println(maxRγD[2][:len])
println("maxRγD[3][:len]: "); println(maxRγD[3][:len])
println("maxRγD[1][:lenScaled]: "); println(maxRγD[1][:lenScaled])
println("maxRγD[2][:lenScaled]: "); println(maxRγD[2][:lenScaled])
println("maxRγD[3][:lenScaled]: "); println(maxRγD[3][:lenScaled])
println("maxRγD[1][:cent]: "); println(maxRγD[1][:cent])
println("maxRγD[2][:cent]: "); println(maxRγD[2][:cent])
println("maxRγD[3][:cent]: "); println(maxRγD[3][:cent])
println("----------------------------------------")

rEpoch             = selEpochs
etas               = ηvals
max_tIerr          = max_tIerr
maxRgammaD         = maxRγD
xgridgamma         = xgridγ'
Risks_typeIerr     = hcat(Risks_typeIerr...)
Risks_len          = hcat(Risks_len...)
Risks_lenScaled    = hcat(Risks_lenScaled...)
Risks_cent         = Risks_cent[1]

writedir = "./saved_models_offsetsLU/risks/"
JLD.save(joinpath(writedir, "deep_CI_risks_lowsigma_142.1K.jld"),
  "rEpoch", rEpoch, "etas", etas, "max_tIerr", max_tIerr, "xgridgamma", xgridgamma,
  "Risks_typeIerr", Risks_typeIerr, "Risks_len", Risks_len, "Risks_lenScaled",
  Risks_lenScaled, "Risks_cent", Risks_cent, "maxRgammaD", maxRgammaD)
