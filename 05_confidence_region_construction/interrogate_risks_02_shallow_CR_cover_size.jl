#!/usr/bin/julia

## **********************************************************************
## 2. Shallow interrogatation of the Confidence Region (CR) models (runs on GPU).
##    Evaluate worst-case coverage and worst-case CR size over 5000 random draws from parameter space (γ).
##    The risk for each γ is averaged over 5,000 datasets.
##    This script finds the epoch with the best performing candidate estimator for
##    the CR example, based on the best worst-case coverage.
## **********************************************************************
## NOTE: Make sure this script is executable to be able to run in batch mode.
## **********************************************************************


## where selected best models are currently residing (along with max risk vector)
destdir = "./saved_models_offsetsLU/models/"
nstarts = 5000 ## number of random γ for random grid search
nbatch = 5000 ## batch size for evaluation of risk at each random γ
ηvals = [0.05f0] ## values of η which are used during interrogation of CIs
selEpochs = 100000:100:160000

println("----------------------------------------")
println("total # of models to interrogate: $(length(selEpochs))")
println("----------------------------------------")
modeln = string.(selEpochs);
destpath = (destdir .* modeln .* ".bson");

## ----------------------------------------------------------------------
srand(12345);
o = Dict(:kcent        => 2.0f0,
         :losstypecent => :L2,
         :epoch        => 1000,
         # :Pirange      => [[-10.0f0, 10.0f0], [0.5f0, 10.0f0]],
         :Pirange      => [[-10.0f0, 10.0f0], [1.0f0, 10.0f0]],
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
         :nstarts      => nstarts)

include("maximinNN1.jl")
main = maximinNN1.main
using JLD2, FileIO, BSON, Flux
using maximinNN1: findMaxRisksPᵏ2, MaxRisksPᵏInterrogate, ParamBox, ParamModelPᵏ, ParamCI, corner, mid, rand, gpu

parboxPiᵏ = ParamBox([par for par in o[:Pirange]])
modelP = ParamModelPᵏ(parboxPiᵏ,Ψ=Symbol(o[:truepsi]);name=parse(o[:name]),
                      xdim=o[:xdim],mTmodel=o[:mTmodel],δ=o[:Piδoffsets])

parboxTᵏ = ParamBox([par for par in o[:Trange]])
paramCI = ParamCI(parboxTᵏ,o[:nparamsT];α=o[:alpha],
                  η1=o[:Tlossη1],η2=o[:Tlossη2],η_α=o[:Tlossη_α],η_β=o[:Tlossη_β],
                  λ=Float32(o[:lambdaCI]),λMC=o[:lambdaMC],
                  δ=o[:Tδoffsets],trtype=Symbol(o[:Ttransform]))

function evalMeanRisk(modeln, destdir, ηvals = [0.15f0])
  destpath = joinpath(destdir, modeln * ".bson");
  BSON.@load String(destpath) mTᵏ
  mTᵏ = mTᵏ |> gpu
  mTᵏ = Flux.mapleaves(Flux.Tracker.data, mTᵏ)

  srand(12345);
  xcorners = corner(modelP)
  xborders = Any[]
  for x in xcorners
    push!(xborders, [x[1], rand(modelP, 1)[1][2]])
    push!(xborders, [rand(modelP, 1)[1][1], x[2]])
  end
  xinit = [xcorners; xborders; [mid(modelP)]; rand(modelP, o[:nstarts])]
  xinit = hcat(xinit...)

  MaxRisksPᵏ(mTᵏ,modelP,paramCI,o,SGAtypes,xinit,ηval) = MaxRisksPᵏInterrogate(
                                                         mTᵏ,modelP,paramCI,o,SGAtypes,xinit,ηval;
                                                         nruns=0,nbatch=o[:nbatch])

  SGAtypes = [:typeIerr,:len,:lenScaled,:cent]

  RisksALL = randγ = SGAtypesRef = maxRγD = max_tIerr = Risks_typeIerr = Risks_len = Risks_lenScaled = Risks_cent = 0.0f0
  for ηval in ηvals
    @time maxR, maxRγ, maxRγD, SGAtypesRef, RisksALL, randγ = MaxRisksPᵏ(mTᵏ,modelP,paramCI,o,SGAtypes,xinit,ηval)

    for j in 1:1
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

  end
  println("----------------------------------------")
  println((:modeln, modeln, :max_tIerr, max_tIerr))
  println("----------------------------------------")
  return max_tIerr, [maxRγD, randγ, Risks_typeIerr, Risks_len, Risks_lenScaled, Risks_cent]
end

## ------------------------------------------------------------------------------
## Deeper interrogate of best models (max and mean risks based on new batch size and new # of random γ)
## ------------------------------------------------------------------------------
max_tIerr = ones(Float64, length(destpath));
maxRγD = Any[]
randγ = Any[]
Risks_typeIerr = Any[]
Risks_len = Any[]
Risks_lenScaled = Any[]
Risks_cent = Any[]

for i = 1:length(destpath)
  try
    @time max_tIerr[i], risks  = evalMeanRisk(modeln[i], destdir, ηvals)
    push!(maxRγD, risks[1])
    push!(randγ, risks[2])
    push!(Risks_typeIerr, risks[3])
    push!(Risks_len, risks[4])
    push!(Risks_lenScaled, risks[5])
    push!(Risks_cent, risks[6])
  catch
    println("model not found, $(modeln[i])")
    max_tIerr[i] = 99.0
  end

  min_tIerr = findmin(max_tIerr)[1]
  indmin_tIerr = findmin(max_tIerr)[2]
  tIerr_Epoch = selEpochs[indmin_tIerr]
  println((:min_tIerr, round(min_tIerr, 4), :indmin_tIerr, indmin_tIerr, :epoch, tIerr_Epoch))
  println("----------------------------------------")
  best_tIerr_CIinterrogate = Dict(
    :rEpoch             => selEpochs,
    :indmin_tIerr       => indmin_tIerr,
    :min_tIerr          => min_tIerr,
    :tIerr_Epoch        => tIerr_Epoch,
    :max_tIerr          => max_tIerr,
    :maxRγD             => maxRγD,
    :randγ              => randγ,
    :Risks_typeIerr     => Risks_typeIerr,
    :Risks_len          => Risks_len,
    :Risks_lenScaled    => Risks_lenScaled,
    :Risks_cent         => Risks_cent
  )
  writedir = "./saved_models_offsetsLU/risks/"
  save(joinpath(writedir, "best_tIerr_CIinterrogate.jld2"), "best_tIerr_CIinterrogate", best_tIerr_CIinterrogate)
end

println("----------------------------------------")
println("FINISHED INTERROGATION")
println("----------------------------------------")
min_tIerr = findmin(max_tIerr)[1]
indmin_tIerr = findmin(max_tIerr)[2]
tIerr_Epoch = selEpochs[indmin_tIerr]
println("----------------------------------------")
println("max_tIerr"); println(max_tIerr)
println("----------------------------------------")
println((:min_tIerr, round(min_tIerr, 4), :indmin_tIerr, indmin_tIerr, :epoch, tIerr_Epoch))
println("----------------------------------------")

best_tIerr_CIinterrogate = Dict(
  :rEpoch             => selEpochs,
  :indmin_tIerr       => indmin_tIerr,
  :min_tIerr          => min_tIerr,
  :tIerr_Epoch        => tIerr_Epoch,
  :max_tIerr          => max_tIerr,
  :maxRγD             => maxRγD,
  :randγ              => randγ,
  :Risks_typeIerr     => Risks_typeIerr,
  :Risks_len          => Risks_len,
  :Risks_lenScaled    => Risks_lenScaled,
  :Risks_cent         => Risks_cent
)

writedir = "./saved_models_offsetsLU/risks/"
save(joinpath(writedir, "best_tIerr_CIinterrogate.jld2"), "best_tIerr_CIinterrogate", best_tIerr_CIinterrogate)
