#!/usr/bin/julia

## **********************************************************************
## 1. Interrogatation of the top Tᵏ models for centers (C(X)) (run on GPU)
## **********************************************************************
## This script evaluates the risk of the candidate estimators for the centers (μ,σ)
## in the Confidence Region (CR) example.
## Each estimator is evaluated at the 1,000 random draws γ from the parameter space.
## The risk at each γ is averaged over 1,000 random datasets.
## The epoch with the best worst-case risk is selected as the final estimator for the
## centers C(X), to be followed by training the offsets L(X)/U(X) estimator.
## Together, the centers C(X) and the offsets L(X)/U(X) are used to construct the
## CR estimator.
## **********************************************************************
## NOTE: Make sure this script is executable to be able to run in batch mode.
## **********************************************************************

## where selected best models are currently residing (along with max risk vector)
destdir = "./saved_models_centersCx_pretrained/"
nstarts = 1000 ## number of random γ for random grid search
nbatch = 1000 ## batch size for evaluation of risk at each random γ
## ----------------------------------------------------------------------
srand(12345);
o = Dict(:kcent        => 2.0f0,
         :losstypecent => :L2,
         :epoch        => 1000,
         :Pirange      => [[-10.0f0, 10.0f0], [0.5f0, 10.0f0]],
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
         :Tlossη2      => 0.30f0,
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
using maximinNN1: findMaxRisksPᵏ2, ParamBox, ParamModelPᵏ, ParamCI, gpu

maxriskpath = destdir*"pretrain_Cxrisks.jld2";
CxRisks = load(maxriskpath)["CxRisks"];

parboxPiᵏ = ParamBox([par for par in o[:Pirange]])
modelP = ParamModelPᵏ(parboxPiᵏ,Ψ=Symbol(o[:truepsi]);name=parse(o[:name]),
                      xdim=o[:xdim],mTmodel=o[:mTmodel],δ=o[:Piδoffsets])
parboxTᵏ = ParamBox([par for par in o[:Trange]])
paramCI = ParamCI(parboxTᵏ,o[:nparamsT];α=o[:alpha],
                  η1=o[:Tlossη1],η2=o[:Tlossη2],η_α=o[:Tlossη_α],η_β=o[:Tlossη_β],
                  λ=Float32(o[:lambdaCI]),λMC=o[:lambdaMC],
                  δ=o[:Tδoffsets],trtype=Symbol(o[:Ttransform]))


function evalMeanRisk(modeln, destdir)
  destpath = joinpath(destdir, modeln * ".bson");
  BSON.@load String(destpath) mTᵏ
  mTᵏ = mTᵏ |> gpu
  mTᵏ = Flux.mapleaves(Flux.Tracker.data, mTᵏ)

  findworstPᵏ2(mTᵏ,modelP,paramCI,o,SGAtypes,nstarts) = findMaxRisksPᵏ2(
                                                            mTᵏ,modelP,paramCI,o,SGAtypes,nstarts,:rand;
                                                            nruns=0,nbatch=o[:nbatch])
  @time maxR, maxRγ, maxRγD, SGAtypesRef, RisksALL, randγ = findworstPᵏ2(mTᵏ,modelP,paramCI,o,[:cent],o[:nstarts])
  meanR = mean(RisksALL)
  maxR  = maxR[1]
  # for j in 1:size(randγ)[2]
  #   println((:γ, randγ[:,j], :cent, round(RisksALL[1,j],3)))
  # end
  println("----------------------------------------")
  println((:modeln, modeln, :maxRisk, maxR, :meanRisk, meanR))
  println("----------------------------------------")
  return maxR, meanR
end

selTᵏ = (CxRisks[:rSGA_Cx] .<= 0.050f0) .& (CxRisks[:rEpoch] .>= 50000);
# selTᵏ = (CxRisks[:rSGA_Cx] .<= 0.035f0) .& (CxRisks[:rEpoch] .>= 50000);
selEpochs = CxRisks[:rEpoch][selTᵏ];
println("----------------------------------------")
println("total # of models to interrogate: $(sum(selTᵏ))") # 355 (0.05 & > 50K epochs)
println("----------------------------------------")
modeln = "0".*string.(selEpochs);
destpath = (destdir .* modeln .* ".bson");

## ------------------------------------------------------------------------------
## Deeper interrogate of best models (max and mean risks based on new batch size and new # of random γ)
## ------------------------------------------------------------------------------
maxCxR = Array{Float64}(length(destpath));
meanCxR = Array{Float64}(length(destpath));
for i = 1:length(destpath)
  maxCxR[i], meanCxR[i] = evalMeanRisk(modeln[i], destdir)
end

minmaxCxRind = findmin(maxCxR)[2]
minmaxCxEpoch = CxRisks[:rEpoch][selTᵏ][minmaxCxRind]
println((:minmaxCxR, findmin(maxCxR), :epoch, minmaxCxEpoch))
minmeanCxRind = findmin(meanCxR)[2]
minmeanCxEpoch = CxRisks[:rEpoch][selTᵏ][minmeanCxRind]
println((:minmeanCxR, findmin(meanCxR), :epoch, minmeanCxEpoch))

# ----------------------------------------
# (:modeln, "072200", :maxRisk, 0.034127563f0, :meanRisk, 0.02933348f0)
# ----------------------------------------
# (:modeln, "080050", :maxRisk, 0.04205643f0, :meanRisk, 0.028595747f0)
# ----------------------------------------
# (:minmaxCxR, (0.03412756323814392, 144), :epoch, 72200)
# (:minmeanCxR, (0.02859574742615223, 208), :epoch, 80050)

deepCxRisks = Dict(
  :rEpoch        => CxRisks[:rEpoch][selTᵏ],
  # :maxCxR      => CxRisks[:rSGA_Cx][selTᵏ],
  :maxCxR        => maxCxR,
  :maxCxREpoch   => minmaxCxEpoch,
  :meanCxR       => meanCxR,
  :meanCxREpoch  => minmeanCxEpoch,
)

save(joinpath(destdir, "deepCxRisks.jld2"), "deepCxRisks", deepCxRisks)
