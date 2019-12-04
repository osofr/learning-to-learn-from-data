#!/usr/bin/julia -p 8

## **********************************************************************
## Deep interrogatation of the risks of the final clustering predictor in Gaussian Mixture Model (GMM).
##    This script evaluates the risks of the procedure developed for the clustering example
##    over an evenly spaced grid of parameters [μ1, μ2] at fixed
##    mixing probabilities (α = 0.0, 0.1, 0.2, 0.3, 0.4, 0.5).
## The risk is also evaluated for the comparison clustering procedures (EM and K-means).

## **********************************************************************
ngrid = 100    ## number of points on the grid for each axis
ntest = 10000  ## number of random dataset draws for each point on the parameter space
selEpochs = 20000
modeln = string.(selEpochs);
atypeCPU = "Array{Float32}"
atype = eval(parse(atypeCPU))

using JLD
## -----------------------------------------------------------------------------
o = Dict(
  :epoch        => 20000,
  :parsrange    => [[-3.0f0, 3.0f0], [-3.0f0, 3.0f0]],
  :truepsi      => "Ψμ",
  :name         => "norm",
  :xdim         => 10,
  :udim         => 2,
  :ntest        => ntest,
  :ngrid        => ngrid,
  :atype        => atype)
## -----------------------------------------------------------------------------

@everywhere include("../prediction_cluster_GMM.jl")
main = prediction_cluster_GMM.main
using Knet, ArgParse, JLD2, FileIO
using Base.Iterators: repeated, partition
using prediction_cluster_GMM: ParamBox, ParamModelPᵏ, mid, rand, cpu, sim_norm!, sample_noise, allocX1n, drawα, evalTᵏrisks, lb, ub, loadNN, γTᵏrisk

parbox = ParamBox([convert(atype, par) for par in o[:parsrange]])
modelP = ParamModelPᵏ(parbox, Ψ=Symbol(o[:truepsi]); name = parse(o[:name]), xdim=o[:xdim])

gridμ1 = linspace(lb(modelP)[1], ub(modelP)[1], o[:ngrid])
gridμ2 = linspace(lb(modelP)[2], ub(modelP)[2], o[:ngrid])

xgridγ = Any[]
for xμ1 in gridμ1, xμ2 in gridμ2
  push!(xgridγ, [xμ1, xμ2])
end
xgridγ = hcat(xgridγ...)

function evalTᵏRisksGrid(xinit,αset)
  loadfile = "../saved_models_n10/models/20000.jld2"
  mPiᵏ, mPiᵣ, mTᵏ = loadNN(o[:atype], loadfile)
  nets = Dict(:mPiᵏ => mPiᵏ, :mPiᵣ => mPiᵣ, :mTᵏ => mTᵏ)

  lossTᵏ = SharedArray{Float32}(size(xinit)[2]);
  lossEM = SharedArray{Float32}(size(xinit)[2]);
  lossKM = SharedArray{Float32}(size(xinit)[2]);
  classerTᵏ = SharedArray{Float32}(size(xinit)[2]);
  classerEM = SharedArray{Float32}(size(xinit)[2]);
  classerKM = SharedArray{Float32}(size(xinit)[2]);

  ## loop over inidividual γ, evaluate the risk for one γ
  @sync @parallel for i = 1:size(xinit)[2]
    noise = prediction_cluster_GMM.sample_noise(o[:atype],o[:udim],o[:ntest]);
    z1ntest = prediction_cluster_GMM.allocX1n(modelP, o[:ntest]; atype=o[:atype]);
    γsamp = xinit[:,i]; μ1 = xinit[1,i]; μ2 = xinit[2,i];
    x1nC1 = prediction_cluster_GMM.sim_norm!(μ1, z1ntest);
    x1nC2 = prediction_cluster_GMM.sim_norm!(μ2, z1ntest);
    αsamp = prediction_cluster_GMM.drawα(o[:atype], size(z1ntest,2));
    αsamp .= αset

    lossTᵏ1, classerTᵏ1 = prediction_cluster_GMM.γTᵏrisk(o[:epoch],z1ntest,αsamp,γsamp,o[:atype],x1nC1,x1nC2,nets);
    lossEM1, classerEM1 = prediction_cluster_GMM.γEMrisk(o[:epoch],z1ntest,αsamp,γsamp,o[:atype],x1nC1,x1nC2);
    lossKM1, classerKM1 = prediction_cluster_GMM.γKmeansrisk(o[:epoch],z1ntest,αsamp,γsamp,o[:atype],x1nC1,x1nC2);

    lossTᵏ[i] = lossTᵏ1
    lossEM[i] = lossEM1
    lossKM[i] = lossKM1
    classerTᵏ[i] = classerTᵏ1
    classerEM[i] = classerEM1
    classerKM[i] = classerKM1
   end

  return lossTᵏ, lossEM, lossKM, classerTᵏ, classerEM, classerKM
end

## evluate the entire risk grid for fixed alpha and then de-bias max risks
function evalTᵏRisksGrid_debias(xinit,αset)
  lossTk_α, lossEM_α, lossKM_α, classerTk_α, classerEM_α, classerKM_α = evalTᵏRisksGrid(xinit, αset)

  Tᵏlmax, iTᵏlmax = findmax(lossTk_α)
  lossTk_α2, lossEM_α2, lossKM_α2, classerTk_α2, classerEM_α2, classerKM_α2 = evalTᵏRisksGrid(reshape(xinit[:,iTᵏlmax], 2, 1), αset)
  Tᵏlmax2 = lossTk_α2[1]
  Tᵏclassmax, iTᵏclassmax = findmax(classerTk_α)
  lossTk_α2, lossEM_α2, lossKM_α2, classerTk_α2, classerEM_α2, classerKM_α2 = evalTᵏRisksGrid(reshape(xinit[:,iTᵏclassmax], 2, 1), αset)
  Tᵏclassmax2 = classerTk_α2[1]

  EMlmax, iEMlmax = findmax(lossEM_α)
  lossTk_α2, lossEM_α2, lossKM_α2, classerTk_α2, classerEM_α2, classerKM_α2 = evalTᵏRisksGrid(reshape(xinit[:,iEMlmax], 2, 1), αset)
  EMlmax2 = lossEM_α2[1]
  EMclassmax, iEMclassmax = findmax(classerEM_α)
  lossTk_α2, lossEM_α2, lossKM_α2, classerTk_α2, classerEM_α2, classerKM_α2 = evalTᵏRisksGrid(reshape(xinit[:,iEMclassmax], 2, 1), αset)
  EMclassmax2 = classerEM_α2[1]

  KMlmax, iKMlmax = findmax(lossKM_α)
  lossTk_α2, lossEM_α2, lossKM_α2, classerTk_α2, classerEM_α2, classerKM_α2 = evalTᵏRisksGrid(reshape(xinit[:,iKMlmax], 2, 1), αset)
  KMlmax2 = lossKM_α2[1]
  KMclassmax, iKMclassmax = findmax(classerKM_α)
  lossTk_α2, lossEM_α2, lossKM_α2, classerTk_α2, classerEM_α2, classerKM_α2 = evalTᵏRisksGrid(reshape(xinit[:,iKMclassmax], 2, 1), αset)
  KMclassmax2 = classerKM_α2[1]

  println((:α, αset, :TᵏlBayes, mean(lossTk_α), :Tᵏlmax, Tᵏlmax, :Tᵏlmax2, Tᵏlmax2, :maxγ, xinit[:,iTᵏlmax]))
  println((:α, αset, :EMlBayes, mean(lossEM_α), :EMlmax, EMlmax, :EMlmax2, EMlmax2, :maxγ, xinit[:,iEMlmax]))
  println((:α, αset, :KMlBayes, mean(lossKM_α), :KMlmax, KMlmax, :KMlmax2, KMlmax2, :maxγ, xinit[:,iKMlmax]))

  println((:α, αset, :TᵏclassBayes, mean(classerTk_α), :Tᵏclassmax, Tᵏclassmax, :Tᵏclassmax2, Tᵏclassmax2, :maxγ, xinit[:,iTᵏclassmax]))
  println((:α, αset, :EMclassBayes, mean(classerEM_α), :EMclassmax, EMclassmax, :EMclassmax2, EMclassmax2, :maxγ, xinit[:,iEMclassmax]))
  println((:α, αset, :KMclassBayes, mean(classerKM_α), :KMclassmax, KMclassmax, :KMclassmax2, KMclassmax2, :maxγ, xinit[:,iKMclassmax]))

  allloss_α = hcat(Float64.(lossTk_α), Float64.(lossEM_α), Float64.(lossKM_α))
  allclasser_α = hcat(Float64.(classerTk_α), Float64.(classerEM_α), Float64.(classerKM_α))

  maxlosses_α = Float64.([Tᵏlmax2, EMlmax2, KMlmax2])
  maxclasser_α = Float64.([Tᵏclassmax2, EMclassmax2, KMclassmax2])

  return allloss_α, allclasser_α, maxlosses_α, maxclasser_α
end

## -----------------------------------------------------------------------------
## GRID INTERROGATION #1 (for best model, max and mean risks based on new batch size and new # of random γ)
## -----------------------------------------------------------------------------
# srand(12345);
# allloss_alpha00, allclasser_alpha00, maxlosses_alpha00, maxclasser_alpha00 = evalTᵏRisksGrid_debias(xgridγ, 0.0f0);
# 220.674245 seconds for 10x10 grid
# srand(12345);
# allloss_alpha01, allclasser_alpha01, maxlosses_alpha01, maxclasser_alpha01 = evalTᵏRisksGrid_debias(xgridγ, 0.1f0);
# srand(12345);
# allloss_alpha02, allclasser_alpha02, maxlosses_alpha02, maxclasser_alpha02 = evalTᵏRisksGrid_debias(xgridγ, 0.2f0);
# srand(12345);
# allloss_alpha03, allclasser_alpha03, maxlosses_alpha03, maxclasser_alpha03 = evalTᵏRisksGrid_debias(xgridγ, 0.3f0);
# srand(12345);
# allloss_alpha04, allclasser_alpha04, maxlosses_alpha04, maxclasser_alpha04 = evalTᵏRisksGrid_debias(xgridγ, 0.4f0);
srand(12345);
allloss_alpha05, allclasser_alpha05, maxlosses_alpha05, maxclasser_alpha05 = evalTᵏRisksGrid_debias(xgridγ, 0.5f0);

estorders = ["Tk", "EM", "KM"]

# writedir = "./MixtureClassify_n10_benz2c/risks/"
writedir = "./"
JLD.save(joinpath(writedir, "interrogate_all_maxdebias_g100x100_alpha05.jld"),
  "xgridgamma", xgridγ,

  "allloss_alpha05", allloss_alpha05,

  "allclasser_alpha05", allclasser_alpha05,

  "maxlosses_alpha05", maxlosses_alpha05,

  "maxclasser_alpha05", maxclasser_alpha05,

  "estorders", estorders
  )

## -----------------------------------------------------------------------------
## Save all risks tables in single JLD file
## -----------------------------------------------------------------------------
# writedir = "./"
# res00 = JLD.load(joinpath(writedir, "interrogate_all_maxdebias_g100x100_alpha00.jld"))
# res01 = JLD.load(joinpath(writedir, "interrogate_all_maxdebias_g100x100_alpha01.jld"))
# res02 = JLD.load(joinpath(writedir, "interrogate_all_maxdebias_g100x100_alpha02.jld"))
# res03 = JLD.load(joinpath(writedir, "interrogate_all_maxdebias_g100x100_alpha03.jld"))
# res04 = JLD.load(joinpath(writedir, "interrogate_all_maxdebias_g100x100_alpha04.jld"))
# res05 = JLD.load(joinpath(writedir, "interrogate_all_maxdebias_g100x100_alpha05.jld"))
#
# JLD.save(joinpath(writedir, "interrogate_all_maxdebias_g100x100.jld"),
#   "xgridgamma", res00["xgridgamma"],
#
#   "allloss_alpha00", res00["allloss_alpha00"],
#   "allloss_alpha01", res01["allloss_alpha01"],
#   "allloss_alpha02", res02["allloss_alpha02"],
#   "allloss_alpha03", res03["allloss_alpha03"],
#   "allloss_alpha04", res04["allloss_alpha04"],
#   "allloss_alpha05", res05["allloss_alpha05"],
#
#   "allclasser_alpha00", res00["allclasser_alpha00"],
#   "allclasser_alpha01", res01["allclasser_alpha01"],
#   "allclasser_alpha02", res02["allclasser_alpha02"],
#   "allclasser_alpha03", res03["allclasser_alpha03"],
#   "allclasser_alpha04", res04["allclasser_alpha04"],
#   "allclasser_alpha05", res05["allclasser_alpha05"],
#
#   "maxlosses_alpha00", res00["maxlosses_alpha00"],
#   "maxlosses_alpha01", res01["maxlosses_alpha01"],
#   "maxlosses_alpha02", res02["maxlosses_alpha02"],
#   "maxlosses_alpha03", res03["maxlosses_alpha03"],
#   "maxlosses_alpha04", res04["maxlosses_alpha04"],
#   "maxlosses_alpha05", res05["maxlosses_alpha05"],
#
#   "maxclasser_alpha00", res00["maxclasser_alpha00"],
#   "maxclasser_alpha01", res01["maxclasser_alpha01"],
#   "maxclasser_alpha02", res02["maxclasser_alpha02"],
#   "maxclasser_alpha03", res03["maxclasser_alpha03"],
#   "maxclasser_alpha04", res04["maxclasser_alpha04"],
#   "maxclasser_alpha05", res05["maxclasser_alpha05"],
#
#   "estorders", res00["estorders"]
#   )
