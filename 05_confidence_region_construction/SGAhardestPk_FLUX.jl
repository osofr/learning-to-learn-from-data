using Flux

## a) SGA for worst-case coverage, i.e., 1 - sup_P P{Psi(P) is not in T(X)}
function SGAlossPᵏtypeIerr(mTᵏ,γ,ψ₀,predTᵏ,z1n,modelP,paramCI)
  ProbΨoutCI₀, ProbΨoutCI, ProbΨoutCITᵏ = coverageTᵏ(ψ₀,predTᵏ[1]...)
  return -mean(ProbΨoutCI₀)
end

# b) SGA for largest CI region, i.e., Worst-case size: sup_P E_P[pi(T(X))/σᵏ]
function meanCIlen1γ(μ1,μ2,σ1,σ2,γ,epoch;kmax=1.0f0,krate=0.0001f0)
  expk = Float32(min(kmax, krate*epoch))
  return mean(abs.(μ2 .- μ1) .* abs.(σ2 .- σ1) ./ (Ψσ(γ)^(expk)))
end

function SGAlossPᵏCIlenScaled(mTᵏ,γ,ψ₀,predTᵏ,z1n,modelP,paramCI,epoch;kmax=1.0f0,krate=0.0001f0)
  R1_PπTx_scaled =  meanCIlen1γ(predTᵏ[1]...,γ,epoch,kmax=kmax,krate=krate) # Expected area of CI rectangle, scaled by √var(IC)
  return -R1_PπTx_scaled
end

# c) SGA for largest CI region, i.e., Worst-case size: sup_P E_P[pi(T(X))]
function SGAlossPᵏCIlen(mTᵏ,γ,ψ₀,predTᵏ,z1n,modelP,paramCI)
  R1_PπTx = mean(CIlen(predTᵏ[1]...))
  return -R1_PπTx
end

function SGAlossPᵏCIcent(mTᵏ,γ,ψ₀,predTᵏ,z1n,modelP,paramCI;kmax=1.0f0,losstypecent=:L2)
  R1_Cx = rCIcent(predTᵏ[2]...,γ,kcent=kmax,losstypecent=losstypecent)
  return -R1_Cx
end

function evalPᵏqLqU(γ,TᵏCx,TᵏOff,α)
  ## CI centers TODO: make this one-line (so that it works for any number of params)
  CX_μ = data(TᵏCx[1])
  CX_σ = data(TᵏCx[2])
  centLX_μ = CX_μ-Ψμ(γ)
  centLX_σ = CX_σ-Ψσ(γ)
  qL_μ = quantile(data(centLX_μ), 1-α/4)
  qL_σ = quantile(data(centLX_σ), 1-α/4)

  centUX_μ = Ψμ(γ)-CX_μ
  centUX_σ = Ψσ(γ)-CX_σ
  qU_μ = quantile(data(centUX_μ), 1-α/4)
  qU_σ = quantile(data(centUX_σ), 1-α/4)

  LX_μ = data(TᵏOff[1])
  UX_μ = data(TᵏOff[2])
  LX_σ = data(TᵏOff[3])
  UX_σ = data(TᵏOff[4])
  return LX_μ,UX_μ,LX_σ,UX_σ,qL_μ,qU_μ,qL_σ,qU_σ
end

## Tᵏ loss for CI offset [L(X)]
## Note that separate Pᵏ are used for L(X) and U(X)
## This means that this loss is best split into two losses (L(X),U(X))
function SGAlossPᵏoffsetLμ(mTᵏ,γL,ψ₀,predTᵏ,z1n,modelP,paramCI)
  α = paramCI.α
  η = predTᵏ[4]
  LX_μ,UX_μ,LX_σ,UX_σ,qL_μ,qU_μ,qL_σ,qU_σ = evalPᵏqLqU(γL,predTᵏ[2],predTᵏ[3],α)
  lossL_μ = (LX_μ .< qL_μ) .* (LX_μ .- qL_μ).^2 .+ η .* (LX_μ .> qL_μ) .* (LX_μ .- qL_μ).^2
  lossL_μ = -mean(lossL_μ)
  LX_μ_cov = round(mean(LX_μ .> qL_μ),3)
  return lossL_μ
end

function SGAlossPᵏoffsetLσ(mTᵏ,γL,ψ₀,predTᵏ,z1n,modelP,paramCI)
  α = paramCI.α
  η = predTᵏ[4]
  LX_μ,UX_μ,LX_σ,UX_σ,qL_μ,qU_μ,qL_σ,qU_σ = evalPᵏqLqU(γL,predTᵏ[2],predTᵏ[3],α)
  lossL_σ = (LX_σ .< qL_σ) .* (LX_σ .- qL_σ).^2 .+ η .* (LX_σ .> qL_σ) .* (LX_σ .- qL_σ).^2
  lossL_σ = -mean(lossL_σ)
  LX_σ_cov = round(mean(LX_σ .> qL_σ),3)
  return lossL_σ
end

## Tᵏ loss for CI offset [U(X)]
## Note that separate Pᵏ are used for L(X) and U(X)
## This means that this loss is best split into two losses (L(X),U(X))
function SGAlossPᵏoffsetUμ(mTᵏ,γU,ψ₀,predTᵏ,z1n,modelP,paramCI)
  α = paramCI.α
  η = predTᵏ[4]
  LX_μ,UX_μ,LX_σ,UX_σ,qL_μ,qU_μ,qL_σ,qU_σ = evalPᵏqLqU(γU,predTᵏ[2],predTᵏ[3],α)
  lossU_μ = (UX_μ .< qU_μ) .* (UX_μ .- qU_μ).^2 .+ η .* (UX_μ .> qU_μ) .* (UX_μ .- qU_μ).^2
  lossU_μ = -mean(lossU_μ)
  UX_μ_cov = round(mean(UX_μ .> qU_μ),3)
  return lossU_μ
end

function SGAlossPᵏoffsetUσ(mTᵏ,γU,ψ₀,predTᵏ,z1n,modelP,paramCI)
  α = paramCI.α
  η = predTᵏ[4]
  LX_μ,UX_μ,LX_σ,UX_σ,qL_μ,qU_μ,qL_σ,qU_σ = evalPᵏqLqU(γU,predTᵏ[2],predTᵏ[3],α)
  lossU_σ = (UX_σ .< qU_σ) .* (UX_σ .- qU_σ).^2 .+ η .* (UX_σ .> qU_σ) .* (UX_σ .- qU_σ).^2
  lossU_σ = -mean(lossU_σ)
  UX_σ_cov = round(mean(UX_σ .> qU_σ),3)
  return lossU_σ
end

## look-up the η value from paramCI.η1 or paramCI.η2
function findMaxRisksPᵏ2(mTᵏ,modelP::ParamModelPᵏ,paramCI::ParamCI,o,SGAtypes,nstarts=10,setη=:η1; multivarSGA=false,atype=Array{Float32},lrSGA=0.005f0,nbatch=200,nruns=0,ntest=2000)
  funCIlenScaled(mTᵏ,γ,ψ₀,predTᵏ,z1n,modelP,paramCI) = SGAlossPᵏCIlenScaled(mTᵏ,γ,ψ₀,predTᵏ,z1n,modelP,paramCI,o[:epoch],kmax=o[:kcent],krate=o[:kcent])
  funCIcent(mTᵏ,γ,ψ₀,predTᵏ,z1n,modelP,paramCI) = SGAlossPᵏCIcent(mTᵏ,γ,ψ₀,predTᵏ,z1n,modelP,paramCI,kmax=o[:kcent],losstypecent=o[:losstypecent])
  loss_funs = Dict(:typeIerr => SGAlossPᵏtypeIerr,
                   :len => SGAlossPᵏCIlen,
                   :lenScaled => funCIlenScaled,
                   :cent => funCIcent,
                   :offLμ => SGAlossPᵏoffsetLμ,
                   :offLσ => SGAlossPᵏoffsetLσ,
                   :offUμ => SGAlossPᵏoffsetUμ,
                   :offUσ => SGAlossPᵏoffsetUσ)
  SGAtypesRef = Dict()
  for i in 1:length(SGAtypes)
    push!(SGAtypesRef, SGAtypes[i] => i)
  end
  g1 = map(x -> istracked(x) && zero_grad!(grad(x)), params(mTᵏ))
  # try gc(); CuArrays.clearpool(); end

  T = Float32
  lrSGA=T(lrSGA)
  xcorners = corner(modelP)
  xborders = Any[]
  for x in xcorners
    push!(xborders, [x[1], rand(modelP, 1)[1][2]])
    push!(xborders, [rand(modelP, 1)[1][1], x[2]])
  end
  xinit = [xcorners; xborders; [mid(modelP)]; rand(modelP, nstarts)]
  xinit = hcat(xinit...)
  optvals = copy(xinit)
  optvals = convert(SharedArray, optvals)
  maxRisksALLγ = Array{T}(length(SGAtypes),size(xinit)[2])

  dβ = Distributions.Beta(paramCI.η_α,paramCI.η_β)
  η = Array{Float32}(nbatch)
  η = rand!(dβ, η)
  if (setη == :η1)
    η .= paramCI.η1
  elseif (setη == :η2)
    η .= paramCI.η2
  end
  println("-----------------------------------------------------------------")
  println((:η, setη, η[1]))
  println("-----------------------------------------------------------------")

  for i = 1:size(xinit)[2]
    maxRisksALLγ[:,i], optvals[:,i] = evalRandSearch2(mTᵏ,xinit[:,i],modelP,paramCI,loss_funs,SGAtypes,η;atype=atype,lrSGA=lrSGA,nbatch=nbatch,nruns=nruns)
  end
  maxRisksALLγ = data(maxRisksALLγ)
  optvals = data(optvals)

  maxRisks = map(row -> maximum(maxRisksALLγ[row,:]), 1:size(maxRisksALLγ)[1])
  indmaxRisks = map(row -> indmax(maxRisksALLγ[row,:]), 1:size(maxRisksALLγ)[1])
  ## array of worst γs (one γ for each loss), γs are by columns
  maxγs = hcat(map(colγ -> optvals[:,colγ], indmaxRisks)...)
  ## re-map γ into a vector of arrays[γ]
  maxγs = map(col -> maxγs[:,col], 1:size(maxγs)[2])

  maxRγsD = Dict()
  for i in 1:length(SGAtypes)
    push!(maxRγsD, SGAtypes[i] => (maxRisks[i], maxγs[i]))
  end

  g1 = map(x -> istracked(x) && zero_grad!(grad(x)), params(mTᵏ))
  return maxRisks, maxγs, maxRγsD, SGAtypesRef, maxRisksALLγ, optvals
end

function MaxRisksPᵏInterrogate(mTᵏ,modelP::ParamModelPᵏ,paramCI::ParamCI,o,SGAtypes,xinit,ηval=0.2f0; multivarSGA=false,atype=Array{Float32},lrSGA=0.005f0,nbatch=200,nruns=0,ntest=2000)
  funCIlenScaled(mTᵏ,γ,ψ₀,predTᵏ,z1n,modelP,paramCI) = SGAlossPᵏCIlenScaled(mTᵏ,γ,ψ₀,predTᵏ,z1n,modelP,paramCI,o[:epoch],kmax=o[:kcent],krate=o[:kcent])
  funCIcent(mTᵏ,γ,ψ₀,predTᵏ,z1n,modelP,paramCI) = SGAlossPᵏCIcent(mTᵏ,γ,ψ₀,predTᵏ,z1n,modelP,paramCI,kmax=o[:kcent],losstypecent=o[:losstypecent])
  loss_funs = Dict(:typeIerr => SGAlossPᵏtypeIerr,
                   :len => SGAlossPᵏCIlen,
                   :lenScaled => funCIlenScaled,
                   :cent => funCIcent,
                   :offLμ => SGAlossPᵏoffsetLμ,
                   :offLσ => SGAlossPᵏoffsetLσ,
                   :offUμ => SGAlossPᵏoffsetUμ,
                   :offUσ => SGAlossPᵏoffsetUσ)
  SGAtypesRef = Dict()
  for i in 1:length(SGAtypes)
    push!(SGAtypesRef, SGAtypes[i] => i)
  end
  g1 = map(x -> istracked(x) && zero_grad!(grad(x)), params(mTᵏ))

  T = Float32
  lrSGA=T(lrSGA)

  optvals = copy(xinit)
  optvals = convert(SharedArray, optvals)
  maxRisksALLγ = Array{T}(length(SGAtypes),size(xinit)[2])

  η = Array{Float32}(nbatch)
  η .= ηval
  println("-----------------------------------------------------------------")
  println((:η, η[1]))
  println("-----------------------------------------------------------------")

  for i = 1:size(xinit)[2]
    maxRisksALLγ[:,i], optvals[:,i] = evalRandSearch2(mTᵏ,xinit[:,i],modelP,paramCI,loss_funs,SGAtypes,η;atype=atype,lrSGA=lrSGA,nbatch=nbatch,nruns=nruns)
  end
  maxRisksALLγ = data(maxRisksALLγ)
  optvals = data(optvals)

  maxRisks = map(row -> maximum(maxRisksALLγ[row,:]), 1:size(maxRisksALLγ)[1])
  indmaxRisks = map(row -> indmax(maxRisksALLγ[row,:]), 1:size(maxRisksALLγ)[1])
  ## array of worst γs (one γ for each loss), γs are by columns
  maxγs = hcat(map(colγ -> optvals[:,colγ], indmaxRisks)...)
  ## re-map γ into a vector of arrays[γ]
  maxγs = map(col -> maxγs[:,col], 1:size(maxγs)[2])

  maxRγsD = Dict()
  for i in 1:length(SGAtypes)
    push!(maxRγsD, SGAtypes[i] => (maxRisks[i], maxγs[i]))
  end

  g1 = map(x -> istracked(x) && zero_grad!(grad(x)), params(mTᵏ))
  return maxRisks, maxγs, maxRγsD, SGAtypesRef, maxRisksALLγ, optvals
end

## Random search for hardest γ over many loss functions
function evalRandSearch2(mTᵏ,γ::AbstractVector{T},modelP::ParamModelPᵏ,paramCI::ParamCI,loss_funs,SGAtypes,η; atype=Array{Float32},lrSGA=0.005f0,nbatch=500,nruns=200) where T
  l = 0.0f0
  maxRisks = Any[]
  γ = copy(γ) |> gpu
  z1n = allocX1n(modelP, nbatch)
  ψ₀, predTᵏ = generateCIparams(γ,mTᵏ,z1n,η,modelP,paramCI)
  # for loss_fun in loss_funs
  for SGAtype in SGAtypes
    l = loss_funs[SGAtype](mTᵏ,γ,ψ₀,predTᵏ,z1n,modelP,paramCI)
    maxRisk = -data(l)[1]
    push!(maxRisks,maxRisk)
  end
  γ = data(γ)
  return maxRisks, γ
end

clip(γ, lbb, ubb) = min.(ubb, max.(lbb, γ))
