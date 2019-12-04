## ************************************************************************
## Coverage for 2 parameter rectangular CONFIDENCE REGION (CR).
## ************************************************************************
function coverage(ψ₀,μ,μlen,σ,σlen)
  LCIμ = μ
  μ₀inLCIμ_true = (ψ₀[1] .>= LCIμ)
  UCIμ = μlen
  μ₀inUCIμ_true = (ψ₀[1] .<= UCIμ)
  LCIσ = σ
  σ₀inLCIσ_true = (ψ₀[2] .>= LCIσ)
  UCIσ = σlen
  σ₀inUCIσ_true = (ψ₀[2] .<= UCIσ)

  ψ₀outCI_true = 1 .- (μ₀inLCIμ_true .* μ₀inUCIμ_true .* σ₀inLCIσ_true .* σ₀inUCIσ_true)

  ProbΨoutCI₀ = mean(ψ₀outCI_true)
  ProbΨoutCI = ProbΨoutCI₀
  return ProbΨoutCI₀, ProbΨoutCI
end

## ************************************************************************
## Aprox. Tᵏ loss for coverage with relu.(), similar to hinge loss
## Loss as an indicator has gradient 0
## approximating the indicators by relu.(⋅) instead
## For Tᵏ that outputs confidence region (rectangle for 2 parameters)
## ************************************************************************
function coverageTᵏ(ψ₀,μ,μlen,σ,σlen)
  c1 = 0.1f0
  c2 = 0.05f0

  LCIμ = μ
  μ₀inLCIμ_true = (ψ₀[1] .>= LCIμ)
  UCIμ = μlen
  μ₀inUCIμ_true = (ψ₀[1] .<= UCIμ)
  LCIσ = σ
  σ₀inLCIσ_true = (ψ₀[2] .>= LCIσ)
  UCIσ = σlen
  σ₀inUCIσ_true = (ψ₀[2] .<= UCIσ)
  ψ₀outCI_true = 1 .- (μ₀inLCIμ_true .* μ₀inUCIμ_true .* σ₀inLCIσ_true .* σ₀inUCIσ_true)

  ProbΨoutCI₀ = mean(ψ₀outCI_true)
  ProbΨoutCI = ProbΨoutCI₀

  μ₀outUCIμ = relu.(ψ₀[1] .- UCIμ  .+ c1) ./ c1
  μ₀outLCIμ = relu.(LCIμ .-  ψ₀[1] .+ c1) ./ c1
  σ₀outUCIσ = relu.(ψ₀[2] .- UCIσ  .+ c2) ./ c2
  σ₀outLCIσ = relu.(LCIσ .-  ψ₀[2] .+ c2) ./ c2

  ψ₀outCITᵏ = max.((μ₀outUCIμ .+ μ₀outLCIμ), (σ₀outUCIσ .+ σ₀outLCIσ))
  ProbΨoutCITᵏ = mean(ψ₀outCITᵏ)
  return ProbΨoutCI₀, ProbΨoutCI, ProbΨoutCITᵏ
end


## ************************************************************************
## Aprox. coverage for 1 parameter CONFIDENCE INTERVAL (CI).
## ************************************************************************
function coverage(ψ₀,μ,μlen)
  h = 0.5f0 ## kernel bandwidth
  LCIμ = μ
  μ₀inLCIμ_true = (ψ₀ .>= LCIμ)
  UCIμ = μlen
  μ₀inUCIμ_true = (ψ₀ .<= UCIμ)
  ψ₀outCI_true = 1 .- (μ₀inLCIμ_true .* μ₀inUCIμ_true)
  ProbΨoutCI₀ = mean(ψ₀outCI_true)
  ProbΨoutCI = ProbΨoutCI₀
  return ProbΨoutCI₀, ProbΨoutCI
end

## ************************************************************************
## Aprox. Tᵏ loss for coverage with relu.(), similar to hinge loss
## For Tᵏ that outputs regular CI (bounds on 1 parameter)
## ************************************************************************
function coverageTᵏ(ψ₀,μ,μlen)
  c1 = 0.5f0
  LCIμ = μ
  μ₀inLCIμ_true = (ψ₀ .>= LCIμ)
  UCIμ = μlen
  μ₀inUCIμ_true = (ψ₀ .<= UCIμ)
  ψ₀outCI_true = 1 .- (μ₀inLCIμ_true .* μ₀inUCIμ_true)
  ProbΨoutCI₀ = mean(ψ₀outCI_true)
  ProbΨoutCI = ProbΨoutCI₀
  μ₀outUCIμ = relu.(ψ₀ .- UCIμ  .+ c1) ./ c1
  μ₀outLCIμ = relu.(LCIμ .-  ψ₀ .+ c1) ./ c1
  ψ₀outCITᵏ = μ₀outUCIμ .+ μ₀outLCIμ
  ProbΨoutCITᵏ = mean(ψ₀outCITᵏ)
  return ProbΨoutCI₀, ProbΨoutCI, ProbΨoutCITᵏ
end
