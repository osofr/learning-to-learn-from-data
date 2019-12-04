
## sample normal on CPU, then move array to GPU, for single γ
function sim_norm_gpu!(γ, z1n)
  randn!(z1n)
  z1n = z1n |> gpu
  return (z1n' .* γ[2] .+ γ[1])'
end
## sample normal on CPU, then move array to GPU, for an array of γs (Piᵏ output)
function sim_norm_gpu!(γ::AbstractArray{T,1}, z1n) where {T<:Real}
  randn!(z1n)
  z1n = z1n |> gpu ##TODO: make it more parsimonious, right now not using GPU for SGA
  return z1n * γ[2] + γ[1]
end
## gpu-based normal sampling, ends up slightly slower than above, so not being used
function sim_norm_gpu_fast!(γ, z1n)
  z1n = CuArrays.CURAND.randn(Float32, size(z1n)...)
  return (z1n' .* γ[2] .+ γ[1])'
end
# gpu-based normal sampling
function sim_norm_gpu_fast!(γ::AbstractArray{T,1}, z1n) where {T<:Real}
  z1n = CuArrays.CURAND.randn(Float32, size(z1n)...)
  return z1n * γ[2] + γ[1]
end

## generates CIs for given mPi (sample prior in response to noice)
function generateCIparams(mPi,noise,mT,z1n,η,modelP,paramCI)
  γ = predictPiᵏ(mPi,noise,modelP,paramCI)
  generateCIparams(γ,mT,z1n,η,modelP,paramCI)
end

## use pre-generated parameters γ to simulate data x1n~γ
## then evaluate estimator Tᵏ(x1n)
function generateCIparams(γ,mT,z1n,η,modelP,paramCI)
  x1n = sim_norm_gpu!(γ,z1n)
  x1n = reshape(x1n, 1, size(x1n,1), size(x1n,2))
  predTᵏ = mT(x1n,η)
  ψ₀ = Ψ(modelP,γ)
  return ψ₀, predTᵏ
end

## generates CI centers for given mPi (sample prior in response to noice)
function generateCIcent(mPi,noise,mT,z1n,modelP,paramCI)
  γ = predictPiᵏ(mPi,noise,modelP,paramCI)
  generateCIcent(γ,mT,z1n,modelP,paramCI)
end
## use pre-generated parameters γ to simulate data x1n~γ
## then evaluate estimator Tᵏ(x1n) for the CI centers
function generateCIcent(γ,mT,z1n,modelP,paramCI)
  x1n = sim_norm_gpu!(γ,z1n)
  x1n = reshape(x1n, 1, size(x1n,1), size(x1n,2))
  predTᵏcent = predictTᵏcent(mT,x1n)
  ψ₀ = Ψ(modelP,γ)
  return ψ₀, predTᵏcent
end

## generates CI offsets for given mPi (sample prior in response to noice)
function generateCIoff(mPi,noise,mT,z1n,η,modelP,paramCI)
  γ = predictPiᵏ(mPi,noise,modelP,paramCI)
  generateCIoff(γ,mT,z1n,η,modelP,paramCI)
end
## use pre-generated parameters γ to simulate data x1n~γ
## then evaluate estimator Tᵏ(x1n) for the CI offsets
function generateCIoff(γ,mT,z1n,η,modelP,paramCI)
  x1n = sim_norm_gpu!(γ,z1n)
  x1n = reshape(x1n, 1, size(x1n,1), size(x1n,2))
  predTᵏoff = predictTᵏoff(mT,x1n,η)
  ψ₀ = Ψ(modelP,γ)
  return ψ₀, predTᵏoff
end
