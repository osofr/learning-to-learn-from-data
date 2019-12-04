## CI size for 2-dim parameter for Tᵏ(X)

CIlen(μ1,μ2,σ1,σ2) = abs.(μ2 .- μ1) .* abs.(σ2 .- σ1)
meanCIlen(μ1,μ2,σ1,σ2) = mean(abs.(μ2 .- μ1) .* abs.(σ2 .- σ1))

function meanCIlen(μ1,μ2,σ1,σ2,γ,epoch;kmax=1.0f0,krate=0.0001f0)
  expk = Float32(min(kmax, krate*epoch))
  return mean(abs.(μ2 .- μ1) .* abs.(σ2 .- σ1) ./ (Ψσ(γ) ^ (expk)))
end

## 1-dim parameter (true interval) for Tᵏ(X)
meanCIlen(μ1,μ2) = mean(abs.(μ2-μ1))
# re-scale by the true log-normal mean
function meanCIlen(μ1,μ2,γ)
  σ² = Ψσ(γ).^2
  mean(abs.(μ2-μ1) ./ .√((σ².^2)./2 .+ σ²))
end

## when cent=0, not adding l2-norm diff b/ween γ and CI center to the risk
## when covγ=0, CI size risk is evaluated across all intervals (all batches)
function meanCIlen(μ1,μ2,σ1,σ2,γ;kmax=2.0f0,covγ=1,cent=1,λcent=1.0f0)
  ## vector of unscaled CI sizes (nbatch)
  CIsizeRisk = abs.(μ2 .- μ1) .* abs.(σ2 .- σ1)
  ## vector of indicators that gamma is covered by Tᵏ(X) (nbatch)
  IγinTᵏ = (γ[1] .>= μ1) .* (γ[1] .<= μ2) .* (γ[2] .>= σ1) .* (γ[2] .<= σ2)
  ## number of intervals that cover gamma (nbatch)
  ncovγ = (covγ==1) ? max(sum(IγinTᵏ), 1) : length(IγinTᵏ)
  ## vector of risks (nbatch) for CI size, rescaled by number of covered intervals
  CIsizeRisk = CIsizeRisk ./ ncovγ
  ## vector of risks (nbatch) for CI size multiplied by indicator of Tᵏ(X) covering γ
  if (covγ==1) CIsizeRisk = CIsizeRisk .* IγinTᵏ; end
  ## vector of risks for CI center (L2-norm distance of each gamma from CI estimated center)
  CIcentRisk = (((μ1 .+ μ2) ./ 2) .- γ[1]).^2 .+ (((σ1 .+ σ2) ./ 2) .- γ[2]).^2
  ## scaling factors for each risk
  if (kmax==0.0f0)
    println("kmax: $kmax")
    CIsizeRisk_norm = sum(CIsizeRisk) ## rescale each risk by σ₀^(kmax)
    CIcentRisk_norm = mean(CIcentRisk)
  elseif (kmax==1.0f0)
    println("kmax: $kmax")
    CIsizeRisk_norm = sum(CIsizeRisk ./ (Ψσ(γ))) ## rescale each risk by σ₀^(kmax)
    CIcentRisk_norm = mean(CIcentRisk ./ (Ψσ(γ)))
  elseif (kmax==2.0f0)
    println("kmax: $kmax")
    CIsizeRisk_norm = sum(CIsizeRisk ./ (Ψσ(γ).^2.0f0)) ## rescale each risk by σ₀^(kmax)
    CIcentRisk_norm = mean(CIcentRisk ./ (Ψσ(γ).^2.0f0))
  elseif (kmax==0.5f0)
    println("kmax: $kmax")
    CIsizeRisk_norm = sum(CIsizeRisk ./ (sqrt.(Ψσ(γ)))) ## rescale each risk by σ₀^(kmax)
    CIcentRisk_norm = mean(CIcentRisk ./ (sqrt.(Ψσ(γ))))
  else
    error("only kmax=0.5, 1.0 or 2.0 are supported on GPU")
  end

  loss = (cent==1) ? (CIsizeRisk_norm + λcent*CIcentRisk_norm) : CIsizeRisk_norm
  println((:ncovγ, ncovγ, :P_γCov, round(mean(IγinTᵏ),2),
           :CIsize, round(data(sum(CIsizeRisk)),3), :CIcent, round(data(mean(CIcentRisk)),3),
           :CIsize_norm, round(data(CIsizeRisk_norm),3),
           :λCIcen_norm, round(data(λcent*CIcentRisk_norm),3),
           :SizeLoss, round(data(loss),3)
           ))

  return loss, CIsizeRisk_norm, λcent*CIcentRisk_norm
end
