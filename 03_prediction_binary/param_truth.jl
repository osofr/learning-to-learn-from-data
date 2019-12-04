## ---------------------------------------------------------------------------------------
## Def functions for evaluation of the truth (Ψ₀) for given parameters γ
## ---------------------------------------------------------------------------------------
Ψ2dim_μ_σ(γ::Any) = γ[1], γ[2] ## define param as [μ,σ] in N(μ,σ)

Ψμ(γ::Any) = γ[1] ## define param as μ in N(μ,σ)
Ψσ(γ::Any) = γ[2] ## define param as σ in N(μ,σ)
Ψμ_σ(γ::Any) = γ[1] ./ γ[2] ## define param as μ/σ in N(μ,σ)

Ψσcentered(γ::Any) = γ[2]-2.5 ## define param as σ-2,5 in N(μ,σ)
Ψ_lognorm_mean(γ::Any) = exp.(γ[1] .+ γ[2].^2 / 2) ## mean in exp(N(μ,σ)) [log-normal]: exp(μ + σ^2/2)
Ψ_lognorm_median(γ::Any) = exp.(γ[1]) ## median in log-normal
Ψident(γ::Any) = γ

## to pull up a fun based on symbol:
# fΨ = getfield(current_module(), symb)
