## ---------------------------------------------------------------------------------------
## Def functions for evaluation of the truth (Ψ₀) for given input parameters γ
## ---------------------------------------------------------------------------------------
## truth is the 1st 2 params (e.g., [μ,σ] in 2-param model P~N(μ,σ))
Ψ2dim_μ_σ(γ::Any) = γ[1], γ[2]
## truth is the 1st param (e.g., μ in 2-param model P=N(μ,σ))
Ψμ(γ::Any) = γ[1]
## truth is the 2nd param (e.g., σ in 2-param model P=N(μ,σ))
Ψσ(γ::Any) = γ[2]
## truth is ratio of 1st and 2nd param (e.g., μ/σ in P=N(μ,σ))
Ψμ_σ(γ::Any) = γ[1] ./ γ[2]
## truth is lognormal mean, i.e., exp(μ + σ^2/2) for (μ=γ[1],σ=γ[1]);
## assumes model P is x1n~N(μ,σ^2);
## the log-normally distr data is obtained as y1n=exp(x1n)~exp(N(μ,σ));
Ψ_lognorm_mean(γ::Any) = γ[1] .+ (γ[2].^2.0f0 ./ 2)
##  median in log-normal model for μ=γ[1] and P=N(μ,σ)
Ψ_lognorm_median(γ::Any) = exp.(γ[1])
