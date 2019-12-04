using Distributions

function CRglobalNormal(X,γ)
    ## define individual α as equal via equation γ = (1-α1)*(1-α2)
    ## confidence level for CI1 (for μ) and CI2 (for σ)
    n = length(X)
    dχ² = Distributions.Chisq(n-1)
    dN = Distributions.Normal(0,1)
    α1 = α2 = 1-√(1-γ) # (1-α1)*(1-α2) ## check roughly equals γ
    a = Distributions.cquantile(dN, α1/2) ## upper α1/2 percentile of N(0,1)
    b = Distributions.quantile(dχ²,α2/2) ## lower α2/2 percentile of χ²(n-1)
    c = Distributions.cquantile(dχ²,α2/2) ## upper α2/2 percentile (1-α2/2) of χ²(n-1)
    ## lower / upper bounds on σ^2:
    # lbσ² = ns²/c
    # ubσ² = ns²/b
    ## lower / upper bounds on σ:
    X̄ = mean(X)
    ns² = sum(abs2, X - X̄) # ns² = ∑(Xᵢ-X̄)², same as: n*var(X, corrected=false)
    lbσ = √(ns²/c)
    ubσ = √(ns²/b)
    ## lower / upper bounds on  μ under ubσ:
    lbμ = X̄ - a * (ubσ / √n)
    ubμ = X̄ + a * (ubσ / √n)
    return [lbμ,ubμ,lbσ,ubσ]
end
