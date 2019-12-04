## ---------------------------------------------------------------------------------------
## Def parameters for CI problem
## ---------------------------------------------------------------------------------------
struct ParamCI{T,F,D}
    bounds::ParamBox{T}
    nparams::Int        ## number of parameters for which CIs are built (e.g, 1 gives the usual confidence interval)
    α::F                ## 1-coverage prob
    η1::F               ## 1st η for Tᵏ stats -- tuning parameter for CI size risk (when CI covers Ψ₀, the risk η*CIsize kicks in)
    η2::F               ## 2nd η for Tᵏ stats -- tuning parameter for CI size risk (when CI covers Ψ₀, the risk η*CIsize kicks in)
    η_α::F              ## Tuning parameter for CI size risk (when CI covers Ψ₀, the risk η*CIsize kicks in)
    η_β::F              ## Tuning parameter for CI size risk (when CI covers Ψ₀, the risk η*CIsize kicks in)
    λ::F                ## constant infating coverage risk term: R2_cov = λ(ProbΨoutCI - α)^2
    λMC::F              ## penalty for mode collapse (loosing coverage wrt diffuse prior)
    δ::D                ## extra offsets added to the edges when rescaling the centers C(X) to the range of the parameter space
    trtype::Symbol
end
@forward ParamCI.bounds bounds, lb, ub, mid, border, linspace, rand
@forward ParamCI.bounds Base.getindex, Base.first, Base.last, Base.endof, Base.push!, Base.length
@forward ParamCI.bounds Base.start, Base.next, Base.done

## define constructor that ensures type consistency (convert each bound to same types as params)
function ParamCI(bounds::ParamBox{T},nparams::Int; α::F=0.05f0,η1::F=0.20f0,η2::F=0.50f0,η_α::F=1.0f0,η_β::F=4.0f0,λ::F=50.0f0,λMC::F=100.0f0,δ::D=[0.5f0,0.2f0],trtype=:linear) where {T,F,D}
    ParamCI(bounds,nparams,α,η1,η2,η_α,η_β,λ,λMC,δ,trtype)
end
