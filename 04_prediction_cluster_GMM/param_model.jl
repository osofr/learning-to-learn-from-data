## ---------------------------------------------------------------------------------------
## Classes that define an object for parmetric models.
## This includes the number of parameters in the model, the bounds of the parameter space,
## the definition of the true parameter (target of estimation) and functionality for
## simulating data from a given parametric model.
## ---------------------------------------------------------------------------------------

using MacroTools: @forward

## ---------------------------------------------------------------------------------------
## Def bounds on the parameter space
## ---------------------------------------------------------------------------------------
struct ParamBox{T}
    bounds::Vector{T}
end

@forward ParamBox.bounds Base.getindex, Base.first, Base.last, Base.endof, Base.push!, Base.length
@forward ParamBox.bounds Base.start, Base.next, Base.done

bounds(γ::ParamBox, idx::Vector) = map((vec, idx) -> vec[idx], γ, idx)
bounds(γ::ParamBox, idx::Int) = bounds(γ, fill(idx, length(γ)))
lb(γ::ParamBox) = bounds(γ,1)
ub(γ::ParamBox) = bounds(γ,2)
mid(γ::ParamBox) = (lb(γ) + ub(γ)) / 2
border(γ::ParamBox) = [lb(γ),
                       ub(γ),
                       mid(γ),
                       bounds(γ, rand([1,2], 3)),
                       bounds(γ, rand([1,2], 3)),
                       bounds(γ, rand([1,2], 3)),
                       bounds(γ, rand([1,2], 3))
                       ]

## parameter space transformations
import Base.linspace
linspace(γ::ParamBox;step=0.01) = map(x -> x[1]:eltype(γ[1])(step):x[2], γ.bounds)
lintrans(x, lb, ub) = x*(ub-lb)+lb

import Base.rand
function rand(γ::ParamBox, nvals; step=0.01)
    range = linspace(γ,step=step)
    x = map(x -> rand(x,nvals), range)
    x = hcat(x...)
    return [x[i,:] for i=1:nvals]
end

## ---------------------------------------------------------------------------------------
## Def model / parameter
## ---------------------------------------------------------------------------------------
abstract type ModelPᵏ end
abstract type NonParamModelPᵏ <: ModelPᵏ end
# abstract type ParamModelPᵏ <: NonParamModelPᵏ end

struct ParamModelPᵏ{T} <: ModelPᵏ
    bounds::ParamBox{T}
    Ψ::Symbol           # symbol for function name that evalutes the truth (under same name)
    name::Symbol        # name of the data-gen fun for simulating 𝐗
    xdim::Int           # dimension of input data 𝐗 ↦ T(𝐗), for sample 𝐗=(Xᵢ:i=1,…,n)
end
@forward ParamModelPᵏ.bounds bounds, lb, ub, mid, border, linspace, rand
@forward ParamModelPᵏ.bounds Base.getindex, Base.first, Base.last, Base.endof, Base.push!, Base.length
@forward ParamModelPᵏ.bounds Base.start, Base.next, Base.done

## define constructor that ensures type consistency (convert each bound to same types as params)
function ParamModelPᵏ(bounds::ParamBox{T}; Ψ::Symbol=Ψμ,name::Symbol=:norm,xdim::Int=10) where T
    ParamModelPᵏ(bounds,Ψ,name,xdim)
end

## clip parameter vector to stay within the bounds, by ref
## only works when γ is a single parameter value (not a vector of params)
function clip!(γ, modelP::ParamModelPᵏ)
   γ .= min.(ub(modelP), max.(lb(modelP), γ))
   return γ
end
## re-allocate γ (use for autodiff or GPU)
function clip(γ, modelP::ParamModelPᵏ)
   γ = min.(ub(modelP), max.(lb(modelP), γ))
   return γ
end

function allocX1n(modelP::ParamModelPᵏ, nbatch::Int; atype=Array{Float32})
    convert(atype, randn(Float32, modelP.xdim, nbatch))
end
